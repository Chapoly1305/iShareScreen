"""NVDEC -> GPU RGB -> EDSR -> GPU1080p, with no host pixel handoff.

An integration benchmark, not a live viewer backend. Demux is outside timing;
compressed packet submission, GPU decode/color conversion, DLPack handoff,
normalization, inference, postprocessing and completion wait are included.
Network, frame pacing, presentation and encoding are excluded. The source is
a frozen HEVC 720p/600K/4:2:0 validation clip without B frames.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import triton
import triton.language as tl
import PyNvVideoCodec as nvc

from benchmarks.super_resolution.profile_edsr_tiles import Engine
from benchmarks.super_resolution.profile_fused_pipeline import pack_area_kernel
from benchmarks.super_resolution.profile_inference_optimization import digest, guard, save, stats


@triton.jit
def normalize_rgbp(X, Y, SC: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
                   N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c = i // (720 * 1280)
    y = (i // 1280) % 720
    x = i % 1280
    value = tl.load(X + c * SC + y * SH + x * SW, i < N, 0).to(tl.float32)
    tl.store(Y + i, value / 255.0, i < N)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--capture", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    result = dict(complete=False, started=time.time(), gpu=guard(), scope=__doc__,
        nvc_version=nvc.__version__, engine_sha256=digest(a.engine),
        script_sha256=digest(Path(__file__)), samples=a.samples, warmup_per_decode_session=50,
        profiler=a.capture, pixel_copy_validation=[], postprocess_validation=[])
    save(a.output, result)
    try:
        runs = [r for r in json.loads((a.validation / "INPUTS.json").read_text())["runs"] if r["config"] == "720p-600k-420"]
        run = runs[0]
        assert run["stream"]["has_b_frames"] == 0
        path = a.validation / run["video"]
        assert digest(path) == run["video_sha256"]
        result["source"] = {k: run[k] for k in ("clip", "video", "video_sha256", "stream")}
        stream = torch.cuda.Stream(); torch.cuda.set_stream(stream)
        engine = Engine(a.engine)
        engine.input.zero_()
        output = torch.empty((1080, 1920, 3), device="cuda", dtype=torch.uint8)
        def compute():
            y = engine.run()
            pack_area_kernel[(triton.cdiv(output.numel(), 256),)](y, output, *y.stride()[1:], output.numel(), 256, enable_fp_fusion=False)
        for _ in range(5):
            compute()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            compute()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        wall, device, decode_wall, handoff_wall = [], [], [], []
        passes = 0
        while len(wall) < a.samples:
            demuxer = nvc.CreateDemuxer(filename=str(path))
            decoder = nvc.CreateDecoder(gpuid=0, codec=demuxer.GetNvCodecId(), cudacontext=0,
                cudastream=stream.cuda_stream, usedevicememory=True, outputColorType=nvc.OutputColorType.RGBP,
                latency=nvc.DisplayDecodeLatencyType.LOW)
            decoded = 0; captured = False
            for packet in demuxer:
                if decoded == 50 and a.capture and not captured:
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push("phase_nvdec_gpu")
                    captured = True
                torch.cuda.nvtx.range_push("frame_nvdec_gpu")
                t0 = time.perf_counter()
                frames = decoder.Decode(packet)
                t1 = time.perf_counter()
                if not frames:
                    torch.cuda.nvtx.range_pop()
                    continue
                assert len(frames) == 1, "LOW latency on this no-B-frame source must not batch frames"
                frame = frames[0]
                tensor = torch.from_dlpack(frame)
                t2 = time.perf_counter()
                assert tensor.device.type == "cuda" and tensor.dtype == torch.uint8
                assert tuple(tensor.shape) == (3, 720, 1280), (tensor.shape, tensor.stride())
                assert tensor.data_ptr() == frame.GetPtrToPlane(0), "DLPack unexpectedly copied GPU pixels"
                begin.record()
                normalize_rgbp[(triton.cdiv(engine.input.numel(), 256),)](tensor, engine.input, *tensor.stride(), engine.input.numel(), 256)
                graph.replay(); end.record(); end.synchronize()
                t3 = time.perf_counter()
                torch.cuda.nvtx.range_pop()
                if decoded >= 50:
                    wall.append((t3-t0)*1000); device.append(begin.elapsed_time(end))
                    decode_wall.append((t1-t0)*1000); handoff_wall.append((t2-t1)*1000)
                elif decoded in (0, 24, 49):
                    # Downloads here are correctness checks before capture/timing.
                    # The regular frame loop keeps all decoded/output pixels CUDA.
                    expected = tensor.float().unsqueeze(0) / 255
                    assert torch.equal(engine.input, expected)
                    native = engine.output[0].mul(255).round().clamp(0,255).byte().permute(1,2,0).contiguous().cpu().numpy()
                    reference = cv2.resize(native, (1920,1080), interpolation=cv2.INTER_AREA)
                    actual = output.cpu().numpy()
                    assert np.array_equal(reference, actual)
                    result["pixel_copy_validation"].append(dict(pass_index=passes, frame=decoded,
                        shape=list(tensor.shape), strides=list(tensor.stride()), pointer_shared=True,
                        cuda_array_interface=str(frame.cuda()), normalization_exact=True))
                    result["postprocess_validation"].append(dict(pass_index=passes, frame=decoded, max_rgb=0))
                decoded += 1
                # Keep both frame and imported tensor alive until all consumers finish.
                del tensor, frame, frames
                if len(wall) >= a.samples:
                    break
            if captured:
                torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
            assert decoded > 50
            passes += 1
            del decoder, demuxer
        result.update(complete=True, finished=time.time(), gpu_end=guard(), passes=passes,
            timings=dict(packet_to_gpu_output_wall=stats(wall), rgb_handoff_to_output_cuda=stats(device),
                         decode_api_wall=stats(decode_wall), dlpack_handoff_wall=stats(handoff_wall)),
            quality_scope="Shared-memory handoff and normalization exact; same-engine GPU postprocessing pixel-exact. Decoder RGB color conversion has NOT been equated to the frozen FFmpeg RGB inputs; this run is not a new 72-frame PSNR quality gate.")
        save(a.output, result)
        print(json.dumps({k: v["p95_ms"] for k,v in result["timings"].items()}, indent=2), flush=True)
    except BaseException as exc:
        result.update(error=repr(exc), failed=time.time()); save(a.output, result); raise


if __name__ == "__main__":
    main()
