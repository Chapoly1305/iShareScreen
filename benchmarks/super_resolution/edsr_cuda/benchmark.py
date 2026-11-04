"""Actual NVDEC frames through the bounded framework GPU-surface interface."""

import argparse
import inspect
import os
from benchmarks.super_resolution.edsr_cuda.telemetry import DeviceTelemetry
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
import PyNvVideoCodec as nvc
from isharescreen.enhancement.cuda_decoder import NvdecDecoder as Decoder

from isharescreen.enhancement.cuda_native import (
    NativeCudaSurfaceEDSR as CudaSurfaceEDSR,
)

ROOT = Path(__file__).resolve().parent
from benchmarks.super_resolution.edsr_cuda.common import (
    Engine,
    digest,
    guard,
    save,
    stats,
    reference_resize,
)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--capture", action="store_true")
    p.add_argument("--native-library", type=Path, required=True)
    p.add_argument("--expected-gpu-uuid", required=True)
    p.add_argument("--mode", choices=("inline", "prearmed"), default="inline")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--backend", choices=("poll",), default="poll")
    p.add_argument(
        "--input-format", choices=("nv12_bt709_full",), default="nv12_bt709_full"
    )
    p.add_argument("--decoder-latency", choices=("LOW",), default="LOW")
    p.add_argument("--engine-sha256", required=True)
    a = p.parse_args()
    os.environ["ISS_NVDEC_LIBRARY"] = str(a.native_library.resolve())
    os.environ["ISS_EXPECTED_GPU_UUID"] = a.expected_gpu_uuid
    runtime_source = Path(inspect.getfile(CudaSurfaceEDSR))
    native_source = runtime_source.parent / "native/nvdec.cpp"
    assert a.fps >= 0
    assert a.backend == "poll"
    assert a.input_format == "nv12_bt709_full" and a.decoder_latency == "LOW"
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    result = dict(
        complete=False,
        started=time.time(),
        gpu=guard(),
        script_sha256=digest(Path(__file__)),
        runtime_sha256=digest(runtime_source),
        nvc=nvc.__version__,
        mode=a.mode,
        samples=a.samples,
        fps=a.fps,
        profiler=a.capture,
        validation=[],
        input_format=a.input_format,
        decoder_latency=a.decoder_latency,
        decoder_backend=a.backend,
        decoder_source_sha256=digest(native_source),
        decoder_binary_sha256=digest(a.native_library),
        kernel_sha256=digest(runtime_source.with_name("cuda_kernels.py")),
        surface_source_sha256=digest(runtime_source.with_name("cuda_surface.py")),
        output_transform="float_area_then_round_clip_rgb8",
        scope="Compressed packet submit -> GPU decode/color -> native GPU mapping -> EDSR -> GPU RGBA1080 -> consumer stream handoff/release. No CPU pixels during measurement. Includes mapped-surface release after readers complete. Excludes demux/network/pacing/presentation/encode. Complete HEVC access units explicitly marked end-of-picture. Same-packet output verified by PTS. Separate same-frame PTS-matched age under controlled 30fps arrival; excludes network/demux/presentation.",
    )
    save(a.output, result)
    try:
        runtime = CudaSurfaceEDSR(
            a.engine, engine_sha256=a.engine_sha256, input_format=a.input_format
        )
        decode_stream = runtime.stream
        torch.cuda.set_stream(decode_stream)
        consumer = torch.cuda.Stream()
        reference = Engine(a.engine)
        runs = [
            r
            for r in json.loads((a.validation / "INPUTS.json").read_text())["runs"]
            if r["config"] == "720p-600k-420"
        ]
        run = runs[0]
        path = a.validation / run["video"]
        assert (
            digest(path) == run["video_sha256"] and run["stream"]["has_b_frames"] == 0
        )
        if a.input_format == "nv12_bt709_full":
            assert (
                run["stream"]["color_space"] == "bt709"
                and run["stream"]["color_range"] == "pc"
            )
        result["source"] = {k: run[k] for k in ("clip", "video_sha256", "stream")}
        wall = []
        ages = []
        compute = []
        decode = []
        passes = 0
        telemetry = []
        arrivals = []
        timed_callback = 0
        preparations = []
        prep_leads = []
        inclusive = []
        monitor = DeviceTelemetry()
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        while len(wall) < a.samples:
            demuxer = nvc.CreateDemuxer(filename=str(path))
            decoder = Decoder(decode_stream, polling=a.backend == "poll")
            index = 0
            captured = False
            next_packet = time.perf_counter()
            submitted = {}
            for packet in demuxer:
                if index == 50 and a.capture and not captured:
                    torch.cuda.cudart().cudaProfilerStart()
                    torch.cuda.nvtx.range_push("phase_surface_nvdec")
                    captured = True
                    if a.fps:
                        next_packet = time.perf_counter() + 1 / a.fps
                prep_ms = 0.0
                prep_end = None
                if a.mode == "prearmed":
                    prep_start = time.perf_counter()
                    ticket = runtime.prepare(decoder)
                    prep_end = time.perf_counter()
                    prep_ms = (prep_end - prep_start) * 1000
                if a.fps:
                    time.sleep(max(0, next_packet - time.perf_counter()))
                    next_packet = time.perf_counter() + 1 / a.fps
                torch.cuda.nvtx.range_push("frame_surface_nvdec")
                t0 = time.perf_counter()
                submitted[packet.pts] = t0
                begin.record(runtime.stream)
                torch.cuda.nvtx.range_push("decode_and_submit")
                if a.mode == "inline":
                    prep_start = time.perf_counter()
                    ticket = runtime.prepare(decoder)
                    prep_ms = (time.perf_counter() - prep_start) * 1000
                    decoded = ticket.submit(packet)
                else:
                    decoded = ticket.submit(packet)
                if decoded is None:
                    torch.cuda.nvtx.range_pop()
                    torch.cuda.nvtx.range_pop()
                    continue
                frame, lease = decoded
                torch.cuda.nvtx.range_pop()
                t1 = time.perf_counter()
                assert frame.pts == packet.pts
                source_arrival = submitted.pop(frame.pts)
                if index >= 50:
                    timed_callback += int(bool(frame.metadata.reserved))
                lease.wait_input_reuse_on(decode_stream)
                output = lease.wait_on(consumer)
                checking = index < 50 and index in (0, 24, 49)
                if checking:
                    with torch.cuda.stream(consumer):
                        snapshot = output.clone()
                lease.release()
                end.record(consumer)
                end.synchronize()
                t2 = time.perf_counter()
                torch.cuda.nvtx.range_pop()
                if index >= 50:
                    arrivals.append(t0)
                    preparations.append(prep_ms)
                    prep_leads.append(
                        (t0 - prep_end) * 1000 if prep_end is not None else 0.0
                    )
                    wall.append((t2 - t0) * 1000)
                    ages.append((t2 - source_arrival) * 1000)
                    compute.append(begin.elapsed_time(end))
                    decode.append((t1 - t0) * 1000)
                elif checking:
                    tensor = frame.as_tensor()
                    assert tensor.data_ptr() == frame.pointer
                    if a.input_format == "nv12_bt709_full":
                        from benchmarks.super_resolution.edsr_cuda.common import (
                            reference_rgb,
                        )

                        rgb = reference_rgb(tensor)
                    else:
                        rgb = tensor
                    reference.input.copy_(rgb.unsqueeze(0).float() / 255)
                    y = reference.run()
                    native = (
                        y[0]
                        .mul(255)
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                    expected = reference_resize(native)
                    actual = snapshot.cpu().numpy()
                    assert np.array_equal(actual[:, :, :3], expected) and np.all(
                        actual[:, :, 3] == 255
                    )
                    result["validation"].append(
                        dict(
                            pass_index=passes,
                            frame=index,
                            pointer_shared=True,
                            max_rgb=0,
                            alpha=255,
                        )
                    )
                    del tensor, rgb
                release_start = time.perf_counter()
                decoder.release(end)
                if index >= 50:
                    release_ms = (time.perf_counter() - release_start) * 1000
                    wall[-1] += release_ms
                    ages[-1] += release_ms
                    inclusive.append(
                        ages[-1] + (prep_ms if a.mode == "prearmed" else 0)
                    )
                if index >= 50 and len(wall) % 10 == 0:
                    telemetry.append(dict(sample=len(wall), **monitor.sample()))
                del frame, decoded
                index += 1
                if len(wall) >= a.samples:
                    break
            if captured:
                torch.cuda.nvtx.range_pop()
                torch.cuda.cudart().cudaProfilerStop()
            assert index > 50 and runtime.collect_completed()
            decoder.close()
            del decoder, demuxer
            passes += 1
        result["graph_builds"] = runtime.graph_builds
        result["callback_launches"] = runtime.callback_launches
        result["cold_launches"] = runtime.cold_launches
        result["timed_callback_launches"] = timed_callback
        assert timed_callback == a.samples
        runtime.close()
        spacing = (np.diff(arrivals) * 1000).tolist()
        if a.fps:
            assert min(spacing) >= 1000 / a.fps - 0.5, min(spacing)
        result.update(
            complete=True,
            finished=time.time(),
            gpu_end=guard(),
            passes=passes,
            telemetry=telemetry,
            arrival_spacing_ms=dict(min_ms=min(spacing), **stats(spacing)),
            timings=dict(
                same_frame_packet_to_consumer_gpu_wall=stats(ages),
                packet_to_consumer_gpu_wall=stats(wall),
                packet_to_consumer_gpu_event=stats(compute),
                decode_submit_api_wall=stats(decode),
                preparation_wall=stats(preparations),
                preparation_lead=stats(prep_leads),
                packet_plus_preparation_wall=stats(inclusive),
            ),
            preparation_note="prearmed reserves before the simulated packet arrival and never inspects that packet. Cost and lead time reported separately. packet_plus_preparation_wall adds this cost back; inline already includes preparation in packet wall.",
            event_timing_note="Start event is recorded before decoder and graph submission in both modes; includes GPU-side waiting. Host wall is the acceptance metric.",
            quality_scope="Native decoder input matches installed decoder on all 72 frozen frames plus 1200 early frames; warmup output equivalence checked before timing; runtime has separate 72-frame pixel equivalence and decoder RGB has separate 72-frame PSNR validation. This is not a new independent quality dataset.",
        )
        save(a.output, result)
        print(
            json.dumps({k: v["p95_ms"] for k, v in result["timings"].items()}),
            flush=True,
        )
    except BaseException as exc:
        result.update(error=repr(exc), failed=time.time())
        save(a.output, result)
        raise


if __name__ == "__main__":
    main()
