"""Actual frozen-frame output equivalence and native-callback ownership checks."""

import argparse
import inspect
import os
import json
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
import PyNvVideoCodec as nvc
from isharescreen.enhancement.cuda_decoder import NvdecDecoder as Decoder
from isharescreen.enhancement.cuda_native import (
    NativeCudaSurfaceEDSR as DirectSurfaceEDSR,
)
from isharescreen.enhancement.cuda_surface import SurfaceBusy
from benchmarks.super_resolution.edsr_cuda.common import (
    Engine,
    reference_rgb,
    guard,
    save,
    digest,
    reference_resize,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--reference-checkpoint", type=Path,
                        help="Optional frozen EDSR weights for FP32/AMP quality gates")
    args = parser.parse_args()
    out = args.output
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    root, engine, sha = args.validation, args.engine, args.engine_sha256
    os.environ["ISS_NVDEC_LIBRARY"] = str(args.native_library.resolve())
    os.environ["ISS_EXPECTED_GPU_UUID"] = args.expected_gpu_uuid
    result = dict(
        complete=False,
        gpu=guard(),
        engine_sha256=sha,
        frames=[],
        ownership={},
        runtime_sha256=digest(Path(inspect.getfile(DirectSurfaceEDSR))),
        script_sha256=digest(Path(__file__)),
        decoder_source_sha256=digest(Path(inspect.getfile(Decoder))),
        native_library_sha256=digest(args.native_library),
        surface_source_sha256=digest(Path(inspect.getfile(DirectSurfaceEDSR)).with_name("cuda_surface.py")),
        kernel_sha256=digest(Path(inspect.getfile(DirectSurfaceEDSR)).with_name("cuda_kernels.py")),
        output_transform="float_area_then_round_clip_rgb8",
        input_manifest_sha256=digest(root / "INPUTS.json"),
    )
    save(out, result)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    runtime = DirectSurfaceEDSR(
        engine, engine_sha256=sha, input_format="nv12_bt709_full"
    )
    stream = runtime.stream
    torch.cuda.set_stream(stream)
    consumer = torch.cuda.Stream()
    end = torch.cuda.Event()
    reference = Engine(engine)
    quality_model = None
    if args.reference_checkpoint is not None:
        from isharescreen.enhancement.models import Upscaler

        checkpoint_sha = digest(args.reference_checkpoint)
        assert checkpoint_sha == "2c923e9ba9541cb85006684c0072868b2d71354d665cdd85cd2b0baac6faaabb"
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.benchmark = True
        quality_model = Upscaler("edsr", device="cuda:0", precision="fp32").model
        payload = torch.load(args.reference_checkpoint, map_location="cpu", weights_only=True)
        torch.nn.Module.load_state_dict(quality_model, payload["state_dict"], strict=True)
        del payload
        result["quality_reference"] = dict(
            checkpoint_sha256=checkpoint_sha, tf32=False,
            scope="Same actual NVDEC RGB input and float Area output for mixed engine, FP32 and AMP. "
                  "Paired legacy output uses the same engine output. 72 Mac development frames; no test data.",
            mean_psnr_loss_limit_db=0.05, worst_frame_psnr_loss_limit_db=0.10)
    runs = [
        x
        for x in json.loads((root / "INPUTS.json").read_text())["runs"]
        if x["config"] == "720p-600k-420"
    ]
    for run in runs:
        path = root / run["video"]
        assert digest(path) == run["video_sha256"]
        demux = nvc.CreateDemuxer(filename=str(path))
        decoder = Decoder(stream)
        selected = {x["frame"] for x in run["samples"]}
        index = 0
        for packet in demux:
            if index in selected or index == 0:
                submitted = runtime.decode_submit(decoder, packet)
                if submitted is None:
                    continue
                frame, lease = submitted
                if index == 0 and index not in selected:
                    lease.discard()
                    stream.synchronize()
            else:
                frame = decoder.decode(packet)
                if frame is None:
                    continue
            assert frame.pts == packet.pts
            if index in selected:
                assert frame.metadata.reserved == 1, (
                    "Frozen frame did not exercise native graph launch"
                )
                output = lease.wait_on(consumer)
                with torch.cuda.stream(consumer):
                    snapshot = output.clone()
                lease.release()
                consumer.synchronize()
                rgb = reference_rgb(frame.as_tensor())
                expected_input = rgb.unsqueeze(0).float() / 255
                assert torch.equal(runtime._input, expected_input), (
                    run["clip"],
                    index,
                    "input",
                )
                reference.input.copy_(expected_input)
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
                assert np.array_equal(expected, actual[:, :, :3]) and np.all(
                    actual[:, :, 3] == 255
                ), (run["clip"], index)
                result["frames"].append(
                    dict(
                        clip=run["clip"],
                        frame=index,
                        input_bit_exact=True,
                        rgba_bit_exact=True,
                        graph_cache_addresses=len(runtime._address_graphs),
                        callback_launched=True,
                    )
                )
                sample = next(s for s in run["samples"] if s["frame"] == index)
                assert digest(root / sample["hr"]) == sample["hr_sha256"]
                hr = np.asarray(Image.open(root / sample["hr"]).convert("RGB"))

                def psnr(pixels):
                    mse = np.mean((pixels.astype(np.float64)-hr.astype(np.float64))**2)
                    return float(10*np.log10(255**2/mse))

                scores = dict(optimized=psnr(actual[:, :, :3]),
                              legacy=psnr(reference_resize(native, quantize_first=True)))
                if quality_model is not None:
                    for label in ("fp32", "amp"):
                        with torch.autocast("cuda", dtype=torch.float16, enabled=label == "amp"):
                            prediction = quality_model(expected_input*255)
                        predicted_rgb = prediction[0].float().permute(1,2,0).contiguous().cpu().numpy()
                        assert np.isfinite(predicted_rgb).all()
                        scores[label] = psnr(reference_resize(predicted_rgb))
                        del prediction
                result["frames"][-1]["psnr"] = scores
            end.record(stream)
            end.synchronize()
            decoder.release(end)
            index += 1
            if index > max(selected):
                break
        assert runtime.collect_completed() and len(runtime._address_graphs) <= 2
        decoder.close()
        save(out, result)
        print("validated", len(result["frames"]), "frames", flush=True)
    assert len(result["frames"]) == 72
    from benchmarks.super_resolution.edsr_cuda.ownership import check_ownership

    result["ownership"] = check_ownership(runtime, root / runs[0]["video"], engine, sha)
    from benchmarks.super_resolution.edsr_cuda.ticket_checks import check_tickets

    result["tickets"] = check_tickets(runtime, root / runs[0]["video"])
    result["callback_launches"] = runtime.callback_launches
    result["cold_launches"] = runtime.cold_launches
    result["graph_builds"] = runtime.graph_builds
    runtime.close()
    scores = result["frames"]
    result["mean_psnr"] = {key: float(np.mean([row["psnr"][key] for row in scores]))
                           for key in scores[0]["psnr"]}
    result["quality_deltas"] = {}
    for key in scores[0]["psnr"]:
        if key == "optimized":
            continue
        deltas = [row["psnr"]["optimized"]-row["psnr"][key] for row in scores]
        result["quality_deltas"][key] = dict(mean_delta=float(np.mean(deltas)),
            worst_delta=min(deltas), best_delta=max(deltas),
            improved_frames=sum(v > 1e-6 for v in deltas),
            regressed_frames=sum(v < -1e-6 for v in deltas))
    if quality_model is not None:
        result["quality_gate_passed"] = all(
            result["quality_deltas"][key]["mean_delta"] >= -0.05 and
            result["quality_deltas"][key]["worst_delta"] >= -0.10 for key in ("fp32", "amp"))
    result["complete"] = True
    save(out, result)
    if quality_model is not None:
        assert result["quality_gate_passed"], result["quality_deltas"]
    print("PASS: 72 exact outputs and native-callback ownership checks", flush=True)


if __name__ == "__main__":
    main()
