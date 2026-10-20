"""Batch-one CUDA/MPS/Core ML latency on sampled desktop recording frames.

Resident forward timing excludes preparation and transfers. Adapter timing uses
the actual inference adapter, including its validation and RGB output conversion.
Video decoding is measured separately, never included in either inference time.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement import candidates, coreml, models


def summary(values):
    return dict(p50_ms=float(np.median(values)), p95_ms=float(np.percentile(values, 95)),
                mean_ms=float(np.mean(values)), samples_ms=values)


def sample_indices(count, requested):
    if count < 1 or requested < 1:
        raise ValueError("Frame counts must be positive")
    return np.linspace(0, count-1, min(count, requested), dtype=int).tolist()


def read_video(path, requested, limit):
    import av
    frames, timings = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec, rate = stream.codec_context.name, str(stream.average_rate)
        iterator = iter(container.decode(stream))
        truncated = False
        for _ in range(limit):
            started = time.perf_counter()
            try:
                frame = next(iterator)
            except StopIteration:
                break
            rgb = frame.to_ndarray(format="rgb24")
            if rgb.shape != (540, 960, 3):
                raise ValueError("Use a 960x540 desktop clip for this protocol")
            timings.append((time.perf_counter()-started)*1000)
            frames.append(rgb)
        if len(frames) == limit:
            truncated = next(iterator, None) is not None
    indices = sample_indices(len(frames), requested)
    selected = [frames[index] for index in indices]
    return selected, dict(sha256=models.sha256(path), codec=codec, frame_rate=rate,
        decoded_frames=len(frames), truncated=truncated, selected_indices=indices,
        rgb_sha256=[hashlib.sha256(rgb.tobytes()).hexdigest() for rgb in selected],
        decode_and_rgb=summary(timings))


def resident(adapter, frames, warmup):
    import torch
    samples = []
    output_shape = None
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16,
                                               enabled=adapter.provenance["precision"] == "amp-fp16"):
        for index in range(warmup+len(frames)):
            rgb = frames[index % len(frames) if index < warmup else index-warmup]
            array = rgb.astype(np.float32)
            if adapter.key != "edsr":
                array /= 255
            value = torch.from_numpy(np.ascontiguousarray(array.transpose(2, 0, 1)[None])).to(
                device=adapter.device, dtype=adapter.dtype)
            adapter.sync()
            if adapter.device.startswith("cuda"):
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                output = adapter.model(value)
                end.record()
                end.synchronize()
                elapsed = begin.elapsed_time(end)
            else:
                started = time.perf_counter()
                output = adapter.model(value)
                adapter.sync()
                elapsed = (time.perf_counter()-started)*1000
            # Validate outside the event interval; do not time CPU observation.
            scale = adapter.spec["scale"] if hasattr(adapter, "spec") else 2
            if tuple(output.shape) != (1, 3, 540*scale, 960*scale) or not torch.isfinite(output).all().item():
                raise ValueError("Model returned invalid native-scale pixels")
            output_shape = list(output.shape)
            if index >= warmup:
                samples.append(elapsed)
    return dict(timer="cuda-events" if adapter.device.startswith("cuda") else "synchronized-host-wall",
                output_shape=output_shape, **summary(samples))


def adapter_probe(adapter, frames, warmup):
    for index in range(warmup):
        adapter(frames[index % len(frames)])
    timings = [adapter(rgb)[1] for rgb in frames]
    return {stage: summary([row[stage] for row in timings]) for stage in timings[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("bicubic", "edsr", "imdn", "safmn", *candidates.IDS),
                        default=["bicubic", "edsr", "imdn", "safmn", *candidates.IDS])
    parser.add_argument("--device", choices=("auto", "cuda", "mps"), default="auto")
    parser.add_argument("--backends", nargs="+", choices=("torch", "coreml"), default=["torch"])
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "amp-fp16"))
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--max-decoded-frames", type=int, default=600)
    args = parser.parse_args()
    if os.environ.get("ISS_SR_GPU_ONLY") == "1":
        if "coreml" in args.backends or "bicubic" in args.models:
            parser.error("GPU-only mode excludes Core ML and CPU baselines; specify only learned Torch models")
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            parser.error("Disable MPS CPU fallback before starting Python")
    if args.output.exists() or min(args.samples, args.warmup, args.max_decoded_frames) < 1:
        parser.error("Use a fresh output directory and positive frame/warm-up counts")
    import torch
    import cv2
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "mps"
    if ((args.device == "cuda" and not torch.cuda.is_available())
            or (args.device == "mps" and not torch.backends.mps.is_available())):
        parser.error("The selected acceleration device is unavailable")
    if args.precisions is None:
        args.precisions = ["fp32", "amp-fp16"] if args.device == "cuda" else ["fp32"]
    if args.device == "mps" and args.precisions != ["fp32"]:
        parser.error("MPS comparison uses FP32; converted Core ML precision is validated separately")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    frames, video = read_video(args.video, args.samples, args.max_decoded_frames)
    args.output.mkdir(parents=True)
    source_root = Path(candidates.__file__).parent.parent
    sources = [*source_root.rglob("*.py"), *source_root.rglob("*.json"), Path(__file__)]
    # Include the legacy official architecture files actually supplied for this run.
    sources.extend((models.project_home()/"vendor").rglob("*.py"))
    def gpu_state():
        if args.device == "mps":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.used,utilization.gpu",
                                        "--format=csv,noheader"], text=True).strip()
    report = dict(schema="sr_video_latency_v1", started_at=time.time(), command=sys.argv, pid=os.getpid(),
        platform=platform.platform(), torch=torch.__version__, cuda=torch.version.cuda,
        device=torch.cuda.get_device_name(0) if args.device == "cuda" else gpu_state(),
        gpu_before=gpu_state(), video=video, records=[],
        source_sha256={str(path): models.sha256(path) for path in sources},
        warmup=args.warmup, tf32=False, cudnn_benchmark=True, batch_size=1,
        gpu_only=os.environ.get("ISS_SR_GPU_ONLY") == "1",
        mps_cpu_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        driver_library_path=os.environ.get("LD_LIBRARY_PATH", ""),
        scope="Serial warmed inference; no decode, network, display, pacing or queueing in inference timings. Native x4 results separate from x2.")
    for key in args.models:
        fp32_rgb = None
        for precision in ((["fp32"] if key == "bicubic" else args.precisions) if "torch" in args.backends else []):
            adapter = None
            try:
                adapter = (candidates.TorchUpscaler(key, device=args.device, precision=precision)
                           if key in candidates.IDS else models.Upscaler(key, device=args.device, precision=precision))
                row = dict(model=key, backend="opencv" if key == "bicubic" else "torch", precision=precision,
                           scale=adapter.spec["scale"] if key in candidates.IDS else 2,
                           provenance=adapter.provenance, status="completed")
                if key != "bicubic":
                    row["resident_forward"] = resident(adapter, frames, args.warmup)
                row["adapter"] = adapter_probe(adapter, frames, args.warmup)
                rgb, _ = adapter(frames[0])
                if precision == "fp32":
                    fp32_rgb = rgb.copy()
                elif fp32_rgb is not None:
                    delta = np.abs(rgb.astype(np.int16)-fp32_rgb.astype(np.int16))
                    row["first_frame_vs_fp32"] = dict(max_rgb_delta=int(delta.max()), mean_rgb_delta=float(delta.mean()))
                report["records"].append(row)
                print(json.dumps(dict(model=key, precision=precision, scale=row["scale"],
                    resident_p50_ms=row.get("resident_forward", {}).get("p50_ms"),
                    adapter_p50_ms=row["adapter"]["total_ms"]["p50_ms"],
                    adapter_p95_ms=row["adapter"]["total_ms"]["p95_ms"])), flush=True)
            except Exception as error:
                report["records"].append(dict(model=key, precision=precision, status="failed", reason=f"{type(error).__name__}: {error}"))
                # Device loss can make cleanup fail too. Preserve the actual
                # experiment failure before touching the broken CUDA context.
                atomic_json(args.output/"results.json", report)
                print(f"FAILED {key} {precision}: {error}", flush=True)
            finally:
                del adapter
                gc.collect()
                try:
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
                    else:
                        torch.mps.empty_cache()
                except RuntimeError as error:
                    report.update(fatal_error=f"Device cleanup failed: {error}", finished_at=time.time())
                    atomic_json(args.output/"results.json", report)
                    raise
            atomic_json(args.output/"results.json", report)
        if "coreml" in args.backends and key in ("quicksrnet", *coreml.CANDIDATE_IDS):
            adapter = None
            try:
                adapter = coreml.CoreMLUpscaler(candidate=None if key == "quicksrnet" else key)
                timing = adapter_probe(adapter, frames, args.warmup)
                report["records"].append(dict(model=key, backend="coreml", scale=2, status="completed",
                    precision=adapter.provenance["precision"], provenance=adapter.provenance, adapter=timing))
                print(json.dumps(dict(model=key, backend="coreml", precision=adapter.provenance["precision"],
                    adapter_p50_ms=timing["total_ms"]["p50_ms"], adapter_p95_ms=timing["total_ms"]["p95_ms"])), flush=True)
            except Exception as error:
                report["records"].append(dict(model=key, backend="coreml", status="failed", reason=f"{type(error).__name__}: {error}"))
                print(f"FAILED {key} coreml: {error}", flush=True)
            finally:
                del adapter
                gc.collect()
            atomic_json(args.output/"results.json", report)
    report.update(completed=True, finished_at=time.time(), gpu_after=gpu_state())
    atomic_json(args.output/"results.json", report)


if __name__ == "__main__":
    main()
