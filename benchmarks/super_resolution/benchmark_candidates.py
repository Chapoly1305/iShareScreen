"""Unified native-scale candidate latency and paired x2 image-quality probe."""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

from benchmarks.super_resolution.metrics import metrics
from benchmarks.super_resolution.profile_live import measure
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement import candidates, coreml, models
from isharescreen.enhancement.models import sha256


def resident_forward(adapter, rgb, warmup, iterations):
    """Resident Torch forward only; CUDA events or synchronized MPS/CPU wall time."""
    import torch
    source = np.ascontiguousarray((rgb.astype(np.float32)/255).transpose(2,0,1)[None])
    value = torch.from_numpy(source).to(device=adapter.device, dtype=adapter.dtype)
    samples = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=adapter.precision == "amp-fp16"):
        for index in range(warmup+iterations):
            adapter.sync()
            if adapter.device.startswith("cuda"):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                output = adapter.model(value)
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end)
            else:
                started = time.perf_counter()
                output = adapter.model(value)
                adapter.sync()
                elapsed = (time.perf_counter()-started)*1000
            if index >= warmup:
                samples.append(elapsed)
        if not torch.isfinite(output).all().item():
            raise ValueError("Resident model returned nonfinite pixels")
    return dict(samples_ms=samples, p50_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples,95)),
                timer="cuda-events" if adapter.device.startswith("cuda") else "synchronized-host-wall")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True, help="code/terminal/desktop folders containing low.png and reference.png")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=candidates.IDS, default=list(candidates.IDS))
    parser.add_argument("--backends", nargs="+", choices=("torch", "coreml"), default=["torch", "coreml"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("fp32", "fp16", "amp-fp16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--lpips", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output directory")
    if args.warmup < 1 or args.iterations < 2:
        parser.error("Use at least one warm-up and two measured calls")
    import torch
    from PIL import Image
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    scenes = {}
    for name in ("code", "terminal", "desktop"):
        folder = args.prepared/name
        rgb = np.array(Image.open(folder/"low.png").convert("RGB"))
        reference = np.array(Image.open(folder/"reference.png").convert("RGB"))
        if rgb.shape != (540,960,3) or reference.shape != (1080,1920,3):
            raise ValueError("Expected paired 540p and 1080p images")
        scenes[name] = rgb, reference
    network, lpips_identity = None, None
    if args.lpips:
        from benchmarks.super_resolution.train import perceptual_network
        metric_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        network, lpips_identity = perceptual_network(metric_device)
    args.output.mkdir(parents=True)
    sources = [p for p in Path(candidates.__file__).parent.rglob("*") if p.suffix in (".py", ".json")]
    sources.extend((Path(coreml.__file__), Path(models.__file__), Path(__file__),
                    Path(__file__).with_name("profile_live.py"), Path(__file__).with_name("metrics.py")))
    report = dict(schema="sr_candidate_probe_v1", platform=platform.platform(), torch=torch.__version__,
        generator_sha256=sha256(Path(__file__)), records=[], warmup=args.warmup, iterations=args.iterations,
        pid=os.getpid(), command=sys.argv, started_at=time.time(), working_directory=str(Path.cwd()),
        source_sha256={str(p): sha256(p) for p in sources},
        input=[1,3,540,960], lpips_identity=lpips_identity,
        scope="Fixed recording posters; native scales reported separately. Adapter excludes decode, canvas assembly and display. Not sustained live FPS.")
    def quality(reference, output):
        result = metrics(reference, output)
        if network is not None:
            from benchmarks.super_resolution.video_perceptual import normalized_tensor
            with torch.inference_mode():
                result["lpips"] = float(network(normalized_tensor(reference, metric_device), normalized_tensor(output, metric_device)).item())
        return result

    baseline = models.Upscaler("bicubic")
    for name, (rgb, reference) in scenes.items():
        summary = measure(baseline, rgb, args.warmup, args.iterations)
        output, _ = baseline(rgb)
        folder = args.output/"bicubic"
        folder.mkdir(exist_ok=True)
        Image.fromarray(output).save(folder/f"{name}.png")
        report["records"].append(dict(model="bicubic", backend="opencv", scale=2, scene=name,
            status="completed", input_sha256=sha256(args.prepared/name/"low.png"),
            reference_sha256=sha256(args.prepared/name/"reference.png"),
            quality=quality(reference, output), **summary))
    atomic_json(args.output/"results.json", report)
    for key in args.models:
        spec = candidates.specification(key)
        for backend in args.backends:
            if backend == "coreml" and spec["scale"] != 2:
                report["records"].append(dict(model=key, backend=backend, scale=spec["scale"], status="unsupported",
                    reason="Only native x4 weights; not installed as a live x2 backend"))
                atomic_json(args.output/"results.json", report)
                continue
            print(f"START {key} {backend}", flush=True)
            adapter = None
            try:
                adapter = (candidates.TorchUpscaler(key, device=args.device, precision=args.precision) if backend == "torch"
                           else coreml.CoreMLUpscaler(candidate=None if key == "quicksrnet" else key))
                for name, (rgb, reference) in scenes.items():
                    summary = measure(adapter, rgb, args.warmup, args.iterations)
                    output, _ = adapter(rgb)
                    row = dict(model=key, backend=backend, scale=spec["scale"], scene=name, status="completed",
                        provenance=adapter.provenance, output_shape=list(output.shape),
                        input_sha256=sha256(args.prepared/name/"low.png"),
                        reference_sha256=sha256(args.prepared/name/"reference.png"), **summary)
                    if backend == "torch":
                        row["resident_forward"] = resident_forward(adapter, rgb, args.warmup, args.iterations)
                    folder = args.output/key/backend
                    folder.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(output).save(folder/f"{name}.png")
                    if spec["scale"] == 2:
                        row["quality"] = quality(reference, output)
                    else:
                        row["quality"] = None
                        row["quality_reason"] = "Native x4 output has no paired 2160p ground truth in this x2 protocol"
                    report["records"].append(row)
                    atomic_json(args.output/"results.json", report)
                    print(json.dumps({k:v for k,v in row.items() if k not in ("samples", "resident_forward")}), flush=True)
            except Exception as error:
                report["records"].append(dict(model=key, backend=backend, scale=spec["scale"],
                    status="failed", reason=f"{type(error).__name__}: {error}"))
                atomic_json(args.output/"results.json", report)
                print(f"FAILED {key} {backend}: {error}", flush=True)
            finally:
                del adapter
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
    report["completed"] = True
    report["finished_at"] = time.time()
    atomic_json(args.output/"results.json", report)


if __name__ == "__main__":
    main()
