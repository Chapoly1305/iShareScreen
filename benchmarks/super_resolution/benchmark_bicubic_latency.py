"""Measure GPU bicubic and the exact OpenCV no-SR baseline implementation."""

import argparse
import hashlib
from pathlib import Path
import platform
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from benchmarks.super_resolution.benchmark_best_latency import guard, save


def summary(samples):
    return dict(
        samples=len(samples),
        samples_ms=samples,
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        p99_ms=float(np.percentile(samples, 99)),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    result = dict(
        complete=False,
        gpu=guard(),
        host=platform.node(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        opencv=cv2.__version__,
        warmup=50,
        samples=1000,
        rows=[],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="Resize only. No codec, transfers or rendering. GPU outputs are not re-scored for PSNR.",
    )
    x = torch.linspace(0, 1, 1280 * 720 * 3, device="cuda:0").reshape(1, 3, 720, 1280)
    for size in [(1080, 1920), (1440, 2560)]:
        guard()
        begin, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        samples = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for _ in range(50):
                y = F.interpolate(
                    x, size=size, mode="bicubic", align_corners=False, antialias=False
                )
            torch.cuda.synchronize()
            for i in range(1000):
                begin.record()
                y = F.interpolate(
                    x, size=size, mode="bicubic", align_corners=False, antialias=False
                )
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end))
                if (i + 1) % 100 == 0:
                    guard()
            assert torch.isfinite(y).all().item()
        result["rows"].append(
            dict(
                method="pytorch_cuda_bicubic",
                output_hw=size,
                input_dtype=str(x.dtype),
                output_dtype=str(y.dtype),
                autocast_enabled=True,
                note="Autocast does not necessarily cast interpolation; actual dtypes are recorded.",
                timer="CUDA events",
                **summary(samples),
            )
        )
    rgb = np.linspace(0, 255, 1280 * 720 * 3, dtype=np.uint8).reshape(720, 1280, 3)
    for _ in range(50):
        y = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    samples = []
    for _ in range(1000):
        start = time.perf_counter_ns()
        y = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    result["rows"].append(
        dict(
            method="opencv_cpu_bicubic",
            output_hw=[1080, 1920],
            input_dtype="uint8",
            output_dtype=str(y.dtype),
            threads=1,
            timer="CPU perf_counter_ns",
            note="Matches the no-SR PSNR baseline resize implementation; CPU timing on4090host, not GPU timing.",
            **summary(samples),
        )
    )
    result["complete"] = True
    save(args.output / "RESULTS.json", result)
    for row in result["rows"]:
        print(
            row["method"],
            row["output_hw"],
            row["output_dtype"],
            {k: row[k] for k in ["p50_ms", "p95_ms", "p99_ms"]},
            flush=True,
        )


if __name__ == "__main__":
    main()
