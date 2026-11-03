"""Score the timed GPU bicubic paths on the frozen Mac validation frames."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from benchmarks.super_resolution.benchmark_best_latency import guard, save


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def psnr(reference, output):
    mse = np.mean((reference.astype(np.float64) - output.astype(np.float64)) ** 2)
    return 10 * math.log10(255**2 / mse) if mse else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    gpu = guard()
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    inputs = json.loads((a.base / "INPUTS.json").read_text())
    rows = []
    for run in inputs["runs"]:
        if run["config"] != "720p-600k-420":
            continue
        guard()
        for sample in run["samples"]:
            for role in ["lr", "hr"]:
                assert sha(a.base / sample[role]) == sample[role + "_sha256"]
            low = np.asarray(Image.open(a.base / sample["lr"]).convert("RGB"))
            high = np.asarray(Image.open(a.base / sample["hr"]).convert("RGB"))
            outputs = {
                "opencv_cpu_1080": cv2.resize(
                    low, (1920, 1080), interpolation=cv2.INTER_CUBIC
                )
            }
            x = (
                torch.from_numpy(low.copy()).permute(2, 0, 1)[None].to("cuda:0").float()
                / 255
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for height, width in [(1080, 1920), (1440, 2560)]:
                    y = F.interpolate(
                        x,
                        size=(height, width),
                        mode="bicubic",
                        align_corners=False,
                        antialias=False,
                    )
                    assert y.dtype == torch.float32
                    native = (
                        (y[0] * 255)
                        .round()
                        .clamp(0, 255)
                        .to(torch.uint8)
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                    outputs[f"gpu_bicubic_{height}"] = (
                        cv2.resize(native, (1920, 1080), interpolation=cv2.INTER_AREA)
                        if height == 1440
                        else native
                    )
            for method, output in outputs.items():
                rows.append(
                    dict(
                        method=method,
                        clip=run["clip"],
                        frame=sample["frame"],
                        psnr=psnr(high, output),
                        lr_sha256=sample["lr_sha256"],
                        hr_sha256=sample["hr_sha256"],
                        output_sha256=hashlib.sha256(output.tobytes()).hexdigest(),
                    )
                )
    summary = {}
    for method in ["opencv_cpu_1080", "gpu_bicubic_1080", "gpu_bicubic_1440"]:
        selected = [r for r in rows if r["method"] == method]
        clips = sorted({r["clip"] for r in selected})
        assert len(selected) == 72 and len(clips) == 24
        summary[method] = float(
            np.mean(
                [
                    np.mean([r["psnr"] for r in selected if r["clip"] == clip])
                    for clip in clips
                ]
            )
        )
    assert abs(summary["opencv_cpu_1080"] - 27.485264697850692) < 1e-8, (
        "Historical baseline did not reproduce"
    )
    save(
        a.output,
        dict(
            complete=True,
            gpu=gpu,
            summary=summary,
            rows=rows,
            input_manifest_sha256=sha(a.base / "INPUTS.json"),
            script_sha256=sha(Path(__file__)),
            torch=str(torch.__version__),
            opencv=cv2.__version__,
            conditions="720p600Kbps420;24clips72frames;equal-clip mean of per-frame RGB PSNR; final1920x1080",
            gpu_dtype="FP32 under autocast, as in the timing run",
            conversion="Multiply255, round, clamp, uint8;1440p intermediate then OpenCV area to1080p, matching SR quality protocol",
            timing_note="Prior1440p latency excludes final area resize, transfers and quantization",
            test_used=False,
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
