"""Render the same fixed validation frame with all seven best checkpoints."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.benchmark_best_latency import guard, save
from benchmarks.super_resolution.legacy_validation_compare import adapter, display


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clip", default="2026-10-15_18-12-43")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    gpu = guard()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    inputs = json.loads((a.base / "INPUTS.json").read_text())
    run = next(
        r
        for r in inputs["runs"]
        if r["config"] == "720p-600k-420" and r["clip"] == a.clip
    )
    sample = run["samples"][1]
    for role in ["lr", "hr"]:
        assert sha(a.base / sample[role]) == sample[role + "_sha256"]
    low = np.asarray(Image.open(a.base / sample["lr"]).convert("RGB"))
    reference = np.asarray(Image.open(a.base / sample["hr"]).convert("RGB"))
    rows = []

    def write(name, image, **extra):
        path = a.output / (name + ".png")
        Image.fromarray(image).save(path)
        mse = float(
            np.mean((reference.astype(np.float64) - image.astype(np.float64)) ** 2)
        )
        rows.append(
            dict(
                id=name,
                path=path.name,
                sha256=sha(path),
                frame_psnr=10 * np.log10(65025 / mse) if mse else None,
                **extra,
            )
        )

    write("reference", reference)
    write("bicubic", cv2.resize(low, (1920, 1080), interpolation=cv2.INTER_CUBIC))
    plan = json.loads((a.weights / "PLAN.json").read_text())
    for item in sorted(plan["models"], key=lambda r: r["psnr"], reverse=True):
        guard()
        key = item["model"]
        checkpoint = a.weights / "checkpoints" / f"{key}.pt"
        assert sha(checkpoint) == item["sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        wrapper = adapter(key, "cuda:0")
        with torch.inference_mode():
            before = display(wrapper, low)
        write(
            key + "-before",
            before,
            role="official_pretrained",
            checkpoint_sha256=wrapper.provenance["checkpoint_sha256"],
        )
        torch.nn.Module.load_state_dict(
            wrapper.model, payload["state_dict"], strict=True
        )
        wrapper.model.eval().requires_grad_(False)
        with torch.inference_mode():
            image = display(wrapper, low)
        write(
            key,
            image,
            epoch=item["epoch"],
            validation_psnr=item["psnr"],
            checkpoint_sha256=item["sha256"],
        )
        del wrapper, payload
        gc.collect()
        torch.cuda.empty_cache()
    save(
        a.output / "MANIFEST.json",
        dict(
            complete=True,
            gpu=gpu,
            clip=run["clip"],
            frame=sample["frame"],
            input=sample,
            precision="FP32, matching the quality table; no source-detail supplement",
            selection="Fixed middle validation frame from a chosen scene; illustrative, not a representative quality estimate",
            rows=rows,
        ),
    )
    print("Rendered", len(rows), "matched images", flush=True)


if __name__ == "__main__":
    main()
