"""Replay pinned EDSR x2 weights on three verified 720p historical video frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from evaluate_edsr_gpu import get_model
from isharescreen.enhancement.models import sha256


def run(root: Path, assets: Path) -> None:
    plan = json.loads((root / "PLAN.json").read_text())
    weights = json.loads((root / "weights.json").read_text())
    if sha256(root / "weights.json") != plan["weights_sha256"]:
        raise ValueError("Weights manifest changed")
    if sha256(Path(__file__)) != plan["remote_runner_sha256"]:
        raise ValueError("Runner changed")
    if weights["models"]["old_finetuned_edsr_x2"]["sha256"] != plan["edsr1500_sha256"]:
        raise ValueError("Wrong 1500-epoch checkpoint")
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    rows = []
    started = time.time()
    for method in ("official_edsr_x2", "old_finetuned_edsr_x2"):
        model, identity = get_model(method, weights["models"], assets)
        model = model.to(device).eval().requires_grad_(False)
        for scene in plan["scenes"]:
            path = root / "inputs" / f"{scene['id']}.png"
            if sha256(path) != scene["input_sha256"]:
                raise ValueError(f"Input changed: {scene['id']}")
            image = np.asarray(Image.open(path).convert("RGB"))
            if image.shape != (720, 1280, 3):
                raise ValueError("Wrong 720p input geometry")
            tensor = (
                torch.from_numpy(image.transpose(2, 0, 1).copy())[None]
                .to(device)
                .float()
                .div_(255)
            )
            torch.cuda.synchronize(device)
            before = time.perf_counter()
            with torch.inference_mode():
                output = model(tensor)
            torch.cuda.synchronize(device)
            forward_ms = (time.perf_counter() - before) * 1000
            if output.shape != (1, 3, 1440, 2560) or not torch.isfinite(output).all():
                raise ValueError("Invalid EDSR output")
            pixels = (
                output[0]
                .float()
                .mul(255)
                .round()
                .clamp(0, 255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            display = cv2.resize(pixels, (1920, 1080), interpolation=cv2.INTER_AREA)
            target = root / "outputs" / method / f"{scene['id']}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(display).save(target)
            rows.append(
                {
                    "method": method,
                    "scene": scene["id"],
                    "input_sha256": scene["input_sha256"],
                    "output": str(target.relative_to(root)),
                    "output_sha256": sha256(target),
                    "checkpoint_sha256": identity["checkpoint_sha256"],
                    "forward_ms": forward_ms,
                }
            )
            print(f"INFER {method} {scene['id']} {forward_ms:.1f} ms", flush=True)
        del model
        torch.cuda.empty_cache()
    result = {
        "schema": "historical_edsr1500_gpu_replay_v1",
        "plan_sha256": sha256(root / "PLAN.json"),
        "host": __import__("socket").gethostname(),
        "gpu_name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_count": 1,
        "started_unix": started,
        "ended_unix": time.time(),
        "rows": rows,
    }
    (root / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    (root / "COMPLETED.json").write_text(
        json.dumps(
            {"results_sha256": sha256(root / "results.json"), "rows": len(rows)},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.assets.resolve())
