"""Screen fixed INT8 range margins using frozen train calibration.

Margins are numerical implementation candidates evaluated on development
validation, never refitted to its activation statistics or the held-out test.
No timing claim is made for this fake-quantization accuracy screen.
"""
import argparse
import copy
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn

from benchmarks.super_resolution.profile_edsr_smooth_int8 import QuantizedConv, image_tensor, pixels
from benchmarks.super_resolution.profile_inference_optimization import adapter, amp, digest, forward, guard, save


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "validation", "calibration", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--margins", type=float, nargs="+", default=[1., 1.25, 1.5, 1.75])
    a = p.parse_args()
    if a.output.exists(): raise FileExistsError(a.output)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(complete=False, started=time.time(), gpu=guard(), scope=__doc__, margins=a.margins,
                  script_sha256=digest(Path(__file__)), calibration_sha256=digest(a.calibration))
    save(a.output, result)
    cal = json.loads(a.calibration.read_text())
    assert cal["complete"] and cal["calibration"]["split"] == "train"
    ckpt = a.weights / "checkpoints/edsr.pt"
    item = next(r for r in json.loads((a.weights / "PLAN.json").read_text())["models"] if r["model"] == "edsr")
    assert digest(ckpt) == item["sha256"]
    result["checkpoint"] = item
    model = adapter("edsr", "cpu").model.eval().requires_grad_(False)
    nn.Module.load_state_dict(model, torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"], strict=True)
    base = copy.deepcopy(model).half().cuda()
    models = {}
    for margin in a.margins:
        assert margin >= 1
        candidate = copy.deepcopy(base)
        for name, maximum in cal["calibration"]["maxima"].items():
            if name.startswith("body.15."): continue
            parent, child = name.rsplit(".", 1)
            layer = QuantizedConv(candidate.get_submodule(name), maximum, ".75")
            layer.scale.mul_(margin)
            candidate.get_submodule(parent)._modules[child] = layer
        models[str(margin)] = candidate
    model.cuda()
    records = []
    runs = [r for r in json.loads((a.validation / "INPUTS.json").read_text())["runs"] if r["config"] == "720p-600k-420"]
    for run in runs:
        for sample in run["samples"]:
            for k in ("lr", "hr"): assert digest(a.validation / sample[k]) == sample[k + "_sha256"]
            x = image_tensor(a.validation / sample["lr"])
            target = np.asarray(Image.open(a.validation / sample["hr"]).convert("RGB"))
            images = {"fp32": pixels(forward(model, "edsr", x))}
            with amp(): images["baseline_amp"] = pixels(forward(model, "edsr", x))
            images["fp16"] = pixels(base((x * 255).half()).float() / 255)
            for mode, candidate in models.items():
                out = candidate((x * 255).half()).float() / 255
                assert torch.isfinite(out).all().item()
                images[mode] = pixels(out)
            row = dict(clip=run["clip"], frame=sample["frame"],
                psnr={k:float(10*np.log10(255**2/np.mean((v.astype(np.float64)-target.astype(np.float64))**2))) for k,v in images.items()},
                max_rgb_vs_fp16={k:int(np.abs(images[k].astype(np.int16)-images["fp16"].astype(np.int16)).max()) for k in models})
            records.append(row)
        print("range quality", len(records), flush=True)
    result["records"] = records; result["modes"] = {}
    for mode in models:
        metrics = {}
        for reference in ("fp32", "baseline_amp"):
            delta = [r["psnr"][mode] - r["psnr"][reference] for r in records]
            metrics[reference] = dict(mean_delta=float(np.mean(delta)), worst_delta=min(delta))
        result["modes"][mode] = dict(metrics=metrics,
            max_rgb_vs_fp16=max(r["max_rgb_vs_fp16"][mode] for r in records),
            passed=all(m["mean_delta"] >= -.05 and m["worst_delta"] >= -.1 for m in metrics.values()))
    result.update(complete=True, finished=time.time(), gpu_end=guard())
    save(a.output, result); print(json.dumps(result["modes"], indent=2), flush=True)


if __name__ == "__main__": main()
