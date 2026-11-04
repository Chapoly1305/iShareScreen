"""Diagnose activation saturation without changing train-only calibration."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from benchmarks.super_resolution.profile_edsr_smooth_int8 import QuantizedConv, image_tensor
from benchmarks.super_resolution.profile_inference_optimization import adapter, digest, guard, save


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("weights", "validation", "calibration", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    result = dict(complete=False, started=time.time(), gpu=guard(), scope=__doc__,
                  script_sha256=digest(Path(__file__)), calibration_sha256=digest(a.calibration))
    save(a.output, result)
    cal = json.loads(a.calibration.read_text())
    assert cal["complete"] and cal["calibration"]["split"] == "train"
    ckpt = a.weights / "checkpoints/edsr.pt"
    item = next(r for r in json.loads((a.weights / "PLAN.json").read_text())["models"] if r["model"] == "edsr")
    assert digest(ckpt) == item["sha256"]
    base = adapter("edsr", "cpu").model.eval().requires_grad_(False)
    nn.Module.load_state_dict(base, torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"], strict=True)
    base = base.half().cuda()
    candidate = copy.deepcopy(base)
    names = [n for n in cal["calibration"]["maxima"] if not n.startswith("body.15.")]
    layers = {}
    for name in names:
        parent, child = name.rsplit(".", 1)
        layer = QuantizedConv(candidate.get_submodule(name), cal["calibration"]["maxima"][name], ".75")
        candidate.get_submodule(parent)._modules[child] = layer
        layers[name] = layer
    current = {}
    for mode, model in (("fp16", base), ("fake_int8", candidate)):
        for name in names:
            def inspect(module, args, mode=mode, name=name):
                q = args[0].float() / layers[name].balance / layers[name].scale
                # Rounding itself stays in range until halfway beyond the rail.
                current[mode][name] = dict(elements=q.numel(),
                    outside_positive=int((q > 127.5).sum().item()),
                    outside_negative=int((q < -127.5).sum().item()),
                    max_quantizer_units=float(q.abs().max().item()))
            model.get_submodule(name).register_forward_pre_hook(inspect)
    records = []
    runs = [r for r in json.loads((a.validation / "INPUTS.json").read_text())["runs"] if r["config"] == "720p-600k-420"]
    for run in runs:
        for sample in run["samples"]:
            path = a.validation / sample["lr"]
            assert digest(path) == sample["lr_sha256"]
            x = (image_tensor(path) * 255).half()
            current = dict(fp16={}, fake_int8={})
            base(x); candidate(x)
            records.append(dict(clip=run["clip"], frame=sample["frame"], modes=current))
        print("audited", len(records), flush=True)
    result["records"] = records
    result["summary"] = {}
    for mode in ("fp16", "fake_int8"):
        values = [v for r in records for v in r["modes"][mode].values()]
        result["summary"][mode] = dict(
            saturated_elements=sum(v["outside_positive"] + v["outside_negative"] for v in values),
            total_elements=sum(v["elements"] for v in values),
            maximum_quantizer_units=max(v["max_quantizer_units"] for v in values),
            frames_with_saturation=sum(any(v["outside_positive"] + v["outside_negative"] for v in r["modes"][mode].values()) for r in records))
    result.update(complete=True, finished=time.time(), gpu_end=guard())
    save(a.output, result)
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
