"""Train-only channel balancing before INT8 quantization of the same EDSR.

Only the 32 residual-block convolutions are quantized. Head, body output,
tail, additions and activations remain FP16. This is fake quantization for
accuracy screening, with no timing or hardware-acceleration claim.
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

from benchmarks.super_resolution.profile_inference_optimization import adapter, amp, digest, forward, guard, save


def quantize(x, scale, mode):
    q = x.float() / scale
    if mode == "int8":
        q = q.round().clamp(-127, 127)
    else:
        q = q.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    return (q * scale).to(x.dtype)


class QuantizedConv(nn.Module):
    def __init__(self, conv, maximum, mode):
        super().__init__()
        self.conv = conv
        self.mode = "int8"
        alpha = float(mode)
        activation = torch.tensor(maximum, device=conv.weight.device, dtype=torch.float32)
        weight = conv.weight.float()
        magnitude = weight.abs().amax(dim=(0, 2, 3)).clamp_min(1e-12)
        balance = activation.clamp_min(1e-12).pow(alpha) / magnitude.pow(1-alpha)
        self.register_buffer("balance", balance.reshape(1,64,1,1))
        self.register_buffer("scale", (activation / balance).max().clamp_min(1e-12) / 127)
        balanced_weight = weight * balance.reshape(1,64,1,1)
        weight_scale = balanced_weight.abs().amax(dim=(1,2,3),keepdim=True).clamp_min(1e-12) / 127
        self.conv.weight.copy_(quantize(balanced_weight,weight_scale,"int8").half())

    def forward(self, x):
        return self.conv(quantize(x.float() / self.balance, self.scale, "int8").half())


def image_tensor(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).cuda().float() / 255


def pixels(tensor):
    native = tensor[0].float().mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).contiguous().cpu().numpy()
    return cv2.resize(native, (1920, 1080), interpolation=cv2.INTER_AREA)


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--validation", type=Path, required=True)
    p.add_argument("--training", type=Path, required=True)
    p.add_argument("--calibration-selection", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--alphas", type=float, nargs="+", default=[0.5,0.75])
    p.add_argument("--protect-terminal-block", action="store_true")
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(1); cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    result = dict(complete=False, started=time.time(), gpu=guard(), script_sha256=digest(Path(__file__)),
                  scope=__doc__, quantization="Per-input-channel balancing, alpha=0.5 or 0.75; static max-absolute INT8 activations; per-output-channel INT8 weights; no retraining", modes={})
    save(a.output, result)
    try:
        ckpt = a.weights / "checkpoints/edsr.pt"
        item = next(r for r in json.loads((a.weights / "PLAN.json").read_text())["models"] if r["model"] == "edsr")
        assert digest(ckpt) == item["sha256"]
        result["checkpoint"] = item
        model = adapter("edsr", "cpu").model.eval().requires_grad_(False)
        nn.Module.load_state_dict(model, torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"], strict=True)
        base = copy.deepcopy(model).half().cuda()
        names = [n for n, m in base.named_modules() if isinstance(m, nn.Conv2d) and n.startswith("body.") and len(n.split(".")) == 4]
        assert len(names) == 32, names
        maxima = {n: torch.zeros(64, device="cuda") for n in names}
        handles = []
        for n in names:
            def calibrate(module, args, name=n):
                maxima[name].copy_(torch.maximum(maxima[name], args[0].float().abs().amax(dim=(0,2,3))))
            handles.append(base.get_submodule(n).register_forward_pre_hook(calibrate))
        selection = json.loads(a.calibration_selection.read_text())
        manifest = a.training / "DATA.json"
        assert digest(manifest) == selection["source_sha256"]
        actual = {r["id"]: r for r in json.loads(manifest.read_text())["rows"]}
        for index, row in enumerate(selection["rows"]):
            assert row["split"] == "train" and actual[row["id"]]["split"] == "train"
            path = a.training / row["lr"]
            assert digest(path) == row["lr_sha256"] == actual[row["id"]]["lr_sha256"]
            base((image_tensor(path) * 255).half())
            if (index + 1) % 12 == 0:
                print("calibrated", index + 1, flush=True)
        for h in handles:
            h.remove()
        maxima = {n: v.cpu().tolist() for n, v in maxima.items()}
        result["calibration"] = dict(selection_sha256=digest(a.calibration_selection), samples=len(selection["rows"]), split="train", maxima=maxima)
        models = {"fp16_control": base}
        result["alphas"] = a.alphas
        result["protect_terminal_block"] = a.protect_terminal_block
        for alpha in a.alphas:
            mode = str(alpha)
            candidate = copy.deepcopy(base)
            for n in names:
                if a.protect_terminal_block and n.startswith("body.15."):
                    continue
                parent, child = n.rsplit(".", 1)
                candidate.get_submodule(parent)._modules[child] = QuantizedConv(candidate.get_submodule(n), maxima[n], mode)
            models["int8_alpha_" + mode] = candidate
        model.cuda()
        runs = [r for r in json.loads((a.validation / "INPUTS.json").read_text())["runs"] if r["config"] == "720p-600k-420"]
        rows = []
        for run in runs:
            for sample in run["samples"]:
                for kind in ("lr", "hr"):
                    assert digest(a.validation / sample[kind]) == sample[kind + "_sha256"]
                x = image_tensor(a.validation / sample["lr"])
                target = np.asarray(Image.open(a.validation / sample["hr"]).convert("RGB"))
                images = {"fp32": pixels(forward(model, "edsr", x))}
                with amp():
                    images["baseline_amp"] = pixels(forward(model, "edsr", x))
                for mode, candidate in models.items():
                    out = candidate((x * 255).half()).float() / 255
                    assert torch.isfinite(out).all().item(), mode
                    images[mode] = pixels(out)
                row = dict(clip=run["clip"], frame=sample["frame"], psnr={k: float(10 * np.log10(255**2 / np.mean((v.astype(np.float64) - target.astype(np.float64))**2))) for k, v in images.items()})
                row["max_rgb_vs_fp32"] = {k: int(np.abs(v.astype(np.int16) - images["fp32"].astype(np.int16)).max()) for k, v in images.items()}
                rows.append(row)
            print("quality", len(rows), flush=True)
        assert len(rows) == 72
        result["records"] = rows
        for mode in models:
            metrics = {}
            for reference in ("fp32", "baseline_amp"):
                deltas = [r["psnr"][mode] - r["psnr"][reference] for r in rows]
                metrics[reference] = dict(mean_delta=float(np.mean(deltas)), worst_delta=min(deltas))
            result["modes"][mode] = dict(mean_psnr=float(np.mean([r["psnr"][mode] for r in rows])), metrics=metrics,
                passed=all(m["mean_delta"] >= -.05 and m["worst_delta"] >= -.1 for m in metrics.values()))
        result.update(complete=True, finished=time.time(), gpu_end=guard())
        save(a.output, result)
        print(json.dumps(result["modes"], indent=2), flush=True)
    except BaseException as exc:
        result.update(error=repr(exc), failed=time.time()); save(a.output, result); raise


if __name__ == "__main__":
    main()
