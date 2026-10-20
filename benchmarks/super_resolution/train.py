"""Shared data loading, L1 updates and validation for the epoch training runner."""
from collections import defaultdict
from functools import lru_cache
import math
from pathlib import Path

from benchmarks.super_resolution.folds import partition, validate
from isharescreen.enhancement.models import Upscaler, sha256


def jobs(plan, models, folds):
    validate(plan)
    if not models or len(set(models)) != len(models) or set(models) - {"edsr", "imdn", "safmn"}:
        raise ValueError("Choose unique eligible models: edsr, imdn, safmn")
    if not folds or len(set(folds)) != len(folds):
        raise ValueError("Choose unique held-out folds")
    result = []
    for model in models:
        for fold in folds:
            train, val = partition(plan, fold)
            result.append(dict(model=model, fold=fold,
                               train_captures=[c["id"] for c in train],
                               validation_captures=[c["id"] for c in val]))
    return result


def enable_training(adapter):
    """Use the shared official architecture, not its inference-only NumPy wrapper."""
    model = adapter.model.train().requires_grad_(True)
    if adapter.key == "edsr":
        # Normalization constants must not become trainable after adapter unfreeze.
        model.sub_mean.requires_grad_(False)
        model.add_mean.requires_grad_(False)
    return model


def predict(model, key, tensor):
    """Differentiable RGB [0,1] interface; preserve EDSR's native 255 range."""
    return model(tensor * 255) / 255 if key == "edsr" else model(tensor)


def step(model, key, optimizer, low, reference):
    import torch
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = predict(model, key, low)
    if output.shape != reference.shape:
        raise ValueError("Training output geometry differs from paired reference")
    loss = torch.nn.functional.l1_loss(output, reference)
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    gradients = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not gradients or not torch.stack(gradients).all().item():
        raise FloatingPointError("Nonfinite or missing gradient")
    optimizer.step()
    if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all().item():
        raise FloatingPointError("Nonfinite updated parameter")
    return float(loss.detach())


class PairStore:
    def __init__(self, root, rows, packed=None):
        self.root = root
        self.rows = rows
        self.arrays = {}
        self.verified = set()
        if packed:
            import numpy as np
            for role in ("lr", "hr"):
                spec = packed[role]
                path = self.path(spec["path"])
                if sha256(path) != spec["sha256"]:
                    raise ValueError("Packed pixels changed")
                array = np.load(path, mmap_mode="r", allow_pickle=False)
                if (array.dtype != np.uint8 or array.ndim != 4 or array.shape[0] != len(rows)
                        or array.shape[-1] != 3 or list(array.shape) != spec["shape"]):
                    raise ValueError("Invalid packed array type/shape")
                self.arrays[role] = array
            return
        for row in rows:
            for role in ("hr", "lr"):
                path = self.path(row[role])
                if sha256(path) != row[f"{role}_sha256"]:
                    raise ValueError(f"Changed training pair: {path}")

    def path(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError("Pair image escapes dataset root")
        return path

    @lru_cache(maxsize=32)
    def load(self, index):
        import numpy as np
        from PIL import Image
        row = self.rows[index]
        images = []
        for role in ("lr", "hr"):
            if self.arrays:
                rgb = self.arrays[role][index]
            else:
                with Image.open(self.path(row[role])) as image:
                    rgb = np.array(image.convert("RGB"))
            if index not in self.verified:
                import hashlib
                if hashlib.sha256(rgb.tobytes()).hexdigest() != row[f"{role}_rgb_sha256"]:
                    raise ValueError("Decoded training pixels differ from sampling audit")
            images.append(rgb)
        low, reference = images
        if reference.shape != (2 * low.shape[0], 2 * low.shape[1], 3):
            raise ValueError("Not an aligned x2 training pair")
        self.verified.add(index)
        return low, reference


def perceptual_network(device):
    import lpips
    import torch
    import torchvision
    from benchmarks.super_resolution.video_perceptual import BACKBONE_SHA256, LINEAR_SHA256
    backbone = Path(torch.hub.get_dir()) / "checkpoints" / Path(
        torchvision.models.AlexNet_Weights.IMAGENET1K_V1.url).name
    linear = Path(lpips.__file__).parent / "weights/v0.1/alex.pth"
    if not backbone.is_file():
        raise FileNotFoundError("Pre-cache official AlexNet weights; no implicit downloads")
    if sha256(backbone) != BACKBONE_SHA256 or sha256(linear) != LINEAR_SHA256:
        raise ValueError("LPIPS weights differ from evaluation protocol")
    network = lpips.LPIPS(net="alex", version="0.1", verbose=False).eval().requires_grad_(False).to(device)
    return network, dict(backbone_sha256=BACKBONE_SHA256, linear_sha256=LINEAR_SHA256)


def evaluate(model, key, store, indices, device, perceptual, pixel_metrics="cpu"):
    import cv2
    import numpy as np
    import torch
    from benchmarks.super_resolution.metrics import metrics
    model.eval()
    rows = []
    with torch.inference_mode():
        for i in indices:
            low, reference = store.load(i)
            tensor = torch.from_numpy(low.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
            if key == "bicubic":
                rgb = cv2.resize(low, (reference.shape[1], reference.shape[0]), interpolation=cv2.INTER_CUBIC)
            else:
                output = predict(model, key, tensor)
                if not torch.isfinite(output).all():
                    raise FloatingPointError("Nonfinite validation output")
                rgb = output[0].clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
            def normalized(image):
                return torch.from_numpy(image.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device) / 127.5 - 1
            score = float(perceptual(normalized(reference), normalized(rgb)).item())
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite LPIPS")
            row = store.rows[i]
            if pixel_metrics == "cuda":
                from benchmarks.super_resolution.video_perceptual import tensor_rgb_metrics
                quality = tensor_rgb_metrics(reference.copy(), rgb, device)
            else:
                quality = metrics(reference, rgb)
            rows.append(dict(capture=row["capture"], frame=row["frame"], **quality, lpips=score))
    return dict(summary=summarize_samples(rows), samples=rows)


def summarize_samples(rows):
    import numpy as np
    # Equal-clip means, not a duration-weighted pool of highly correlated frames.
    captures = sorted({r["capture"] for r in rows})
    summary = {metric: float(np.mean([
        np.mean([r[metric] for r in rows if r["capture"] == capture])
        for capture in captures])) for metric in ("psnr", "ssim", "lpips")}
    return summary


def selection_indices(rows, indices, cap):
    if cap <= 0:
        return list(indices)
    grouped = defaultdict(list)
    for i in indices:
        grouped[rows[i]["capture"]].append(i)
    selected = []
    for capture in sorted(grouped):
        values = sorted(grouped[capture], key=lambda i: rows[i]["frame"])
        n = min(cap, len(values))
        offsets = [len(values)//2] if n == 1 else [i*(len(values)-1)//(n-1) for i in range(n)]
        selected.extend(values[i] for i in offsets)
    return selected
