"""One inference adapter shared by offline evaluation and the live viewer.

Architectures are imported unchanged from pinned official checkouts. Torch stays
optional for the ordinary viewer. No implicit downloads or untrained fallback.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import importlib.util
import os
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace

import numpy as np

LABELS = {
    "off": "Original (native scaling)",
    "bilinear": "Bilinear ×2", "bicubic": "Bicubic ×2",
    "sharpen": "Bicubic + sharpening ×2",
    "edsr": "EDSR-baseline ×2", "imdn": "IMDN ×2",
    "ecbsr": "ECBSR m4c16 ×2 (Y)", "safmn": "SAFMN ×2",
}
CHECKPOINTS = {"edsr": "edsr_baseline_x2.pt", "imdn": "IMDN_x2.pth",
               "ecbsr": "ecbsr_x2_m4c16.pth", "safmn": "SAFMN_DF2K_x2.pth"}
REPOS = {"edsr": "EDSR-PyTorch", "imdn": "IMDN", "ecbsr": "ECBSR", "safmn": "SAFMN"}
_IMPORT_LOCK = threading.Lock()


def project_home() -> Path:
    """Model assets are user-owned data, independent of checkout/wheel location."""
    return Path(os.environ.get("ISS_SR_HOME", Path.home() / ".cache" / "isharescreen" / "super_resolution")).expanduser()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def availability(home: Path | None = None) -> list[dict]:
    home = home or project_home()
    torch_ok = importlib.util.find_spec("torch") is not None
    cv_ok = importlib.util.find_spec("cv2") is not None
    result = []
    for key, label in LABELS.items():
        reason = ""
        if key != "off" and not cv_ok:
            reason = "Install the optional video enhancement dependencies"
        if key in CHECKPOINTS:
            if not torch_ok:
                reason = "PyTorch is not installed"
            elif not (home / "vendor" / REPOS[key]).is_dir():
                reason = "Official source checkout is missing"
            elif not (home / "checkpoints" / CHECKPOINTS[key]).is_file():
                reason = "Official ×2 checkpoint is not installed"
        result.append({"id": key, "label": label, "available": not reason, "reason": reason})
    return result


@contextmanager
def _packages(paths: dict[str, Path]):
    """Isolate legacy absolute import names without executing package __init__."""
    with _IMPORT_LOCK:
        roots = {name.split(".")[0] for name in paths}
        saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in roots}
        for k in list(saved):
            del sys.modules[k]
        try:
            for name, path in paths.items():
                package = ModuleType(name)
                package.__path__ = [str(path)]
                sys.modules[name] = package
            yield
        finally:
            for k in list(sys.modules):
                if k.split(".")[0] in roots:
                    del sys.modules[k]
            sys.modules.update(saved)


class Upscaler:
    def __init__(self, key: str, device: str = "auto", home: Path | None = None, precision: str = "fp32"):
        if key not in LABELS or key == "off":
            raise ValueError(f"Not an upscaler: {key}")
        self.key, self.home, self.model = key, home or project_home(), None
        self.device = "cpu"
        self.provenance = {"id": key, "label": LABELS[key]}
        info = next(x for x in availability(self.home) if x["id"] == key)
        if not info["available"]:
            raise RuntimeError(info["reason"])
        if key not in CHECKPOINTS:
            return
        import torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else (
                "mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        if precision not in ("fp32","fp16","amp-fp16"):
            raise ValueError(f"Unsupported precision: {precision}")
        if precision != "fp32" and not device.startswith("cuda"):
            raise ValueError("FP16 mode requires CUDA; use FP32 on CPU/MPS")
        if precision == "fp16" and key == "imdn":
            raise ValueError("IMDN pure FP16 overflows spatial reductions at 540p; use amp-fp16")
        self._use_amp = precision == "amp-fp16"
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        repo = self.home / "vendor" / REPOS[key]
        if key == "edsr":
            with _packages({"model": repo / "src/model"}):
                cls = importlib.import_module("model.edsr").EDSR
                model = cls(SimpleNamespace(n_resblocks=16, n_feats=64, scale=[2],
                                           n_colors=3, rgb_range=255, res_scale=1))
        elif key == "imdn":
            with _packages({"model": repo / "model"}):
                model = importlib.import_module("model.architecture").IMDN(upscale=2)
        elif key == "ecbsr":
            with _packages({"models": repo / "models"}):
                model = importlib.import_module("models.ecbsr").ECBSR(4, 16, False, "prelu", 2, 1)
        else:
            with _packages({"basicsr": repo / "basicsr", "basicsr.utils": repo / "basicsr/utils",
                            "basicsr.archs": repo / "basicsr/archs"}):
                model = importlib.import_module("basicsr.archs.safmn_arch").SAFMN(
                    dim=36, n_blocks=8, ffn_scale=2.0, upscaling_factor=2)
        checkpoint = self.home / "checkpoints" / CHECKPOINTS[key]
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if "params" in state:
            state = state["params"]
        elif "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        # Bypass EDSR's custom permissive loader: require every key and shape.
        torch.nn.Module.load_state_dict(model, state, strict=True)
        self.model = model.eval().requires_grad_(False).to(device=device,dtype=self.dtype)
        self.provenance.update(checkpoint_sha256=sha256(checkpoint),
                               parameters=sum(p.numel() for p in model.parameters()),
                               device=device, dtype=str(self.dtype).removeprefix("torch."),
                               precision=precision, checkpoint=str(checkpoint))

    def sync(self):
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize(self.device)
        elif self.device == "mps":
            import torch
            torch.mps.synchronize()

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        import cv2
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected H×W×3 RGB uint8")
        self.sync()
        t0 = time.perf_counter()
        h, w = rgb.shape[:2]
        if self.model is None:
            interpolation = cv2.INTER_LINEAR if self.key == "bilinear" else cv2.INTER_CUBIC
            out = cv2.resize(rgb, (2*w, 2*h), interpolation=interpolation)
            if self.key == "sharpen":
                out = cv2.addWeighted(out, 1.5, cv2.GaussianBlur(out, (0, 0), 1.0), -0.5, 0)
            elapsed = (time.perf_counter() - t0) * 1000
            return out, {"total_ms": elapsed, "prepare_ms": elapsed, "upload_ms": 0,
                         "inference_ms": 0, "download_ms": 0}
        import torch
        source = rgb.astype(np.float32)
        if self.key == "ecbsr":
            from skimage.color import rgb2ycbcr
            # Official datas/benchmark.py + utils.ndarray2tensor keep the
            # digital Y range (16–235); there is no normalization by 255.
            source = rgb2ycbcr(rgb)[..., :1].astype(np.float32)
        elif self.key != "edsr":
            source /= 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(source.transpose(2, 0, 1))).unsqueeze(0)
        t1 = time.perf_counter()
        tensor = tensor.to(device=self.device,dtype=self.dtype)
        self.sync()
        t2 = time.perf_counter()
        # SAFMN multiscale pooling needs at least 8 pixels per input axis.
        if min(h, w) < 8:
            raise ValueError("Input must be at least 8×8")
        with torch.inference_mode(), torch.autocast("cuda",dtype=torch.float16,enabled=getattr(self,"_use_amp",False)):
            output = self.model(tensor)
        if not torch.isfinite(output).all().item():
            raise FloatingPointError(f"{self.key} returned non-finite pixels; original rendering restored")
        self.sync()
        t3 = time.perf_counter()
        if self.device != "cpu" and self.key != "ecbsr":
            # Quantize before download: transfer 3 bytes/px instead of 12.
            # The FP32 forward pass and original round/clamp convention remain.
            quantized = output[0].float() * (1.0 if self.key == "edsr" else 255.0)
            out = quantized.round().clamp(0,255).to(torch.uint8).permute(1,2,0).contiguous().cpu().numpy()
            t4 = time.perf_counter()
            return out, {"prepare_ms":(t1-t0)*1000,"upload_ms":(t2-t1)*1000,
                         "inference_ms":(t3-t2)*1000,"download_ms":(t4-t3)*1000,"total_ms":(t4-t0)*1000}
        out = output[0].detach().cpu().numpy().transpose(1, 2, 0)
        if self.key == "ecbsr":
            from skimage.color import rgb2ycbcr, ycbcr2rgb
            color = cv2.resize(rgb2ycbcr(rgb), (2*w, 2*h), interpolation=cv2.INTER_CUBIC)
            color[..., 0] = out[..., 0]
            out = ycbcr2rgb(color) * 255.0
        elif self.key != "edsr":
            out = out * 255.0
        out = np.ascontiguousarray(np.clip(np.rint(out), 0, 255).astype(np.uint8))
        t4 = time.perf_counter()
        return out, {"prepare_ms": (t1-t0)*1000, "upload_ms": (t2-t1)*1000,
                     "inference_ms": (t3-t2)*1000, "download_ms": (t4-t3)*1000,
                     "total_ms": (t4-t0)*1000}
