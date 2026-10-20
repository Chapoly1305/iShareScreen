"""Per-model pinned assets and loaders, independent of frozen training sources.

Discovery reads metadata only. Model definitions remain in user-owned official
assets; optional ML libraries are imported only when explicitly loading a model.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import time

import numpy as np

from ..models import project_home, sha256

IDS = ("plainusr", "quicksrnet", "rt4ksr", "efrlfn", "casr", "spanv2")


def specification(key: str) -> dict:
    if key not in IDS:
        raise ValueError(f"Unknown candidate model: {key}")
    spec = json.loads((Path(__file__).parent / key / "model.json").read_text())
    if spec["id"] != key or spec["scale"] not in (2, 4):
        raise ValueError("Candidate model identity mismatch")
    for name, asset in spec["files"].items():
        path = PurePosixPath(name)
        if (path.is_absolute() or ".." in path.parts or not asset["url"].startswith("https://")
                or not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"])):
            raise ValueError("Unsafe candidate asset path or URL")
    return spec


def asset_home(key: str, home: Path | None = None) -> Path:
    if key not in IDS:
        raise ValueError(f"Unknown candidate model: {key}")
    return (home or project_home()) / "candidates" / key


def verify_assets(key: str, home: Path | None = None) -> dict:
    spec, root = specification(key), asset_home(key, home)
    for name, asset in spec["files"].items():
        path = root / name
        parents = (root / parent for parent in Path(name).parents)
        if (not path.is_file() or any(p.is_symlink() for p in (path, root, *parents))
                or sha256(path) != asset["sha256"]):
            raise ValueError(f"Missing, unsafe or changed {key} asset: {name}")
    return spec


def torch_id(key: str) -> str:
    return "quicksrnet-torch" if key == "quicksrnet" else key


def availability(home: Path | None = None) -> list[dict]:
    result = []
    for key in IDS:
        spec, reason = specification(key), ""
        if spec["scale"] != 2:
            reason = "Official weights are ×4; offline native-scale evaluation only"
        elif any(importlib.util.find_spec(module) is None for module in ("torch", "cv2")):
            reason = "Install the optional video enhancement dependencies"
        elif key == "rt4ksr" and importlib.util.find_spec("torchvision") is None:
            reason = "RT4KSR requires torchvision matching the installed PyTorch"
        elif not all((asset_home(key, home) / name).is_file() for name in spec["files"]):
            reason = "Pinned official candidate assets are not installed"
        result.append(dict(id=torch_id(key), label=f"{spec['label']} - Pretrained ×{spec['scale']} (PyTorch)",
                           available=not reason, reason=reason))
    return result


def load_network(key: str, home: Path | None = None):
    """Strict-load official weights before any deployment-only transformation."""
    import torch
    spec = verify_assets(key, home)
    root = asset_home(key, home)
    module = importlib.import_module(f"{__name__}.{key}.loader")
    model = module.build(root / "source")
    with (root / "weights.pth").open("rb") as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
        if digest.hexdigest() != spec["files"]["weights.pth"]["sha256"]:
            raise ValueError("Candidate checkpoint changed during loading")
        handle.seek(0)
        allowed = [torch.optim.Adam, dict, defaultdict] if key == "quicksrnet" else []
        with torch.serialization.safe_globals(allowed):
            state = torch.load(handle, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {name.removeprefix("module."): value for name, value in state.items()}
    if not state or not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all().item() for v in state.values()):
        raise ValueError("Candidate checkpoint must contain finite tensors")
    torch.nn.Module.load_state_dict(model, state, strict=True)
    model.eval().requires_grad_(False)
    if hasattr(module, "deploy"):
        model = module.deploy(model)
    return model.eval().requires_grad_(False), spec


class TorchUpscaler:
    """Native-scale pretrained inference with the shared live timing contract."""

    def __init__(self, key, device="auto", home=None, precision="fp32"):
        import torch
        self.key = key
        self.model, self.spec = load_network(key, home)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        if precision not in ("fp32", "fp16", "amp-fp16") or (precision != "fp32" and not device.startswith("cuda")):
            raise ValueError("Use FP32 on CPU/MPS; FP16 and AMP require CUDA")
        self.device, self.precision = device, precision
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        self.model.to(device=device, dtype=self.dtype)
        self.provenance = dict(id=torch_id(key), label=self.spec["label"], variant="pretrained",
            backend="pytorch", device=device, precision=precision, scale=self.spec["scale"],
            parameters=sum(p.numel() for p in self.model.parameters()), source_revision=self.spec["revision"],
            checkpoint=str(asset_home(key, home)/"weights.pth"),
            checkpoint_sha256=self.spec["files"]["weights.pth"]["sha256"])

    def sync(self):
        if self.device == "mps":
            import torch
            torch.mps.synchronize()
        elif self.device.startswith("cuda"):
            import torch
            torch.cuda.synchronize(self.device)

    def __call__(self, rgb):
        import torch
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected H×W×3 RGB uint8")
        self.sync()
        t0 = time.perf_counter()
        x = torch.from_numpy(np.ascontiguousarray((rgb.astype(np.float32)/255).transpose(2, 0, 1)[None]))
        t1 = time.perf_counter()
        x = x.to(device=self.device, dtype=self.dtype)
        self.sync()
        t2 = time.perf_counter()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=self.precision == "amp-fp16"):
            output = self.model(x)
        if output.shape != (1, 3, rgb.shape[0]*self.spec["scale"], rgb.shape[1]*self.spec["scale"]) or not torch.isfinite(output).all().item():
            raise ValueError("Candidate model returned invalid pixels")
        self.sync()
        t3 = time.perf_counter()
        out = (output[0].float()*255).round().clamp(0, 255).to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
        t4 = time.perf_counter()
        return out, dict(prepare_ms=(t1-t0)*1000, upload_ms=(t2-t1)*1000, inference_ms=(t3-t2)*1000,
                         download_ms=(t4-t3)*1000, total_ms=(t4-t0)*1000)
