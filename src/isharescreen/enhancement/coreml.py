"""Validated fixed-resolution adapters; Core ML stays optional and lazy-loaded."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

from .models import project_home, sha256

MODEL_ID = "quicksrnet"
CANDIDATE_IDS = ("plainusr", "rt4ksr", "efrlfn")
LABEL = "QuickSRNet-Small - Pretrained ×2 (Core ML FP16, 540p)"
CHECKPOINT_SHA256 = "d95d70f1d2366cb9c28d99f8c7aa5bb07e1ffeaf5d7e30d9c66ab0fa28c6d0f8"
# The validated Core ML Tools 9.0 conversion of the official Small x2 weights.
# Pin the entire package, not just its weight file: graph and preprocessing matter.
PACKAGE_SHA256 = {
    "Manifest.json": "513adc87239e932f6dca61d5d40385c626d5ed995bb822a5cd9a93e3f05efa38",
    "Data/com.apple.CoreML/model.mlmodel": "86282ce2c27077a203156d897a5daeb76eab1e485a4316129e5e944e6fd815a0",
    "Data/com.apple.CoreML/weights/weight.bin": "e82e2863bcea0f4778a6f1577a2877b6815d557ec8a3d69191e62d5d04cdba82",
}


def bundle_path(home: Path | None = None) -> Path:
    return (home or project_home()) / "coreml" / "quicksrnet-small-x2"


def read_candidate_bundle(key: str, home: Path | None = None) -> tuple[Path, dict]:
    """Read setup-generated conversion identity without importing ML libraries."""
    from .candidates import specification, asset_home
    spec = specification(key)
    folder = asset_home(key, home) / "coreml"
    manifest = folder / "manifest.json"
    if folder.is_symlink() or manifest.is_symlink():
        raise ValueError("Unsafe candidate Core ML bundle")
    data = json.loads(manifest.read_text())
    if (data["schema"] != "isharescreen_candidate_coreml_v1" or data["model"] != key
            or data["scale"] != 2 or spec["scale"] != 2
            or data["official_checkpoint_sha256"] != spec["files"]["weights.pth"]["sha256"]
            or data["source_revision"] != spec["revision"]
            or data["compute_units"] not in ("ALL", "CPU_AND_GPU", "CPU_AND_NE")
            or data["precision"] not in ("fp16", "fp32") or data["input"] != [1, 3, 540, 960]
            or data["output"] != [1, 3, 1080, 1920]
            or len(data["validation"]) < 3
            or len({row["image_sha256"] for row in data["validation"]}) != len(data["validation"])
            or any(not isinstance(row["image_sha256"], str)
                   or not re.fullmatch(r"[0-9a-f]{64}", row["image_sha256"])
                   or not 0 <= row["max_rgb_delta"] <= 1 for row in data["validation"])
            or type(data["parameters"]) is not int or data["parameters"] < 1):
        raise ValueError("Candidate Core ML conversion identity or validation mismatch")
    for name, digest in data["package_sha256"].items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Unsafe candidate Core ML package entry")
    if "Data/com.apple.CoreML/weights/weight.bin" not in data["package_sha256"]:
        raise ValueError("Missing candidate Core ML weight identity")
    return folder, data


def verify_package(package: Path, expected: dict | None = None) -> None:
    """Reject changed graphs, weights, extra files, and symlinked assets."""
    if not package.is_dir() or package.is_symlink() or package.parent.is_symlink():
        raise ValueError("Missing or unsafe Core ML package")
    paths = list(package.rglob("*"))
    if any(p.is_symlink() for p in paths):
        raise ValueError("Core ML package must not contain symlinks")
    files = {str(p.relative_to(package)): p for p in paths if p.is_file()}
    expected = PACKAGE_SHA256 if expected is None else expected
    if files.keys() != expected.keys():
        raise ValueError("Core ML package file set differs from the validated conversion")
    for name, digest in expected.items():
        if sha256(files[name]) != digest:
            raise ValueError(f"Core ML package hash mismatch: {name}")


def availability(home: Path | None = None, candidate: str | None = None) -> dict:
    reason = ""
    if sys.platform != "darwin":
        reason = "Core ML prediction requires macOS 13 or newer"
    elif int(platform.mac_ver()[0].split(".")[0] or 0) < 13:
        reason = "This Core ML package requires macOS 13 or newer"
    elif (importlib.util.find_spec("coremltools") is None
          or importlib.util.find_spec("cv2") is None):
        reason = "Install the optional enhancement-coreml dependencies"
    elif candidate is None and not (bundle_path(home) / "model.mlpackage").is_dir():
        reason = "Validated QuickSRNet Core ML package is not installed"
    if candidate is None:
        return dict(id=MODEL_ID, label=LABEL, available=not reason, reason=reason)
    from .candidates import specification
    spec = specification(candidate)
    label = f"{spec['label']} - Pretrained ×2 (Core ML, 540p)"
    if not reason:
        try:
            folder, data = read_candidate_bundle(candidate, home)
            label = f"{spec['label']} - Pretrained ×2 (Core ML {data['precision'].upper()}, 540p)"
            if not (folder / "model.mlpackage").is_dir():
                raise ValueError("Core ML package is missing")
        except (OSError, KeyError, TypeError, ValueError) as error:
            reason = f"Candidate Core ML bundle unavailable: {error}"
    return dict(id=f"{candidate}-coreml", label=label, available=not reason, reason=reason)


class CoreMLUpscaler:
    """Validated RGB uint8 960x540 -> 1920x1080, blocking prediction.

    FP32 tensor I/O and internal precision are properties of each validated
    conversion. ALL permits CPU, GPU and Neural Engine; it does not prove actual
    operation placement. No PyTorch inference or implicit resizing/fallback.
    inference_ms measures predict(), including its internal data transfers;
    download_ms measures the subsequent CPU RGB conversion, not a GPU barrier.
    Instances belong to one inference worker; scratch buffers are not shared
    between workers. Returned RGB frames own their storage independently.
    """

    def __init__(self, device="auto", home: Path | None = None, precision="fp32", candidate=None):
        if device not in ("auto", "coreml"):
            raise ValueError("This model uses Core ML; unset ISS_SR_DEVICE or set it to auto")
        if precision != "fp32":
            raise ValueError("Core ML has fixed FP32 I/O; internal precision is bound to the installed conversion")
        info = availability(home, candidate)
        if not info["available"]:
            raise RuntimeError(info["reason"])
        if candidate is None:
            folder, data = bundle_path(home), dict(package_sha256=PACKAGE_SHA256, compute_units="ALL",
                parameters=22860, precision="fp16", official_checkpoint_sha256=CHECKPOINT_SHA256)
        else:
            folder, data = read_candidate_bundle(candidate, home)
        package = folder / "model.mlpackage"
        verify_package(package, data["package_sha256"])
        import coremltools as ct
        self.model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, data["compute_units"]))
        self._output_float = np.empty((3, 1080, 1920), np.float32)
        self._output_u8 = np.empty((3, 1080, 1920), np.uint8)
        self.key, self.device = info["id"], f"coreml-{data['compute_units'].lower()}"
        weights = package / "Data/com.apple.CoreML/weights/weight.bin"
        self.provenance = dict(id=self.key, label=info["label"], variant="pretrained", backend="coreml",
                               device=self.device, precision=data["precision"], dtype="float16" if data["precision"] == "fp16" else "float32", parameters=data["parameters"],
                               checkpoint=str(weights), checkpoint_sha256=data["package_sha256"][str(weights.relative_to(package))],
                               official_checkpoint_sha256=data["official_checkpoint_sha256"])

    def __call__(self, rgb: np.ndarray) -> tuple[np.ndarray, dict]:
        if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8 or rgb.shape != (540, 960, 3):
            raise ValueError("Core ML enhancement requires RGB uint8 960×540 input in a non-strict session")
        t0 = time.perf_counter()
        source = np.ascontiguousarray((rgb.astype(np.float32) / 255).transpose(2, 0, 1)[None])
        t1 = time.perf_counter()
        output = self.model.predict({"image": source})["output"]
        t2 = time.perf_counter()
        if (not isinstance(output, np.ndarray) or output.dtype != np.float32
                or output.shape != (1, 3, 1080, 1920) or not np.isfinite(output).all()):
            raise ValueError("Core ML enhancement returned invalid pixels; original rendering restored")
        import cv2
        # Preserve FP32 multiplication and NumPy round-to-even before casting.
        # Quantize contiguous planes, then interleave with a vectorized merge;
        # strided NumPy RGB packing is substantially slower at 1080p.
        np.multiply(output[0], np.float32(255), out=self._output_float)
        np.rint(self._output_float, out=self._output_float)
        np.clip(self._output_float, 0, 255, out=self._output_float)
        np.copyto(self._output_u8, self._output_float, casting="unsafe")
        # Do not pass a reusable destination: an earlier frame can still be
        # waiting for render-thread upload while the next prediction finishes.
        out = cv2.merge(tuple(self._output_u8))
        t3 = time.perf_counter()
        return out, dict(prepare_ms=(t1-t0)*1000, inference_ms=(t2-t1)*1000,
                         download_ms=(t3-t2)*1000, total_ms=(t3-t0)*1000)
