"""Live model choices, separate from the source-bound training adapter.

Fine-tuned bundles contain only weights and an identity manifest. They never
replace official assets, and discovery does not import optional ML packages.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
import platform
import re
import sys

from . import models
from . import coreml
from . import candidates
from . import execution

log = logging.getLogger(__name__)
MODELS = ("edsr", "imdn", "safmn")
SCHEMA = "isharescreen_live_weights_v1"
MPS_FP16_ID = "quicksrnet-mps-fp16"
MPS_FP16_LABEL = "QuickSRNet-Small - Pretrained ×2 (PyTorch MPS FP16, 540p)"


def bundle_name(model: str, fold: int, epoch: int, role: str) -> str:
    if (model not in MODELS or type(fold) is not int or fold < 0
            or type(epoch) is not int or epoch < 0 or role not in ("current", "best")):
        raise ValueError("Invalid live model identity")
    return f"{model}-fold-{fold}-epoch-{epoch}-{role}"


def read_bundle(folder: Path) -> dict:
    """Read metadata only; weights are verified again in the inference worker."""
    if folder.is_symlink():
        raise ValueError("Live model bundles must not be symlinks")
    for name in ("manifest.json", "weights.pt"):
        if not (folder / name).is_file() or (folder / name).is_symlink():
            raise ValueError(f"Missing or unsafe live model file: {name}")
    spec = json.loads((folder / "manifest.json").read_text())
    if spec["schema"] != SCHEMA or folder.name != bundle_name(
            spec["model"], spec["fold"], spec["epoch"], spec["role"]):
        raise ValueError("Live model manifest identity differs from its directory")
    for field in ("weights_sha256", "official_checkpoint_sha256", "source_checkpoint_sha256", "protocol_binding"):
        if not isinstance(spec[field], str) or not re.fullmatch(r"[0-9a-f]{64}", spec[field]):
            raise ValueError(f"Invalid live model hash: {field}")
    return spec


def _label(spec: dict) -> str:
    role = "current" if spec["role"] == "current" else "selected best"
    return (f"{spec['model'].upper()} - Fine-tuned ×2 "
            f"(epoch {spec['epoch']}, fold {spec['fold']}, {role})")


def availability(home: Path | None = None) -> list[dict]:
    home = home or models.project_home()
    choices = models.availability(home)
    for item in choices:
        if item["id"] in MODELS:
            item["label"] = f"{item['id'].upper()} - Pretrained ×2"
    choices.append(coreml.availability(home))
    choices.extend(candidates.availability(home))
    choices.extend(coreml.availability(home, key) for key in coreml.CANDIDATE_IDS)
    # Metadata-only discovery: constructing the selected model verifies its
    # official source/weights again and requires an actual MPS device.
    quick = next(item for item in choices if item["id"] == "quicksrnet-torch")
    reason = quick["reason"]
    if sys.platform != "darwin" or platform.machine() != "arm64":
        reason = "QuickSRNet MPS FP16 requires Apple Silicon macOS"
    choices.append(dict(id=MPS_FP16_ID,label=MPS_FP16_LABEL,available=not reason,reason=reason))
    base = {item["id"]: item for item in choices}
    root = home / "fine-tuned"
    if not root.exists():
        return execution.filter_choices(choices)
    try:
        folders = sorted(root.iterdir())
    except OSError as error:
        log.warning("Cannot discover fine-tuned models: %s", error)
        return execution.filter_choices(choices)
    for folder in folders:
        if not re.fullmatch(r"[a-z0-9-]+", folder.name) or not folder.is_dir():
            continue  # Ignore staging directories and unrelated files.
        item = dict(id=f"ft-{folder.name}", label=folder.name, available=False, reason="")
        try:
            spec = read_bundle(folder)
            item.update(label=_label(spec), available=base[spec["model"]]["available"],
                        reason=base[spec["model"]]["reason"])
        except (OSError, ValueError, KeyError, TypeError) as error:
            item["reason"] = f"Invalid fine-tuned bundle: {error}"
        choices.append(item)
    return execution.filter_choices(choices)


def load(key: str, device: str = "auto", home: Path | None = None, precision: str = "fp32"):
    device = execution.resolve_device(key, device)
    return execution.validate_adapter(_load(key, device, home, precision))


def _load(key: str, device: str, home: Path | None, precision: str):
    """Use the shared adapter's preprocessing with a strictly identified state."""
    home = home or models.project_home()
    if key == MPS_FP16_ID:
        if device not in ("auto", "mps"):
            raise ValueError("QuickSRNet MPS FP16 requires the MPS GPU")
        if sys.platform != "darwin" or platform.machine() != "arm64":
            raise ValueError("QuickSRNet MPS FP16 requires Apple Silicon macOS")
        if precision not in ("fp32", "fp16"):
            raise ValueError("QuickSRNet MPS FP16 has fixed FP16 GPU execution")
        # Precision is bound to this explicit choice, like the Core ML variants.
        # The ordinary Torch IDs continue to use their requested precision.
        adapter = candidates.TorchUpscaler("quicksrnet",device="mps",home=home,precision="fp16")
        adapter.provenance.update(id=MPS_FP16_ID,label=MPS_FP16_LABEL,input_shape=[540,960,3])
        return adapter
    if key == coreml.MODEL_ID:
        return coreml.CoreMLUpscaler(device=device, home=home, precision=precision)
    if key.endswith("-coreml") and key.removesuffix("-coreml") in coreml.CANDIDATE_IDS:
        return coreml.CoreMLUpscaler(device=device, home=home, precision=precision,
                                    candidate=key.removesuffix("-coreml"))
    candidate = "quicksrnet" if key == "quicksrnet-torch" else key
    if candidate in candidates.IDS:
        if candidates.specification(candidate)["scale"] != 2:
            raise ValueError("Native ×4 candidates are offline-only; the live viewer requires ×2")
        return candidates.TorchUpscaler(candidate, device=device, home=home, precision=precision)
    if not key.startswith("ft-"):
        adapter = models.Upscaler(key, device=device, home=home, precision=precision)
        if key in MODELS:
            adapter.provenance.update(label=f"{key.upper()} - Pretrained ×2", variant="pretrained")
        return adapter
    name = key.removeprefix("ft-")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        raise ValueError("Unsafe live model ID")
    folder = home / "fine-tuned" / name
    spec = read_bundle(folder)
    checkpoint = folder / "weights.pt"
    # Hash and deserialize the same open file, before allocating a GPU model.
    with checkpoint.open("rb") as handle:
        import hashlib
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        if digest.hexdigest() != spec["weights_sha256"]:
            raise ValueError("Fine-tuned weights hash mismatch")
        handle.seek(0)
        import torch
        state = torch.load(handle, map_location="cpu", weights_only=True)
    if (not isinstance(state, dict) or not state
            or not all(isinstance(v, torch.Tensor) and torch.isfinite(v).all().item() for v in state.values())):
        raise ValueError("Fine-tuned weights must be a finite state dictionary")
    adapter = models.Upscaler(spec["model"], device=device, home=home, precision=precision)
    if adapter.provenance["checkpoint_sha256"] != spec["official_checkpoint_sha256"]:
        raise ValueError("Official initializer differs from the fine-tuned model")
    torch.nn.Module.load_state_dict(adapter.model, state, strict=True)
    adapter.provenance.update(id=key, label=_label(spec), variant="fine-tuned",
                              checkpoint=str(checkpoint), checkpoint_sha256=spec["weights_sha256"],
                              epoch=spec["epoch"], fold=spec["fold"], role=spec["role"],
                              protocol_binding=spec["protocol_binding"],
                              source_checkpoint_sha256=spec["source_checkpoint_sha256"])
    return adapter
