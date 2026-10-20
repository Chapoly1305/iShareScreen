"""Opt-in GPU-only inference policy; never change the frozen training adapter."""
import os


def gpu_only():
    return os.environ.get("ISS_SR_GPU_ONLY") == "1"


def disabled_reason(key):
    if not gpu_only() or key == "off":
        return ""
    if key in ("bilinear", "bicubic", "sharpen"):
        return "GPU-only mode excludes CPU interpolation baselines"
    if key == "quicksrnet" or key.endswith("-coreml"):
        return "Core ML cannot prohibit CPU execution; use the PyTorch GPU variant"
    return ""


def filter_choices(choices):
    for item in choices:
        reason = disabled_reason(item["id"])
        if reason:
            item.update(available=False, reason=reason)
    return choices


def resolve_device(key, device):
    if not gpu_only():
        return device
    reason = disabled_reason(key)
    if reason:
        raise ValueError(reason)
    if device == "cpu":
        raise ValueError("GPU-only mode prohibits CPU inference")
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps"
    if device == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise ValueError("GPU-only mode requires MPS CPU fallback disabled before starting Python")
        if not torch.backends.mps.is_available():
            raise ValueError("MPS GPU is unavailable; CPU fallback is prohibited")
    elif not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("CUDA GPU is unavailable; CPU fallback is prohibited")
    return device


def validate_adapter(adapter):
    if gpu_only():
        device = adapter.device
        if device != "mps" and not device.startswith("cuda"):
            raise ValueError("GPU-only mode rejected a non-GPU adapter")
        tensors = (*adapter.model.parameters(), *adapter.model.buffers())
        if any(value.device.type != device.split(":")[0] for value in tensors):
            raise ValueError("Model parameters or buffers are not on the selected GPU")
        adapter.provenance["gpu_only"] = True
    return adapter
