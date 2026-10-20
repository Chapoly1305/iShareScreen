"""GPU-only selection must fail closed, without importing optional ML eagerly."""
from types import SimpleNamespace

import pytest

from isharescreen.enhancement import execution, variants


@pytest.mark.parametrize("key", ["quicksrnet", "plainusr-coreml", "rt4ksr-coreml", "efrlfn-coreml", "bicubic"])
def test_gpu_only_disables_cpu_capable_choices(monkeypatch, tmp_path, key):
    monkeypatch.setenv("ISS_SR_GPU_ONLY", "1")
    option = next(x for x in variants.availability(tmp_path) if x["id"] == key)
    assert not option["available"] and "CPU" in option["reason"]
    with pytest.raises(ValueError, match="CPU"):
        variants.load(key, home=tmp_path)


def test_cpu_and_enabled_mps_fallback_are_rejected(monkeypatch):
    monkeypatch.setenv("ISS_SR_GPU_ONLY", "1")
    with pytest.raises(ValueError, match="CPU inference"):
        execution.resolve_device("plainusr", "cpu")
    pytest.importorskip("torch")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    with pytest.raises(ValueError, match="fallback disabled"):
        execution.resolve_device("plainusr", "mps")


def test_non_gpu_adapter_or_parameters_are_rejected(monkeypatch):
    monkeypatch.setenv("ISS_SR_GPU_ONLY", "1")
    with pytest.raises(ValueError, match="non-GPU"):
        execution.validate_adapter(SimpleNamespace(device="cpu"))
    torch = pytest.importorskip("torch")
    adapter = SimpleNamespace(device="mps", model=torch.nn.Conv2d(3,3,1), provenance={})
    with pytest.raises(ValueError, match="parameters"):
        execution.validate_adapter(adapter)


def test_non_strict_mode_is_unchanged(monkeypatch):
    monkeypatch.delenv("ISS_SR_GPU_ONLY", raising=False)
    assert execution.resolve_device("quicksrnet", "auto") == "auto"
