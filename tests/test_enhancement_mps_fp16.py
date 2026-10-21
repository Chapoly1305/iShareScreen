"""The explicitly labeled MPS deployment choice must stay GPU-only and fixed."""
from types import SimpleNamespace

import numpy as np
import pytest

from isharescreen.enhancement import candidates,variants


def test_explicit_choice_binds_backend_precision_and_provenance(monkeypatch,tmp_path):
    monkeypatch.delenv("ISS_SR_GPU_ONLY",raising=False)
    monkeypatch.setattr(variants.sys,"platform","darwin")
    monkeypatch.setattr(variants.platform,"machine",lambda:"arm64")
    calls=[]
    def factory(key,**kwargs):
        calls.append((key,kwargs))
        return SimpleNamespace(device="mps",provenance={})
    monkeypatch.setattr(candidates,"TorchUpscaler",factory)
    adapter=variants.load(variants.MPS_FP16_ID,home=tmp_path)
    assert calls==[("quicksrnet",dict(device="mps",home=tmp_path,precision="fp16"))]
    assert adapter.provenance["id"]==variants.MPS_FP16_ID
    assert "FP16" in adapter.provenance["label"]
    assert adapter.provenance["input_shape"]==[540,960,3]


@pytest.mark.parametrize("device",["cpu","cuda"])
def test_fixed_backend_never_substitutes_a_different_device(monkeypatch,device):
    monkeypatch.delenv("ISS_SR_GPU_ONLY",raising=False)
    with pytest.raises(ValueError,match="MPS GPU"):
        variants.load(variants.MPS_FP16_ID,device=device)


@pytest.mark.parametrize("key,device,precision",[("plainusr","mps","fp16"),
    ("rt4ksr","mps","fp16"),("efrlfn","mps","fp16"),
    ("quicksrnet","cpu","fp16"),("quicksrnet","mps","amp-fp16")])
def test_unvalidated_half_precision_combinations_remain_rejected(monkeypatch,key,device,precision):
    pytest.importorskip("torch")
    monkeypatch.setattr(candidates,"load_network",lambda *args:(None,{}))
    with pytest.raises(ValueError,match="validated QuickSRNet"):
        candidates.TorchUpscaler(key,device=device,precision=precision)


def test_mps_half_uses_float16_and_rejects_unvalidated_input_sizes(monkeypatch):
    torch=pytest.importorskip("torch")
    from isharescreen.enhancement import mps_output
    monkeypatch.setattr(mps_output,"MPSRGBPacker",lambda:None)
    network=torch.nn.Conv2d(3,3,1)
    calls=[]
    monkeypatch.setattr(network,"to",lambda **kwargs:calls.append(kwargs) or network)
    spec=candidates.specification("quicksrnet")
    monkeypatch.setattr(candidates,"load_network",lambda *args:(network,spec))
    adapter=candidates.TorchUpscaler("quicksrnet",device="mps",precision="fp16")
    assert calls==[dict(device="mps",dtype=torch.float16)]
    assert adapter.provenance["precision"]=="fp16"
    with pytest.raises(ValueError,match="960×540"):
        adapter(np.zeros((8,8,3),np.uint8))


def test_fixed_choice_discovery_preserves_missing_asset_reason_and_platform(monkeypatch,tmp_path):
    monkeypatch.setattr(variants.sys,"platform","darwin")
    monkeypatch.setattr(variants.platform,"machine",lambda:"arm64")
    choices={r["id"]:r for r in variants.availability(tmp_path)}
    assert not choices[variants.MPS_FP16_ID]["available"]
    assert choices[variants.MPS_FP16_ID]["reason"]==choices["quicksrnet-torch"]["reason"]
    monkeypatch.setattr(variants.platform,"machine",lambda:"x86_64")
    option=next(r for r in variants.availability(tmp_path) if r["id"]==variants.MPS_FP16_ID)
    assert not option["available"] and "Apple Silicon" in option["reason"]
    with pytest.raises(ValueError,match="Apple Silicon"):
        variants.load(variants.MPS_FP16_ID)
