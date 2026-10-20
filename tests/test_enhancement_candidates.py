"""Candidate metadata, strict official assets, native scale and backend routing."""
import copy
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from isharescreen.enhancement import candidates, coreml, variants


def test_each_candidate_has_isolated_pinned_metadata_and_native_scale():
    for key in candidates.IDS:
        spec = candidates.specification(key)
        assert spec["id"] == key and len(spec["revision"]) == 40
        assert all(len(asset["sha256"]) == 64 for asset in spec["files"].values())
        assert (Path(candidates.__file__).parent / key / "loader.py").is_file()
    assert candidates.specification("casr")["scale"] == 4
    assert candidates.specification("spanv2")["scale"] == 4
    with pytest.raises(ValueError):
        candidates.specification("../../outside")


def test_native_x4_is_never_offered_as_a_live_x2_model(tmp_path):
    options = {row["id"]: row for row in candidates.availability(tmp_path)}
    for key in ("casr", "spanv2"):
        assert not options[key]["available"] and "×4" in options[key]["reason"]
        with pytest.raises(ValueError, match="offline-only"):
            variants.load(key)


@pytest.fixture
def fake_assets(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")

    class Network(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 3, 1)

        def forward(self, x):
            return self.conv(x).repeat_interleave(2, 2).repeat_interleave(2, 3)

    root = candidates.asset_home("plainusr", tmp_path)
    (root / "source").mkdir(parents=True)
    (root / "source/architecture.py").write_text("official source fixture")
    state = Network().state_dict()
    state["conv.weight"].zero_()
    state["conv.bias"].fill_(0.5)
    torch.save(state, root/"weights.pth")
    spec = dict(id="plainusr", label="PlainUSR-U", scale=2, revision="a"*40, license_id="MIT", files={
        name: dict(url=f"https://example.invalid/{name}", sha256=hashlib.sha256((root/name).read_bytes()).hexdigest())
        for name in ("source/architecture.py", "weights.pth")})
    monkeypatch.setattr(candidates, "specification", lambda key: spec)
    original_import = candidates.importlib.import_module
    monkeypatch.setattr(candidates.importlib, "import_module",
        lambda name: SimpleNamespace(build=lambda source: Network()) if name == f"{candidates.__name__}.plainusr.loader" else original_import(name))
    return SimpleNamespace(root=root, home=tmp_path, spec=spec, state=state, torch=torch)


def test_strict_assets_and_cpu_adapter(fake_assets):
    c = fake_assets
    adapter = candidates.TorchUpscaler("plainusr", device="cpu", home=c.home)
    output, timing = adapter(np.zeros((8, 12, 3), np.uint8))
    assert output.shape == (16, 24, 3) and (output == 128).all()
    assert adapter.provenance["scale"] == 2 and timing["total_ms"] >= 0
    (c.root / "source/architecture.py").write_text("changed source")
    with pytest.raises(ValueError, match="changed"):
        candidates.load_network("plainusr", c.home)


@pytest.mark.parametrize("change", ["missing", "shape", "nonfinite"])
def test_hash_valid_but_invalid_candidate_state_is_rejected(fake_assets, change):
    c = fake_assets
    state = dict(c.state)
    if change == "missing":
        state.pop("conv.bias")
    elif change == "shape":
        state["conv.weight"] = c.torch.zeros(1)
    else:
        state["conv.weight"].fill_(float("nan"))
    c.torch.save(state, c.root/"weights.pth")
    c.spec["files"]["weights.pth"]["sha256"] = hashlib.sha256((c.root/"weights.pth").read_bytes()).hexdigest()
    with pytest.raises((ValueError, RuntimeError)):
        candidates.load_network("plainusr", c.home)


def test_unlicensed_download_requires_an_explicit_private_evaluation_choice(tmp_path):
    from benchmarks.super_resolution.setup_candidates import install
    with pytest.raises(ValueError, match="allow-unlicensed"):
        install("casr", tmp_path)
    assert not candidates.asset_home("casr", tmp_path).exists()


def test_candidate_installer_is_verified_idempotent_and_preserves_changed_assets(fake_assets, monkeypatch):
    from io import BytesIO
    from benchmarks.super_resolution import setup_candidates
    c = fake_assets
    assets = {asset["url"]: (c.root/name).read_bytes() for name, asset in c.spec["files"].items()}
    monkeypatch.setattr(setup_candidates.urllib.request, "urlopen", lambda url, timeout: BytesIO(assets[url]))
    destination = c.home/"new-home"
    installed = setup_candidates.install("plainusr", destination)
    assert installed == setup_candidates.install("plainusr", destination)
    candidates.verify_assets("plainusr", destination)
    (installed/"source/architecture.py").write_bytes(b"local edit")
    with pytest.raises(ValueError):
        setup_candidates.install("plainusr", destination)
    assert (installed/"source/architecture.py").read_bytes() == b"local edit"


def test_candidate_installer_does_not_publish_partial_downloads(fake_assets, monkeypatch):
    from io import BytesIO
    from benchmarks.super_resolution import setup_candidates
    c = fake_assets
    monkeypatch.setattr(setup_candidates.urllib.request, "urlopen", lambda url, timeout: BytesIO(b"bad download"))
    destination = c.home/"bad-home"
    with pytest.raises(ValueError):
        setup_candidates.install("plainusr", destination)
    assert not candidates.asset_home("plainusr", destination).exists()


def test_resident_timing_contract_and_invalid_output():
    torch = pytest.importorskip("torch")
    from benchmarks.super_resolution.benchmark_candidates import resident_forward
    adapter = SimpleNamespace(device="cpu", dtype=torch.float32, precision="fp32", sync=lambda: None,
                              model=torch.nn.Identity())
    result = resident_forward(adapter, np.zeros((8,8,3), np.uint8), 1, 2)
    assert len(result["samples_ms"]) == 2 and result["timer"] == "synchronized-host-wall"
    adapter.model = lambda value: value*float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        resident_forward(adapter, np.zeros((8,8,3), np.uint8), 1, 2)


@pytest.fixture
def coreml_manifest(tmp_path):
    root = candidates.asset_home("plainusr", tmp_path)/"coreml"
    root.mkdir(parents=True)
    spec = candidates.specification("plainusr")
    data = dict(schema="isharescreen_candidate_coreml_v1", model="plainusr", scale=2,
        official_checkpoint_sha256=spec["files"]["weights.pth"]["sha256"], source_revision=spec["revision"],
        compute_units="ALL", precision="fp32", input=[1,3,540,960], output=[1,3,1080,1920], parameters=13628,
        validation=[dict(image_sha256=character*64, max_rgb_delta=1) for character in "abc"],
        package_sha256={"Data/com.apple.CoreML/weights/weight.bin": "e"*64})
    (root/"manifest.json").write_text(json.dumps(data))
    return tmp_path, root, data


@pytest.mark.parametrize("change", ["weights", "scale", "validation", "duplicate", "path", "hash", "image_hash", "parameters"])
def test_coreml_bundle_rejects_identity_or_validation_changes(coreml_manifest, change):
    home, root, data = coreml_manifest
    assert coreml.read_candidate_bundle("plainusr", home)[1]["precision"] == "fp32"
    if change == "weights":
        data["official_checkpoint_sha256"] = "a"*64
    elif change == "scale":
        data["scale"] = 4
    elif change == "validation":
        data["validation"][0]["max_rgb_delta"] = 6
    elif change == "duplicate":
        data["validation"] *= 2
    elif change == "path":
        data["package_sha256"]["../outside"] = "b"*64
    elif change == "hash":
        data["package_sha256"]["Data/com.apple.CoreML/weights/weight.bin"] = "z"*64
    elif change == "image_hash":
        data["validation"][0]["image_sha256"] = "z"*64
    else:
        data["parameters"] = -1
    (root/"manifest.json").write_text(json.dumps(data))
    with pytest.raises(ValueError):
        coreml.read_candidate_bundle("plainusr", home)


@pytest.mark.parametrize("key", ["quicksrnet", "casr", "spanv2"])
def test_converter_rejects_models_without_a_generic_live_conversion(key, tmp_path):
    from benchmarks.super_resolution.convert_candidate import convert
    with pytest.raises(ValueError, match="install_coreml"):
        convert(key, [], tmp_path)


@pytest.mark.skipif(os.environ.get("ISS_SR_CANDIDATE_SMOKE") != "1", reason="Set ISS_SR_CANDIDATE_SMOKE=1 for installed official sources")
@pytest.mark.parametrize("key", candidates.IDS)
def test_installed_official_networks_and_deployment_parity(key):
    torch = pytest.importorskip("torch")
    root = candidates.asset_home(key)
    if not root.exists():
        pytest.skip(f"Install {key} assets")
    torch.set_num_threads(1)
    model, spec = candidates.load_network(key)
    with torch.inference_mode():
        output = model(torch.rand(1, 3, 18, 24))
    assert output.shape == (1, 3, 18*spec["scale"], 24*spec["scale"]) and torch.isfinite(output).all()
    if key == "rt4ksr":
        from isharescreen.enhancement.candidates.rt4ksr.loader import build, deploy
        original = build(root/"source").eval()
        state = torch.load(root/"weights.pth", weights_only=True, map_location="cpu")["state_dict"]
        original.load_state_dict({name.removeprefix("module."): value for name, value in state.items()}, strict=True)
        deployed = deploy(copy.deepcopy(original)).eval()
        with torch.inference_mode():
            image = torch.rand(1,3,18,24)
            torch.testing.assert_close(original(image), deployed(image), atol=1e-5, rtol=1e-5)
