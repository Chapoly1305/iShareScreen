"""Optional Core ML discovery, package identity, RGB contract and worker wiring."""
import hashlib
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from isharescreen.enhancement import coreml, variants


@pytest.fixture
def installed(tmp_path, monkeypatch):
    package = coreml.bundle_path(tmp_path) / "model.mlpackage"
    files = {"Manifest.json": b"manifest", "Data/model.mlmodel": b"graph", "Data/weights.bin": b"weights"}
    for name, data in files.items():
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    monkeypatch.setattr(coreml, "PACKAGE_SHA256", {k: hashlib.sha256(v).hexdigest() for k, v in files.items()})
    # Keep provenance's weight path consistent with the production package.
    weight = "Data/com.apple.CoreML/weights/weight.bin"
    (package / weight).parent.mkdir(parents=True)
    (package / "Data/weights.bin").rename(package / weight)
    coreml.PACKAGE_SHA256[weight] = coreml.PACKAGE_SHA256.pop("Data/weights.bin")
    (package.parent / "LICENSE.md").write_text("test license")
    monkeypatch.setattr(coreml.sys, "platform", "darwin")
    monkeypatch.setattr(coreml.platform, "mac_ver", lambda: ("15.0", (), "arm64"))
    find_spec = coreml.importlib.util.find_spec
    monkeypatch.setattr(coreml.importlib.util, "find_spec", lambda name: object() if name in ("coremltools", "cv2") else find_spec(name))
    monkeypatch.setenv("ISS_SR_HOME", str(tmp_path))
    return package


@pytest.mark.parametrize("kind", ["graph", "weights", "extra", "missing", "symlink"])
def test_package_verification_rejects_changed_assets(installed, tmp_path, kind):
    coreml.verify_package(installed)
    graph = installed / "Data/model.mlmodel"
    if kind == "graph":
        graph.write_bytes(b"modified graph")
    elif kind == "weights":
        (installed / "Data/com.apple.CoreML/weights/weight.bin").write_bytes(b"modified weights")
    elif kind == "extra":
        (installed / "extra").write_bytes(b"extra")
    elif kind == "missing":
        graph.unlink()
    else:
        external = tmp_path / "external"
        external.write_bytes(graph.read_bytes())
        graph.unlink()
        graph.symlink_to(external)
    with pytest.raises(ValueError, match="hash mismatch|file set|symlinks"):
        coreml.verify_package(installed)


def test_install_is_atomic_idempotent_and_does_not_overwrite(installed, tmp_path):
    from benchmarks.super_resolution.install_coreml import install
    destination = tmp_path / "other-home"
    target = install(installed, destination, installed.parent / "LICENSE.md")
    assert (target / "LICENSE.md").read_text() == "test license"
    assert install(installed, destination, installed.parent / "LICENSE.md") == target
    (target / "model.mlpackage/Manifest.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        install(installed, destination, installed.parent / "LICENSE.md")
    assert (target / "model.mlpackage/Manifest.json").read_bytes() == b"changed"


def test_discovery_explains_platform_dependency_and_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(coreml.sys, "platform", "linux")
    assert "macOS" in coreml.availability(tmp_path)["reason"]
    monkeypatch.setattr(coreml.sys, "platform", "darwin")
    monkeypatch.setattr(coreml.platform, "mac_ver", lambda: ("12.7", (), "arm64"))
    assert "macOS 13" in coreml.availability(tmp_path)["reason"]
    monkeypatch.setattr(coreml.platform, "mac_ver", lambda: ("15.0", (), "arm64"))
    monkeypatch.setattr(coreml.importlib.util, "find_spec", lambda name: None)
    assert "dependencies" in coreml.availability(tmp_path)["reason"]
    monkeypatch.setattr(coreml.importlib.util, "find_spec", lambda name: object())
    assert "not installed" in coreml.availability(tmp_path)["reason"]


@pytest.fixture
def prediction(installed, monkeypatch):
    pytest.importorskip("cv2")
    class FakeModel:
        def __init__(self, path, compute_units):
            assert path == str(installed) and compute_units == "ALL"
            self.inputs = []
            self.output = np.zeros((1, 3, 1080, 1920), np.float32)
            self.output[:, 1] = 0.5
            self.output[:, 2] = 1.1

        def predict(self, inputs):
            self.inputs.append(inputs["image"])
            return {"output": self.output}

    monkeypatch.setitem(sys.modules, "coremltools", SimpleNamespace(
        models=SimpleNamespace(MLModel=FakeModel), ComputeUnit=SimpleNamespace(ALL="ALL")))
    return variants.load("quicksrnet")


def test_rgb_layout_range_quantization_and_timing(prediction):
    rgb = np.empty((540, 960, 3), np.uint8)
    rgb[:] = [0, 127, 255]
    # Non-contiguous RGB input is valid; preprocessing owns a contiguous tensor.
    out, timing = prediction(rgb[:, ::-1])
    source = prediction.model.inputs[-1]
    assert source.shape == (1, 3, 540, 960) and source.dtype == np.float32
    assert source.flags.c_contiguous
    np.testing.assert_allclose(source[0, :, 0, 0], [0, 127/255, 1])
    assert out.shape == (1080, 1920, 3) and out.dtype == np.uint8 and out.flags.c_contiguous
    np.testing.assert_array_equal(out[0, 0], [0, 128, 255])
    assert all(value >= 0 for value in timing.values())
    assert timing["total_ms"] == pytest.approx(sum(timing[k] for k in ("prepare_ms", "inference_ms", "download_ms")))
    assert prediction.provenance["precision"] == "fp16" and prediction.device == "coreml-all"


@pytest.mark.parametrize("shape,dtype", [((1080, 1920, 3), np.uint8), ((540, 960, 4), np.uint8), ((540, 960, 3), np.float32)])
def test_invalid_input_is_not_silently_resized(prediction, shape, dtype):
    with pytest.raises(ValueError, match="960×540"):
        prediction(np.zeros(shape, dtype))
    assert not prediction.model.inputs


@pytest.mark.parametrize("kind", ["nan", "infinity", "shape", "dtype", "not-array"])
def test_invalid_prediction_is_rejected(prediction, kind):
    if kind == "nan":
        prediction.model.output[0, 0, 0, 0] = np.nan
    elif kind == "infinity":
        prediction.model.output[0, 0, 0, 0] = np.inf
    elif kind == "shape":
        prediction.model.output = np.zeros((1, 3, 540, 960), np.float32)
    elif kind == "dtype":
        prediction.model.output = prediction.model.output.astype(np.float64)
    else:
        prediction.model.output = None
    with pytest.raises(ValueError, match="invalid pixels"):
        prediction(np.zeros((540, 960, 3), np.uint8))


@pytest.mark.parametrize("strided", [False, True])
def test_postprocessing_is_exact_at_rounding_boundaries_without_mutating_prediction(prediction, strided):
    pytest.importorskip("cv2")
    rng = np.random.default_rng(17)
    output = rng.uniform(-0.1, 1.1, size=(1, 3, 1080, 1920)).astype(np.float32)
    boundaries = (np.arange(255, dtype=np.float32) + .5) / np.float32(255)
    output.ravel()[:255] = boundaries
    output.ravel()[255:510] = np.nextafter(boundaries, np.float32(-np.inf))
    output.ravel()[510:765] = np.nextafter(boundaries, np.float32(np.inf))
    if strided:
        output = output[..., ::-1]
    prediction.model.output = output
    before = output.copy()
    expected = np.ascontiguousarray(np.clip(np.rint(output[0].transpose(1, 2, 0)*255), 0, 255).astype(np.uint8))
    actual, _ = prediction(np.zeros((540, 960, 3), np.uint8))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(output, before)
    assert actual.flags.c_contiguous and not np.shares_memory(actual, output)


def test_repeated_predictions_reuse_only_scratch_and_never_overwrite_queued_frames(prediction):
    source = np.zeros((540, 960, 3), np.uint8)
    first, _ = prediction(source)
    saved = first.copy()
    float_scratch, byte_scratch = prediction._output_float, prediction._output_u8
    prediction.model.output.fill(0)
    second, _ = prediction(source)
    assert prediction._output_float is float_scratch and prediction._output_u8 is byte_scratch
    assert not np.shares_memory(first, second)
    assert not np.shares_memory(first, float_scratch) and not np.shares_memory(first, byte_scratch)
    np.testing.assert_array_equal(first, saved)
    assert not second.any()
    second[:] = 255
    np.testing.assert_array_equal(first, saved)


def test_package_failure_precedes_model_construction(installed, monkeypatch):
    monkeypatch.setitem(sys.modules, "coremltools", None)
    (installed / "Manifest.json").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        variants.load("quicksrnet")


def test_worker_selects_processes_and_bypasses_coreml(prediction):
    from isharescreen.enhancement.live import EnhancementWorker
    worker = EnhancementWorker()
    try:
        assert next(x for x in worker.snapshot()["options"] if x["id"] == "quicksrnet")["available"]
        worker.select("quicksrnet")
        worker.submit(np.zeros((540, 960, 3), np.uint8))
        deadline = time.monotonic() + 5
        result = None
        while result is None and time.monotonic() < deadline:
            result = worker.poll()
            time.sleep(0.005)
        assert result is not None and result.model == "quicksrnet"
        assert worker.snapshot()["provenance"]["backend"] == "coreml"
        worker.select("off")
        assert worker.poll() is None and worker.snapshot()["state"] == "bypass"
    finally:
        worker.close()
