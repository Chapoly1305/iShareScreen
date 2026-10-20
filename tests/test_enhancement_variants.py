"""Live variant identity, strict loading and inference-only installation."""
import json
import time
from types import SimpleNamespace

import numpy as np
import pytest

from isharescreen.enhancement import models, variants


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("cv2")
    from benchmarks.super_resolution import export_live, training_state
    home, source = tmp_path / "assets", tmp_path / "edsr-fold-0"
    (home / "vendor" / "EDSR-PyTorch").mkdir(parents=True)
    (home / "checkpoints").mkdir()
    source.mkdir()
    official = home / "checkpoints" / models.CHECKPOINTS["edsr"]
    official.write_bytes(b"official initializer")
    lock = dict(sha256=models.sha256(official), revision="a" * 40,
                directory="EDSR-PyTorch", checkpoint=official.name)

    class Adapter:
        def __init__(self, key, device="auto", home=None, precision="fp32"):
            self.device, self.key = "cpu", key
            self.model = torch.nn.Conv2d(3, 3, 1).eval().requires_grad_(False)
            self.provenance = dict(checkpoint_sha256=lock["sha256"], device="cpu", precision=precision)

        def __call__(self, image):
            return image.repeat(2, axis=0).repeat(2, axis=1), {"total_ms": 1}

    monkeypatch.setattr(models, "Upscaler", Adapter)
    monkeypatch.setattr(export_live, "verify_source", lambda *args: lock["revision"])
    original_read = export_live.read
    monkeypatch.setattr(export_live, "read", lambda p: {"edsr": lock} if p.name == "models.lock.json" else original_read(p))
    monkeypatch.setattr(export_live.train_epochs, "recorded_sources_match", lambda s: s == {"approved": True})
    policy = dict(schema="sr_true_epoch_protocol_v1", sources={"approved": True}, folds_sha256="b" * 64,
                  schedule=dict(job=dict(model="edsr", fold=0), epochs=1500, total_updates=3000),
                  official_checkpoint_sha256=lock["sha256"], official_source_revision=lock["revision"])
    binding = training_state.fingerprint(policy)
    state = Adapter("edsr").model.state_dict()
    state["weight"].fill_(0.25)
    state["bias"].zero_()
    best = dict(state_dict={k: v + 0.5 for k, v in state.items()}, model="edsr", fold=0,
                epoch=1480, step=2960, protocol_binding=binding)
    current = dict(state_dict=state, binding=binding,
                   metadata=dict(epoch=1500, step=3000, cursor=0), optimizer={"not": "exported"})
    torch.save(current, source / "last.pt")
    torch.save(best, source / "best.pt")
    (source / "protocol.json").write_text(json.dumps(policy))
    done = dict(model="edsr", fold=0, epochs=1500, steps=3000, best_epoch=1480, best_step=2960,
                early_stopped=False, protocol_binding=binding, checkpoint_sha256=models.sha256(source / "best.pt"))
    (source / "COMPLETED.json").write_text(json.dumps(done))
    audit = tmp_path / "audit.json"

    def refresh_audit():
        audit.write_text(json.dumps(dict(complete_cv=True, folds_sha256=policy["folds_sha256"],
            evidence={source.name: {p.name: models.sha256(p) for p in source.iterdir()}})))

    refresh_audit()
    monkeypatch.setenv("ISS_SR_HOME", str(home))
    return SimpleNamespace(export=export_live.export, home=home, source=source, audit=audit,
                           current=current, best=best, refresh_audit=refresh_audit, torch=torch)


@pytest.mark.parametrize("role,epoch", [("current", 1500), ("best", 1480)])
def test_install_and_load_exact_weights_without_changing_official_or_training_files(campaign, role, epoch):
    c = campaign
    before = {p: models.sha256(p) for p in c.source.iterdir()}
    official = c.home / "checkpoints" / models.CHECKPOINTS["edsr"]
    original_hash = models.sha256(official)
    folder = c.export(c.source, c.audit, c.home, role)
    spec = variants.read_bundle(folder)
    assert spec["epoch"] == epoch and spec["role"] == role
    exported = c.torch.load(folder / "weights.pt", weights_only=True)
    expected = getattr(c, role)["state_dict"]
    assert set(exported) == set(expected)  # No Adam, RNG or training metadata.
    assert all(c.torch.equal(v, exported[k]) for k, v in expected.items())
    assert c.export(c.source, c.audit, c.home, role) == folder  # Idempotent install.
    option = next(x for x in variants.availability() if x["id"] == f"ft-{folder.name}")
    assert option["available"] and f"epoch {epoch}, fold 0" in option["label"]
    assert "Pretrained" in next(x for x in variants.availability() if x["id"] == "edsr")["label"]
    adapter = variants.load(option["id"], device="cpu")
    assert all(c.torch.equal(v, adapter.model.state_dict()[k]) for k, v in expected.items())
    assert adapter.key == "edsr"  # Preserve architecture-specific preprocessing.
    assert adapter.provenance["checkpoint_sha256"] == spec["weights_sha256"]
    assert adapter.provenance["variant"] == "fine-tuned"
    assert before == {p: models.sha256(p) for p in c.source.iterdir()}
    assert models.sha256(official) == original_hash


def test_tampered_weights_fail_instead_of_falling_back_to_pretrained(campaign):
    c = campaign
    folder = c.export(c.source, c.audit, c.home)
    (folder / "weights.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        variants.load(f"ft-{folder.name}")
    with pytest.raises(ValueError, match="different contents"):
        c.export(c.source, c.audit, c.home)


def test_repeat_install_compares_tensors_not_just_manifest_claims(campaign):
    c = campaign
    folder = c.export(c.source, c.audit, c.home)
    c.torch.save({k: v + 1 for k, v in c.current["state_dict"].items()}, folder / "weights.pt")
    spec = json.loads((folder / "manifest.json").read_text())
    spec["weights_sha256"] = models.sha256(folder / "weights.pt")
    (folder / "manifest.json").write_text(json.dumps(spec))
    with pytest.raises(ValueError, match="Installed tensors differ"):
        c.export(c.source, c.audit, c.home)


@pytest.mark.parametrize("kind", ["shape", "missing", "nonfinite"])
def test_hash_valid_but_invalid_state_is_rejected(campaign, kind):
    c = campaign
    folder = c.export(c.source, c.audit, c.home)
    state = dict(c.current["state_dict"])
    if kind == "shape":
        state["weight"] = c.torch.zeros(1)
    elif kind == "missing":
        del state["bias"]
    else:
        state["weight"].fill_(float("nan"))
    c.torch.save(state, folder / "weights.pt")
    spec = json.loads((folder / "manifest.json").read_text())
    spec["weights_sha256"] = models.sha256(folder / "weights.pt")
    (folder / "manifest.json").write_text(json.dumps(spec))
    with pytest.raises((ValueError, RuntimeError)):
        variants.load(f"ft-{folder.name}")


def test_broken_bundle_does_not_break_original_or_official_choices(campaign):
    c = campaign
    broken = c.home / "fine-tuned" / "broken"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text("not json")
    (broken / "weights.pt").write_bytes(b"not weights")
    options = {x["id"]: x for x in variants.availability()}
    assert options["off"]["available"] and options["edsr"]["available"]
    assert not options["ft-broken"]["available"]
    assert "Invalid fine-tuned bundle" in options["ft-broken"]["reason"]
    with pytest.raises(ValueError, match="Unsafe"):
        variants.load("ft-../../outside")


def test_missing_official_initializer_disables_finetuned_option(campaign):
    c = campaign
    folder = c.export(c.source, c.audit, c.home)
    (c.home / "checkpoints" / models.CHECKPOINTS["edsr"]).unlink()
    option = next(x for x in variants.availability() if x["id"] == f"ft-{folder.name}")
    assert not option["available"] and "checkpoint" in option["reason"]


@pytest.mark.parametrize("change", ["audit", "cursor", "binding"])
def test_export_rejects_changed_or_incomplete_checkpoints_before_install(campaign, change):
    c = campaign
    if change == "cursor":
        c.current["metadata"]["cursor"] = 32
    else:
        c.current["binding"] = "changed"
    c.torch.save(c.current, c.source / "last.pt")
    if change != "audit":
        c.refresh_audit()
    with pytest.raises(ValueError, match="audit|Incomplete"):
        c.export(c.source, c.audit, c.home)
    assert not (c.home / "fine-tuned").exists()


def test_live_worker_switches_finetuned_pretrained_and_original(campaign):
    from isharescreen.enhancement.live import EnhancementWorker
    c = campaign
    folder = c.export(c.source, c.audit, c.home)
    worker = EnhancementWorker(device="cpu")
    try:
        for key in (f"ft-{folder.name}", "edsr", f"ft-{folder.name}"):
            processed = worker.snapshot()["processed"]
            worker.select(key)
            assert worker.snapshot()["provenance"] == {}
            worker.submit(np.zeros((8, 8, 3), np.uint8))
            deadline = time.monotonic() + 4
            while worker.snapshot()["processed"] == processed and time.monotonic() < deadline:
                time.sleep(0.005)
            result = worker.poll()
            assert result is not None and result.model == key and result.rgb.shape == (16, 16, 3)
            provenance = worker.snapshot()["provenance"]
            assert provenance["variant"] == ("pretrained" if key == "edsr" else "fine-tuned")
            worker.snapshot()["provenance"].clear()
            assert worker.snapshot()["provenance"] == provenance
        worker.select("off")
        assert worker.snapshot()["provenance"] == {} and worker.poll() is None
    finally:
        worker.close()


def test_connect_form_offers_fixed_540p_and_weight_identity():
    from isharescreen.gui import connect
    assert '<option value="960x540">' in connect._FORM
    dashboard = connect._dash_page().decode()
    assert 'id="enhancement-details"' in dashboard
    assert "provenance.checkpoint_sha256" in dashboard
