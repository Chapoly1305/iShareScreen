"""Grouped development boundaries and differentiable training, without downloads."""
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from benchmarks.super_resolution.folds import make_plan, partition, validate
from benchmarks.super_resolution.training_data import keep_unique, validate_pairs


def manifest(count=24):
    return {"clips": [dict(id=f"clip-{i:02d}", frames=600+i*60, fps="60/1",
                           content=["code", "web", "terminal"][i % 3]) for i in range(count)]}


def test_folds_are_balanced_deterministic_and_every_group_is_held_out_once():
    original = manifest()
    plan = make_plan(original)
    assert sorted(s["groups"] for s in plan["summary"]) == [4, 5, 5, 5, 5]
    assert plan == make_plan({"clips": list(reversed(original["clips"]))})
    seen = []
    for fold in range(5):
        train, val = partition(plan, fold)
        assert {c["group"] for c in train}.isdisjoint(c["group"] for c in val)
        assert len(train) + len(val) == 24
        seen += [c["id"] for c in val]
    assert sorted(seen) == sorted(c["id"] for c in original["clips"])
    assert make_plan(original, seed=42) != plan
    assert all("fold" not in c for c in original["clips"])  # Input not mutated.


def test_related_captures_stay_together_even_when_input_order_changes():
    plan = make_plan(manifest(), group_map={"clip-00": "same-doc", "clip-17": "same-doc"})
    assigned = {c["id"]: c["fold"] for c in plan["clips"]}
    assert assigned["clip-00"] == assigned["clip-17"]
    plan["clips"][0]["fold"] = (assigned["clip-17"] + 1) % 5
    with pytest.raises(ValueError, match="crosses folds"):
        validate(plan)


@pytest.mark.parametrize("change,match", [
    (lambda d: d["clips"].append(d["clips"][0]), "duplicate"),
    (lambda d: d["clips"][0].update(frames=0), "duration"),
])
def test_bad_source_inventory_is_rejected(change, match):
    data = manifest()
    change(data)
    with pytest.raises(ValueError, match=match):
        make_plan(data)


def test_invalid_group_maps_and_too_few_groups_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        make_plan(manifest(), group_map={"missing": "g"})
    with pytest.raises(ValueError, match="Empty content"):
        make_plan(manifest(), group_map={"clip-00": ""})
    with pytest.raises(ValueError, match="Fewer"):
        make_plan(manifest(4))
    with pytest.raises(ValueError, match="held-out"):
        partition(make_plan(manifest()), -1)


def pair(clip, frame=0, digest=None):
    return dict(capture=clip["id"], group=clip["group"], fold=clip["fold"], frame=frame,
                hr_rgb_sha256=digest or clip["id"])


def test_fold_blind_dedup_excludes_shared_references_from_all_groups():
    plan = make_plan(manifest(5))
    a, b = plan["clips"][:2]
    rows = [pair(a, 0, "shared"), pair(b, 0, "shared"), pair(a, 60, "unique"),
            pair(a, 120, "unique"), pair(b, 60, "another")]
    kept, dropped = keep_unique(rows)
    assert [(r["capture"], r["frame"]) for r in kept] == [(a["id"], 60), (b["id"], 60)]
    assert [r["reason"] for r in dropped] == ["shared_exact_reference", "shared_exact_reference", "repeated_reference"]


def test_pairs_cannot_move_folds_leak_references_or_drop_captures():
    plan = make_plan(manifest(5))
    data = {"pairs": [pair(c) for c in plan["clips"]]}
    validate_pairs(data, plan)
    bad = copy.deepcopy(data)
    bad["pairs"][0]["fold"] = (bad["pairs"][0]["fold"] + 1) % 5
    with pytest.raises(ValueError, match="frozen assignment"):
        validate_pairs(bad, plan)
    bad = copy.deepcopy(data)
    bad["pairs"][1]["hr_rgb_sha256"] = bad["pairs"][0]["hr_rgb_sha256"]
    with pytest.raises(ValueError, match="duplicated across"):
        validate_pairs(bad, plan)
    with pytest.raises(ValueError, match="without retained"):
        validate_pairs({"pairs": data["pairs"][:-1]}, plan)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_pairs({"pairs": data["pairs"] + data["pairs"][:1]}, plan)


def test_all_models_share_the_same_split():
    from benchmarks.super_resolution.train import jobs
    plan = make_plan(manifest())
    schedule = jobs(plan, ["edsr", "imdn", "safmn"], list(range(5)))
    assert len(schedule) == 15
    for fold in range(5):
        selected = [job for job in schedule if job["fold"] == fold]
        assert all(job["train_captures"] == selected[0]["train_captures"] for job in selected)
        assert all(job["validation_captures"] == selected[0]["validation_captures"] for job in selected)
    with pytest.raises(ValueError, match="unique eligible"):
        jobs(plan, ["imdn", "imdn"], [0])


def test_real_gradient_updates_and_nonfinite_gradient_fail_closed():
    torch = pytest.importorskip("torch")
    from benchmarks.super_resolution.train import predict, step
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 12, 1), torch.nn.PixelShuffle(2))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    low = torch.ones(1, 3, 8, 8)
    reference = torch.zeros(1, 3, 16, 16)
    before = model[0].weight.detach().clone()
    assert step(model, "imdn", optimizer, low, reference) > 0
    assert not torch.equal(before, model[0].weight)
    torch.testing.assert_close(predict(torch.nn.Identity(), "edsr", low), low)
    hook = model[0].weight.register_hook(lambda grad: grad * float("nan"))
    before = model[0].weight.detach().clone()
    with pytest.raises(FloatingPointError, match="gradient"):
        step(model, "imdn", optimizer, low, reference)
    assert torch.equal(before, model[0].weight)  # No optimizer update on bad gradients.
    hook.remove()


def test_edsr_normalization_stays_frozen():
    torch = pytest.importorskip("torch")
    from benchmarks.super_resolution.train import enable_training
    model = torch.nn.Module()
    model.sub_mean = torch.nn.Conv2d(3, 3, 1)
    model.add_mean = torch.nn.Conv2d(3, 3, 1)
    model.body = torch.nn.Conv2d(3, 3, 1)
    enable_training(SimpleNamespace(key="edsr", model=model.requires_grad_(False)))
    assert model.training and model.body.weight.requires_grad
    assert not model.sub_mean.weight.requires_grad
    assert not model.add_mean.bias.requires_grad


def test_real_video_sampling_hashes_alignment_and_nonoverwrite(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("av")
    pytest.importorskip("torch")
    pytest.importorskip("cv2")
    pytest.importorskip("skimage")
    from fractions import Fraction
    from benchmarks.super_resolution.video_benchmark import LosslessWriter
    from isharescreen.enhancement.models import sha256
    clips = []
    for capture in range(2):
        clip = dict(id=f"capture-{capture}", frames=3, fps="60/1", content="code")
        for role, size in (("reference", 32), ("low", 16)):
            path = tmp_path / f"{capture}-{role}.mkv"
            writer = LosslessWriter(path, size, size, Fraction(60))
            try:
                for index in range(3):
                    writer.write(np.full((size, size, 3), capture*50+index*10, np.uint8))
            finally:
                writer.close()
            clip[role], clip[f"{role}_sha256"] = path.name, sha256(path)
        clips.append(clip)
    source = tmp_path / "video.json"
    source.write_text(json.dumps(dict(clips=clips)))
    folds = tmp_path / "folds.json"
    command = [sys.executable, "-m", "benchmarks.super_resolution.folds", "--manifest", str(source),
               "--output", str(folds), "--folds", "2"]
    subprocess.run(command, check=True, capture_output=True)
    before = folds.read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert folds.read_bytes() == before
    output = tmp_path / "sampled"
    subprocess.run([sys.executable, "-m", "benchmarks.super_resolution.training_data",
                    "--manifest", str(source), "--folds", str(folds), "--output", str(output),
                    "--every", "2"], check=True, capture_output=True)
    data = json.loads((output / "manifest.json").read_text())
    validate_pairs(data, json.loads(folds.read_text()))
    assert len(data["pairs"]) == 4
    assert {r["frame"] for r in data["pairs"]} == {0, 2}
    for row in data["pairs"]:
        assert sha256(output / row["hr"]) == row["hr_sha256"]


def test_pair_images_cannot_escape_dataset_directory(tmp_path):
    from benchmarks.super_resolution.train import PairStore
    store = PairStore(tmp_path, [])
    with pytest.raises(ValueError, match="escapes"):
        store.path("../outside.png")


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_atomic_resume_restores_optimizer_rng_weights_and_rejects_new_identity(tmp_path, device):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA checkpoint continuation requires CUDA")
    from benchmarks.super_resolution import training_state as state
    from benchmarks.super_resolution.train import step
    torch.manual_seed(1)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 12, 1), torch.nn.PixelShuffle(2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    rng = np.random.default_rng(2)
    def update(net, optim, generator):
        low = torch.tensor(generator.random((1, 3, 8, 8)), dtype=torch.float32, device=device)
        reference = torch.rand(1, 3, 16, 16, device=device)
        return step(net, "imdn", optim, low, reference)
    update(model, optimizer, rng)
    path = tmp_path / "last.pt"
    binding = state.fingerprint(dict(data="frozen", fold=0, steps=2))
    state.save(path, model, optimizer, rng, binding, {"step": 1}, state.cpu_state(model))
    expected_loss = update(model, optimizer, rng)
    expected = state.cpu_state(model)
    restored = torch.nn.Sequential(torch.nn.Conv2d(3, 12, 1), torch.nn.PixelShuffle(2)).to(device)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=1e-3)
    restored_rng = np.random.default_rng(500)
    metadata, _ = state.load(path, restored, restored_optimizer, restored_rng, binding)
    assert metadata["step"] == 1
    assert update(restored, restored_optimizer, restored_rng) == expected_loss
    for key, tensor in restored.state_dict().items():
        assert torch.equal(tensor.cpu(), expected[key])
    assert not path.with_suffix(".pt.tmp").exists()
    with pytest.raises(ValueError, match="identity changed"):
        state.load(path, restored, restored_optimizer, restored_rng, "different-fold-or-data")


def test_packed_pixels_match_png_and_detect_tampering(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")
    from benchmarks.super_resolution.pack_training import pack
    from benchmarks.super_resolution.train import PairStore
    from isharescreen.enhancement.models import sha256
    rows = []
    for i in range(3):
        row = {}
        for role, edge in (("lr", 8), ("hr", 16)):
            rgb = np.full((edge, edge, 3), i*40, np.uint8)
            path = tmp_path / f"{i}-{role}.png"
            Image.fromarray(rgb).save(path)
            row[role], row[f"{role}_sha256"] = path.name, sha256(path)
            row[f"{role}_rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
        rows.append(row)
    source = tmp_path / "pairs.json"
    source.write_text(json.dumps(dict(pairs=rows)))
    data = pack(source, tmp_path / "packed")
    store = PairStore(tmp_path / "packed", rows, data["packed"])
    original = PairStore(tmp_path, rows)
    for i in range(3):
        for actual, expected in zip(store.load(i), original.load(i)):
            np.testing.assert_array_equal(actual, expected)
            assert not actual.flags.writeable
    with pytest.raises(FileExistsError):
        pack(source, tmp_path / "packed")
    with (tmp_path / "packed" / "hr.npy").open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError, match="Packed pixels changed"):
        PairStore(tmp_path / "packed", rows, data["packed"])


def test_fixed_selection_subset_preserves_each_capture_and_final_full_coverage():
    from benchmarks.super_resolution.train import selection_indices
    rows = [dict(capture=c, frame=i) for c in ("a", "b") for i in range(10)]
    indices = list(range(20))
    selected = selection_indices(rows, indices, 3)
    assert selected == [0, 4, 9, 10, 14, 19]
    assert indices == list(range(20))
    assert selection_indices(rows, indices, 0) == indices


@pytest.mark.parametrize("key", ["edsr", "imdn", "safmn"])
@pytest.mark.skipif(os.environ.get("ISS_SR_TRAIN_SMOKE") != "1", reason="Opt-in cached official-model backprop smoke")
def test_cached_official_models_backprop_and_strict_checkpoint_roundtrip(key, tmp_path):
    torch = pytest.importorskip("torch")
    from benchmarks.super_resolution.train import enable_training, step
    from isharescreen.enhancement.models import Upscaler, sha256
    torch.set_num_threads(2)
    torch.manual_seed(4)
    adapter = Upscaler(key, device="cpu")
    checkpoint = Path(adapter.provenance["checkpoint"])
    original_hash = sha256(checkpoint)
    model = enable_training(adapter)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-5)
    before = next(p for p in model.parameters() if p.requires_grad).detach().clone()
    value = step(model, key, optimizer, torch.rand(1, 3, 8, 8), torch.rand(1, 3, 16, 16))
    assert value > 0
    assert not torch.equal(before, next(p for p in model.parameters() if p.requires_grad))
    path = tmp_path / "fine-tuned.pt"
    torch.save(dict(state_dict=model.state_dict(), provenance={"torch": str(torch.__version__)}), path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    torch.nn.Module.load_state_dict(model, state["state_dict"], strict=True)
    assert sha256(checkpoint) == original_hash
