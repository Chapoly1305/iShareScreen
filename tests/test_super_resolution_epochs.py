"""Epoch coverage, clean holdouts, fresh initialization and exact CPU resume."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.super_resolution import epochs


def test_epoch_order_is_complete_reproducible_and_model_independent():
    pytest.importorskip("numpy")
    indices = list(range(37))
    a = epochs.epoch_order(indices, 7, 0)
    assert sorted(a) == indices and len(set(a)) == len(indices)
    assert a == epochs.epoch_order(indices, 7, 0)
    assert a != epochs.epoch_order(indices, 7, 1)
    assert a != epochs.epoch_order(indices, 8, 0)
    with pytest.raises(ValueError):
        epochs.epoch_order([1, 1], 0, 0)


def test_epoch_budgets_keep_partial_batches():
    assert epochs.budget(3316, 32, 500) == 52000
    assert epochs.budget(3252, 32, 500) == 51000
    assert epochs.budget(3352, 32, 500) == 52500
    assert epochs.budget(3259, 32, 500) == 51000
    assert epochs.budget(3441, 32, 500) == 54000
    with pytest.raises(ValueError):
        epochs.budget(0, 32, 500)


def test_clean_validation_excludes_all_new_heldout_frames_from_training():
    data = dict(schema="sr_frozen_curated_pairs_v1", training_ready=True, pairs=[
        dict(fold=f, baseline=b, validation_eligible=b) for f in range(2) for b in (False, True)])
    training, validation = epochs.split_indices(data, 0)
    assert training == [2, 3] and validation == [1]
    data["pairs"][0]["validation_eligible"] = True
    with pytest.raises(ValueError, match="clean baseline"):
        epochs.split_indices(data, 0)


def test_aligned_crops_and_partial_batch_cover_each_frame_once():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    visited = []
    class Store:
        def load(self, i):
            visited.append(i)
            low = (np.arange(10*12*3).reshape(10, 12, 3) + i).astype(np.uint8)
            return low, low.repeat(2, 0).repeat(2, 1)
    order = epochs.epoch_order(list(range(5)), 2, 0)
    rng = np.random.default_rng(2)
    sizes = []
    for cursor in range(0, 5, 2):
        low, high, count = epochs.batch_at(Store(), order, cursor, 2, rng, 4, "cpu")
        assert (high[:, :, ::2, ::2] == low).all()
        assert low.shape == (count, 3, 4, 4)
        sizes.append(count)
    assert sizes == [2, 2, 1] and visited == order


@pytest.fixture
def tiny_campaign(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    Image = pytest.importorskip("PIL.Image")
    from benchmarks.super_resolution import train_epochs as runner
    from benchmarks.super_resolution.folds import make_plan
    from isharescreen.enhancement.models import sha256
    plan = make_plan(dict(clips=[dict(id=f"clip-{i}", frames=120, fps="60", content="code") for i in range(5)]))
    plan["manifest_sha256"] = "source-video-hash"
    folds = tmp_path / "folds.json"
    folds.write_text(json.dumps(plan))
    rows = []
    for i, clip in enumerate(plan["clips"]):
        for frame in (0, 60):
            low = np.random.default_rng(i*100+frame).integers(0, 256, (6, 8, 3), dtype=np.uint8)
            row = dict(capture=clip["id"], group=clip["group"], fold=clip["fold"], frame=frame,
                       baseline=frame == 0, validation_eligible=frame == 0)
            for role, rgb in (("lr", low), ("hr", low.repeat(2, 0).repeat(2, 1))):
                path = tmp_path / f"{i}-{frame}-{role}.png"
                Image.fromarray(rgb).save(path)
                row.update({role: path.name, f"{role}_sha256": sha256(path),
                            f"{role}_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest()})
            rows.append(row)
    pairs = tmp_path / "manifest.json"
    pairs.write_text(json.dumps(dict(schema="sr_frozen_curated_pairs_v1", training_ready=True, pairs=rows,
                                    folds_sha256=sha256(folds), manifest_sha256=plan["manifest_sha256"])))
    home = tmp_path / "assets"
    (home / "checkpoints").mkdir(parents=True)
    lock = json.loads(Path(runner.__file__).with_name("models.lock.json").read_text())["imdn"]
    checkpoint = home / "checkpoints" / lock["checkpoint"]
    checkpoint.write_bytes(b"official-toy-checkpoint")
    original_sha = runner.sha256
    monkeypatch.setattr(runner, "sha256", lambda p: lock["sha256"] if p == checkpoint else original_sha(p))
    monkeypatch.setattr("benchmarks.super_resolution.setup_models.verify_source", lambda *a: lock["revision"])
    starts = []
    class Adapter:
        def __init__(self, key, device, home, precision):
            self.key, self.device = key, device
            self.model = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 1), torch.nn.Upsample(scale_factor=2, mode="nearest"))
            starts.append(copy.deepcopy(self.model.state_dict()))
    monkeypatch.setattr(runner.train, "Upscaler", Adapter)
    monkeypatch.setattr(runner.train, "perceptual_network", lambda device: (None, {"mock": True}))
    def evaluate(model, key, store, indices, device, perceptual, pixel_metrics):
        # No pretrained downloads in unit tests. Exercise real training and Adam.
        value = sum(float(p.detach().sum()) for p in model.parameters())
        samples = [dict(capture=store.rows[i]["capture"], frame=store.rows[i]["frame"],
                        psnr=30+value/100, ssim=.9, lpips=.1) for i in indices]
        return dict(summary=runner.train.summarize_samples(samples), samples=samples)
    monkeypatch.setattr(runner.train, "evaluate", evaluate)
    args = runner.parser().parse_args(["--pairs", str(pairs), "--folds", str(folds), "--output", str(tmp_path / "run"),
        "--home", str(home), "--model", "imdn", "--fold", "0", "--epochs", "3", "--batch", "3", "--patch", "3",
        "--validate-every", "1", "--full-every", "2", "--device", "cpu", "--pixel-metrics", "cpu", "--threads", "1"])
    return runner, args, starts


def test_epoch_cli_prints_plan_without_starting_training(tiny_campaign, monkeypatch, capsys):
    import sys
    runner, args, _ = tiny_campaign
    def unexpected_training(*values):
        raise AssertionError("Planning must not launch training")
    monkeypatch.setattr(runner, "run", unexpected_training)
    monkeypatch.setattr(sys, "argv", [
        "train_epochs", "--pairs", str(args.pairs), "--folds", str(args.folds),
        "--output", str(args.output), "--model", args.model, "--fold", str(args.fold),
    ])
    runner.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["epochs"] == 1500
    assert plan["total_updates"] == plan["updates_per_epoch"] * 1500
    assert not args.output.exists()


def test_runner_resume_mid_epoch_matches_uninterrupted_adam_and_weights(tiny_campaign):
    import torch
    runner, args, starts = tiny_campaign
    assert runner.run(args) == 0
    full = torch.load(args.output / "last.pt", weights_only=True)
    assert full["metadata"]["step"] == 9 and full["metadata"]["epoch"] == 3
    fresh = args.output.with_name("resumed")
    args.output = fresh
    def pause_after_two_updates():
        log = fresh / "progress.jsonl"
        return log.exists() and sum('"event": "update"' in line for line in log.read_text().splitlines()) >= 2
    assert runner.run(args, pause_after_two_updates) == 75
    paused = torch.load(fresh / "last.pt", weights_only=True)
    assert paused["metadata"]["step"] == 2 and paused["metadata"]["cursor"] == 6
    assert starts[0].keys() == starts[1].keys()
    assert all(torch.equal(starts[0][k], starts[1][k]) for k in starts[0])
    args.resume = True
    assert runner.run(args) == 0
    resumed = torch.load(fresh / "last.pt", weights_only=True)
    for k in full["state_dict"]:
        assert torch.equal(full["state_dict"][k], resumed["state_dict"][k])
    for k, values in full["optimizer"]["state"].items():
        for name, value in values.items():
            assert torch.equal(value, resumed["optimizer"]["state"][k][name])
    assert full["rng"] == resumed["rng"]
    assert torch.equal(full["torch_rng"], resumed["torch_rng"])
    assert json.loads((fresh / "COMPLETED.json").read_text())["epochs"] == 3


def test_epoch_zero_snapshot_contains_fresh_initial_weights(tiny_campaign):
    import torch
    runner, args, starts = tiny_campaign
    assert runner.run(args) == 0
    snapshot = torch.load(args.output / "snapshots/epoch-0000.pt", weights_only=True)
    assert snapshot["step"] == snapshot["epoch"] == 0
    assert all(torch.equal(starts[0][k], snapshot["state_dict"][k]) for k in starts[0])
    final = torch.load(args.output / "last.pt", weights_only=True)
    assert any(not torch.equal(snapshot["state_dict"][k], final["state_dict"][k]) for k in starts[0])


def test_runner_rejects_overwrite_and_changed_resume_protocol(tiny_campaign):
    runner, args, _ = tiny_campaign
    assert runner.run(args, lambda: True) == 75
    before = (args.output / "last.pt").read_bytes()
    with pytest.raises(ValueError, match="explicit resume"):
        runner.run(args)
    args.resume = True
    args.epochs += 1
    with pytest.raises(ValueError, match="identical protocol"):
        runner.run(args)
    assert (args.output / "last.pt").read_bytes() == before


def test_unreviewed_manifests_cannot_enter_epoch_runner():
    with pytest.raises(ValueError, match="frozen curated"):
        epochs.split_indices({"pairs": []}, 0)


def freeze_inputs(args):
    from isharescreen.enhancement.models import sha256
    data = json.loads(args.pairs.read_text())
    root = args.pairs.parent
    def write(name, value):
        path = root / name
        path.write_text(json.dumps(value))
        return path
    baseline = write("baseline.json", {"pairs": [r for r in data["pairs"] if r["baseline"]]})
    candidates = write("candidates.json", dict(candidates=data["pairs"], folds_sha256=sha256(args.folds),
        baseline_sha256=sha256(baseline), manifest_sha256=data["manifest_sha256"]))
    proposal = write("proposal.json", dict(decisions=[dict(index=i, selected=True) for i in range(len(data["pairs"]))],
        candidates_sha256=sha256(candidates), folds_sha256=sha256(args.folds)))
    pool = write("pool.json", dict(candidates_sha256=sha256(candidates), proposal_sha256=sha256(proposal),
        eligible_candidate_indices=list(range(len(data["pairs"]))), eligible_pair_count=len(data["pairs"])))
    qc = write("qc.json", dict(status="complete_with_known_legacy_baseline_warnings",
        assistant_visual_review={"complete": True}, evidence_sha256=dict(proposal=sha256(proposal),
            training_pool_index=sha256(pool), original_baseline=sha256(baseline), original_folds=sha256(args.folds))))
    return dict(candidates=candidates, proposal=proposal, pool=pool, qc=qc, folds=args.folds, baseline=baseline)


def test_freeze_copies_pixels_and_preserves_clean_validation_membership(tiny_campaign):
    from benchmarks.super_resolution.freeze_training import freeze
    from isharescreen.enhancement.models import sha256
    _, args, _ = tiny_campaign
    inputs = freeze_inputs(args)
    output = args.output.with_name("frozen")
    result = freeze(**inputs, output=output, workers=2)
    assert len(result["pairs"]) == 10
    assert sum(r["validation_eligible"] for r in result["pairs"]) == 5
    row = result["pairs"][0]
    origin = inputs["candidates"].parent / row["lr"]
    origin.write_bytes(b"changed after independent freeze")
    assert sha256(output / row["lr"]) == row["lr_sha256"]


def test_freeze_rejects_changed_signoff_before_creating_output(tiny_campaign):
    from benchmarks.super_resolution.freeze_training import freeze
    _, args, _ = tiny_campaign
    inputs = freeze_inputs(args)
    inputs["proposal"].write_text(inputs["proposal"].read_text() + " ")
    output = args.output.with_name("frozen")
    with pytest.raises(ValueError, match="identity changed"):
        freeze(**inputs, output=output)
    assert not output.exists()


def test_transport_restores_exact_bytes_and_refuses_changed_existing_arrays(tmp_path, monkeypatch):
    import shutil
    import subprocess
    from benchmarks.super_resolution import restore_packed
    decoder = shutil.which("zstd")
    if decoder is None and Path("/opt/homebrew/bin/zstd").is_file():
        decoder = "/opt/homebrew/bin/zstd"
    if decoder is None:
        pytest.skip("Optional zstd executable is not installed")
    monkeypatch.setattr(restore_packed.shutil, "which", lambda name: decoder)
    packed = {}
    for role in ("lr", "hr"):
        raw = (role.encode() + b"frozen pixels") * 100
        compressed = subprocess.run([decoder, "-q", "--stdout"], input=raw, capture_output=True, check=True).stdout
        (tmp_path / f"{role}.npy.zst").write_bytes(compressed)
        packed[role] = dict(path=f"{role}.npy", sha256=hashlib.sha256(raw).hexdigest())
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"packed": packed}))
    receipt = restore_packed.restore(manifest, tmp_path)
    assert receipt["verified"] == {r: s["sha256"] for r, s in packed.items()}
    (tmp_path / "lr.npy").write_bytes(b"changed")
    with pytest.raises(ValueError, match="refusing overwrite"):
        restore_packed.restore(manifest, tmp_path)
