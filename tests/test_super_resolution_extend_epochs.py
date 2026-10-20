import copy
import json

import pytest

from benchmarks.super_resolution import extend_epochs
from test_super_resolution_epochs import tiny_campaign
from test_super_resolution_sources import ARCHIVED_SOURCES


def hashes(folder):
    from isharescreen.enhancement.models import sha256
    return {str(p.relative_to(folder)): sha256(p) for p in folder.rglob("*") if p.is_file()}


def archived_parent(folder):
    """Rebind a synthetic fixture to the recognized historical source identity."""
    import torch
    from benchmarks.super_resolution import training_state
    from isharescreen.enhancement.models import sha256
    policy = json.loads((folder / "protocol.json").read_text())
    policy["sources"] = dict(ARCHIVED_SOURCES)
    binding = training_state.fingerprint(policy)
    (folder / "protocol.json").write_text(json.dumps(policy))
    for name, field in (("last.pt", "binding"), ("best.pt", "protocol_binding"),
                        ("snapshots/epoch-0000.pt", "protocol_binding")):
        value = torch.load(folder / name, weights_only=True)
        value[field] = binding
        torch.save(value, folder / name)
    done = json.loads((folder / "COMPLETED.json").read_text())
    done.update(protocol_binding=binding, checkpoint_sha256=sha256(folder / "best.pt"))
    (folder / "COMPLETED.json").write_text(json.dumps(done))


def test_explicit_source_upgrade_preserves_state_and_continues_identically(tiny_campaign):
    import torch
    runner, args, _ = tiny_campaign
    args.epochs, args.full_every = 4, 2
    assert runner.run(args) == 0
    expected = torch.load(args.output / "last.pt", weights_only=True)
    args.output, args.epochs = args.output.with_name("archived-parent"), 2
    assert runner.run(args) == 0
    archived_parent(args.output)
    before = hashes(args.output)
    child = args.output.with_name("upgraded")
    with pytest.raises(ValueError, match="upgrade-sources"):
        extend_epochs.prepare(args.output, child, 4, args.pairs, args.folds, args.home)
    receipt = extend_epochs.prepare(args.output, child, 4, args.pairs, args.folds,
                                   args.home, upgrade_sources=True)
    assert receipt["source_upgrade"]
    assert receipt["allowed_policy_changes"] == ["schedule.epochs", "schedule.total_updates", "sources"]
    state = torch.load(child / "last.pt", weights_only=True)
    assert state["metadata"]["step"] == 6
    assert hashes(args.output) == before
    args.output, args.epochs, args.resume = child, 4, True
    assert runner.run(args) == 0
    actual = torch.load(child / "last.pt", weights_only=True)
    for field in ("state_dict", "optimizer", "rng", "torch_rng", "cuda_rng", "best_state"):
        assert extend_epochs.same(expected[field], actual[field]), field
    assert actual["metadata"]["step"] == 12


def test_source_upgrade_rejects_unrecognized_source_before_writing(tiny_campaign):
    runner, args, _ = tiny_campaign
    args.epochs = 2
    assert runner.run(args) == 0
    policy = json.loads((args.output / "protocol.json").read_text())
    policy["sources"]["train.py"] = "unknown"
    (args.output / "protocol.json").write_text(json.dumps(policy))
    child = args.output.with_name("invalid-upgrade")
    with pytest.raises(ValueError, match="Unrecognized training sources"):
        extend_epochs.prepare(args.output, child, 4, args.pairs, args.folds,
                             args.home, upgrade_sources=True)
    assert not child.exists()


def test_already_extended_parent_can_continue_again(tiny_campaign):
    runner, args, _ = tiny_campaign
    args.epochs, args.full_every = 2, 2
    assert runner.run(args) == 0
    first = args.output.with_name("first-extension")
    extend_epochs.prepare(args.output, first, 4, args.pairs, args.folds, args.home)
    args.output, args.epochs, args.resume = first, 4, True
    assert runner.run(args) == 0
    before = hashes(first)
    second = first.with_name("second-extension")
    extend_epochs.prepare(first, second, 6, args.pairs, args.folds, args.home)
    assert extend_epochs.verify_prepared(second)["old_epochs"] == 4
    assert hashes(first) == before
    args.output, args.epochs = second, 6
    assert runner.run(args) == 0


def test_budget_extension_matches_uninterrupted_current_adam_rng_and_best(tiny_campaign):
    import torch
    runner, args, _ = tiny_campaign
    args.epochs = 4
    args.full_every = 2
    assert runner.run(args) == 0
    full = torch.load(args.output / "last.pt", weights_only=True)
    parent = args.output.with_name("parent")
    args.output, args.epochs = parent, 2
    assert runner.run(args) == 0
    before = hashes(parent)
    child = parent.with_name("extended")
    receipt = extend_epochs.prepare(parent, child, 4, args.pairs, args.folds, args.home)
    assert receipt["old_epochs"] == receipt["added_epochs"] == 2
    assert receipt["new_steps"] == 12
    assert hashes(parent) == before
    args.output, args.epochs, args.resume = child, 4, True
    assert runner.run(args) == 0
    continued = torch.load(child / "last.pt", weights_only=True)
    for field in ("state_dict", "optimizer", "rng", "torch_rng", "cuda_rng", "best_state"):
        assert extend_epochs.same(full[field], continued[field]), field
    assert continued["metadata"]["best_epoch"] == full["metadata"]["best_epoch"]
    events = [json.loads(line) for line in (child / "progress.jsonl").read_text().splitlines()]
    assert [r["step"] for r in events if r["event"] == "update"] == list(range(1, 13))
    assert [r["completed_epoch"] for r in events if r["event"] == "selection"] == [1, 2, 3, 4]
    assert hashes(parent) == before


@pytest.mark.parametrize("mutation", ["current", "optimizer", "best", "source", "data", "failed", "ceiling"])
def test_extension_rejects_invalid_parent_before_creating_output(tiny_campaign, mutation):
    import torch
    runner, args, _ = tiny_campaign
    args.epochs = 2
    assert runner.run(args) == 0
    if mutation in ("current", "optimizer", "best"):
        path = args.output / "last.pt"
        state = torch.load(path, weights_only=True)
        if mutation == "current":
            state["metadata"]["epoch"] = 1
        elif mutation == "optimizer":
            next(iter(state["optimizer"]["state"].values()))["step"].zero_()
        else:
            next(iter(state["best_state"].values())).add_(1)
        torch.save(state, path)
    elif mutation in ("source", "data"):
        path = args.output / "protocol.json"
        policy = json.loads(path.read_text())
        if mutation == "source":
            policy["sources"]["train_epochs.py"] = "changed"
        else:
            policy["pairs_sha256"] = "changed"
        path.write_text(json.dumps(policy))
    elif mutation == "failed":
        (args.output / "FAILED.json").write_text("{}")
    child = args.output.with_name("extended")
    with pytest.raises(ValueError):
        extend_epochs.prepare(args.output, child, 2 if mutation == "ceiling" else 4,
                             args.pairs, args.folds, args.home)
    assert not child.exists()


def test_prepared_extension_preserves_earlier_best_and_detects_tampering(tiny_campaign):
    import torch
    runner, args, _ = tiny_campaign
    args.epochs = 2
    assert runner.run(args) == 0
    # Exercise a best-at-zero parent while CURRENT/Adam remain at epoch two.
    parent = args.output
    initial = torch.load(parent / "snapshots/epoch-0000.pt", weights_only=True)
    torch.save(initial, parent / "best.pt")
    state = torch.load(parent / "last.pt", weights_only=True)
    state["best_state"] = copy.deepcopy(initial["state_dict"])
    state["metadata"].update(best_epoch=0, best_step=0)
    torch.save(state, parent / "last.pt")
    from isharescreen.enhancement.models import sha256
    done = json.loads((parent / "COMPLETED.json").read_text())
    done.update(best_epoch=0, best_step=0, checkpoint_sha256=sha256(parent / "best.pt"))
    (parent / "COMPLETED.json").write_text(json.dumps(done))
    child = parent.with_name("extended")
    extend_epochs.prepare(parent, child, 4, args.pairs, args.folds, args.home)
    loaded = torch.load(child / "last.pt", weights_only=True)
    assert loaded["metadata"]["step"] == 6 and loaded["metadata"]["best_epoch"] == 0
    assert not extend_epochs.same(loaded["state_dict"], loaded["best_state"])
    loaded["rng"]["state"]["state"] += 1
    torch.save(loaded, child / "last.pt")
    with pytest.raises(ValueError, match="Adam, RNG"):
        extend_epochs.verify_prepared(child)
