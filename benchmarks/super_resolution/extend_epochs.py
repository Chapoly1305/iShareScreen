"""Budget-only true-epoch continuation into a fresh, traceable output directory.

The source run is never edited. Recovery uses CURRENT weights, Adam and RNG;
historical best weights and all prefix observations remain available. Only the
epoch ceiling changes by default. An explicit source upgrade can migrate the
recognized pre-cleanup epoch campaign to the current implementation.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from benchmarks.super_resolution import train_epochs, training_state
from benchmarks.super_resolution.training_state import atomic_json, atomic_weights
from isharescreen.enhancement.models import sha256


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text())


def same(a, b):
    """Exact nested tensor/state comparison, without serializing pickle bytes."""
    import torch
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(same(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    return a == b


def extended_policy(policy, epochs, *, upgrade_sources=False):
    result = copy.deepcopy(policy)
    old = result["schedule"]
    require(isinstance(epochs, int) and not isinstance(epochs, bool) and epochs > old["epochs"],
            "New total epochs must exceed the completed parent ceiling")
    # A non-cadence parent final evaluation would add an extra observation to
    # the uninterrupted schedule. Reject rather than silently change selection.
    require(all(old["epochs"] % policy[k] == 0 for k in
                ("selection_every_epochs", "full_every_epochs")),
            "Parent ceiling must be on both validation cadences")
    old.update(epochs=epochs, total_updates=epochs*old["updates_per_epoch"])
    if upgrade_sources:
        require(train_epochs.recorded_sources_match(policy["sources"]),
                "Unrecognized training sources; source upgrade refused")
        result["sources"] = train_epochs.source_hashes()
    return result


def prepare(source, output, epochs, pairs, folds, home, *, upgrade_sources=False):
    import torch
    torch.set_num_threads(1)
    source, output = Path(source), Path(output)
    require(not output.exists(), "Continuation requires a fresh output directory")
    require(not (source / "FAILED.json").exists(), "Failed parent requires investigation")
    policy, config, done = (read(source / n) for n in ("protocol.json", "config.json", "COMPLETED.json"))
    require(policy["schema"] == "sr_true_epoch_protocol_v1", "Not a true-epoch parent")
    require(upgrade_sources or policy["sources"] == train_epochs.source_hashes(),
            "Pinned training sources changed; recognized epoch checkpoints need --upgrade-sources")
    new_policy = extended_policy(policy, epochs, upgrade_sources=upgrade_sources)
    require(policy["pairs_sha256"] == sha256(pairs) and policy["folds_sha256"] == sha256(folds),
            "Frozen data or fold identity changed")
    schedule = policy["schedule"]
    key, fold = schedule["job"]["model"], schedule["job"]["fold"]
    expected, *_ = train_epochs.make_plan(SimpleNamespace(pairs=pairs, folds=folds, model=key,
        fold=fold, epochs=schedule["epochs"], batch=policy["batch"]))
    require(schedule == expected, "Parent epoch/data schedule differs")
    binding = training_state.fingerprint(policy)
    cap, total = schedule["epochs"], schedule["total_updates"]
    require((done["model"], done["fold"], done["epochs"], done["steps"], done["protocol_binding"])
            == (key, fold, cap, total, binding) and done["early_stopped"] is False,
            "Parent must have completed its entire bound ceiling")
    require(sha256(source / "best.pt") == done["checkpoint_sha256"], "Parent selected weights changed")
    load = lambda p: torch.load(p, map_location="cpu", weights_only=True)
    current, best, initial = (load(source / n) for n in ("last.pt", "best.pt", "snapshots/epoch-0000.pt"))
    md = current["metadata"]
    require(current["binding"] == binding and (md["epoch"], md["cursor"], md["step"], md["phase"])
            == (cap, 0, total, "training"), "Parent CURRENT recovery state is incomplete")
    require((md["best_epoch"], md["best_step"]) == (done["best_epoch"], done["best_step"])
            and same(current["best_state"], best["state_dict"]), "Parent embedded best differs")
    for payload, e, s in ((best, done["best_epoch"], done["best_step"]), (initial, 0, 0)):
        require((payload["model"], payload["fold"], payload["epoch"], payload["step"], payload["protocol_binding"])
                == (key, fold, e, s, binding), "Parent exported checkpoint identity differs")
        require(payload["official_checkpoint_sha256"] == policy["official_checkpoint_sha256"],
                "Parent official checkpoint identity differs")
    require(current["optimizer"]["state"] and all(int(v["step"]) == total
            for v in current["optimizer"]["state"].values()), "Adam counters differ from CURRENT update count")
    require(current["rng"] and current["torch_rng"].numel() and "cuda_rng" in current, "Missing RNG state")
    require(all(torch.isfinite(v).all().item() for state in
            (current["state_dict"], current["best_state"], initial["state_dict"]) for v in state.values()),
            "Nonfinite model weights")
    # Derive recipe from the bound policy, not a historical machine's config.
    updated = dict(config, pairs=str(pairs), folds=str(folds), home=str(home), output=str(output),
        model=key, fold=fold, epochs=epochs, resume=True, execute=True,
        batch=policy["batch"], patch=policy["patch"], lr=policy["lr"], train_tf32=policy["train_tf32"],
        pixel_metrics=policy["pixel_metrics"], validate_every=policy["selection_every_epochs"],
        full_every=policy["full_every_epochs"], checkpoint_every=policy["checkpoint_every_updates"],
        selection_per_clip=policy["selection_per_clip"])
    new_binding = training_state.fingerprint(new_policy)
    # Archive an independent, byte-identical parent subset. The full original
    # snapshot archive remains at source. Never hardlink writable child files.
    files = [p for p in source.iterdir() if p.is_file() and
             (p.suffix in (".json", ".jsonl") or p.name in ("last.pt", "best.pt"))]
    files.append(source / "snapshots/epoch-0000.pt")
    hashes = {str(p.relative_to(source)): sha256(p) for p in files}
    output.mkdir(parents=True, exist_ok=False)
    archive = output / "parent-history"
    archive.mkdir()
    for path in files:
        relative = path.relative_to(source)
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    # Copy observations, not old completion/status/control markers. The old
    # selected-validation result is archived until this child finishes.
    observations = [p for p in files if p.name in
        ("bicubic.json", "pretrained.json", "selection-pretrained.json", "progress.jsonl", "attempts.jsonl")
        or p.name.startswith(("selection-epoch-", "full-epoch-"))]
    for path in observations:
        shutil.copy2(path, output / path.name)
    (output / "snapshots").mkdir()
    for name, payload in (("best.pt", best), ("snapshots/epoch-0000.pt", initial)):
        rebound = dict(payload, protocol_binding=new_binding)
        atomic_weights(output / name, rebound)
    atomic_weights(output / "last.pt", dict(current, binding=new_binding))
    atomic_json(output / "protocol.json", new_policy)
    atomic_json(output / "config.json", updated)
    receipt = dict(schema="sr_epoch_budget_extension_v1", parent=str(source.resolve()),
        parent_artifacts_sha256=hashes, old_binding=binding, new_binding=new_binding,
        old_epochs=cap, new_epochs=epochs, added_epochs=epochs-cap,
        old_steps=total, new_steps=new_policy["schedule"]["total_updates"],
        allowed_policy_changes=["schedule.epochs", "schedule.total_updates"],
        source_upgrade=policy["sources"] != new_policy["sources"],
        resume_from="parent CURRENT last.pt, not selected best.pt",
        exported_checkpoint_rebinding="metadata only; original payloads retained in parent-history",
        optimizer_and_rng_preserved=True, cross_device_bitwise_equivalence=False,
        generator_sha256=sha256(Path(__file__)))
    if receipt["source_upgrade"]:
        receipt["allowed_policy_changes"].append("sources")
    atomic_json(output / "EXTENSION.json", receipt)
    verify_prepared(output)
    return receipt


def verify_prepared(output):
    """Verify exact state/observation preservation before launching any updates."""
    import torch
    output = Path(output)
    receipt, policy = read(output / "EXTENSION.json"), read(output / "protocol.json")
    archive = output / "parent-history"
    for name, digest in receipt["parent_artifacts_sha256"].items():
        require(sha256(archive / name) == digest, "Archived parent artifact changed")
    old = read(archive / "protocol.json")
    upgraded = receipt.get("source_upgrade", False)
    require(policy == extended_policy(old, receipt["new_epochs"], upgrade_sources=upgraded),
            "Extension changed more than the permitted budget/source upgrade")
    allowed = ["schedule.epochs", "schedule.total_updates"] + (["sources"] if upgraded else [])
    require(receipt["allowed_policy_changes"] == allowed, "Unexpected permitted policy changes")
    require(training_state.fingerprint(old) == receipt["old_binding"]
            and training_state.fingerprint(policy) == receipt["new_binding"], "Extension binding differs")
    for name, field in (("last.pt", "binding"), ("best.pt", "protocol_binding"),
                        ("snapshots/epoch-0000.pt", "protocol_binding")):
        before = torch.load(archive / name, map_location="cpu", weights_only=True)
        after = torch.load(output / name, map_location="cpu", weights_only=True)
        require(before[field] == receipt["old_binding"] and after[field] == receipt["new_binding"],
                "Rebound checkpoint identity differs")
        before[field] = after[field]
        require(same(before, after), "Continuation changed CURRENT weights, Adam, RNG or historical best")
    for path in output.iterdir():
        if path.name in receipt["parent_artifacts_sha256"] and path.name not in (
                "last.pt", "best.pt", "protocol.json", "config.json", "EXTENSION.json"):
            require(sha256(path) == receipt["parent_artifacts_sha256"][path.name], "Imported observation changed")
    require(not any((output / n).exists() for n in ("COMPLETED.json", "FAILED.json", "status.json")),
            "Prepared continuation contains a stale execution marker")
    return receipt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "pairs", "folds", "home"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--epochs", type=int, required=True, help="Cumulative ceiling, not extra epochs")
    p.add_argument("--upgrade-sources", action="store_true",
                   help="Explicitly migrate the recognized pre-cleanup epoch sources; preserve all training state")
    print(json.dumps(prepare(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
