"""Install inference-only bundles from audited, completed epoch trajectories."""
import argparse
import json
from pathlib import Path
import tempfile

from benchmarks.super_resolution import train_epochs, training_state
from benchmarks.super_resolution.setup_models import verify_source
from isharescreen.enhancement import models, variants


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def export(folder: Path, audit_path: Path, home: Path, role: str = "current") -> Path:
    """Preserve training files and official weights; publish one atomic bundle."""
    import torch
    require(role in ("current", "best"), "Choose current or best weights")
    require(not (folder / "FAILED.json").exists(), "Cannot install a failed trajectory")
    policy, done, audit = (read(path) for path in
                           (folder / "protocol.json", folder / "COMPLETED.json", audit_path))
    job = policy["schedule"]["job"]
    key, fold = job["model"], job["fold"]
    require(key in variants.MODELS, "Unsupported live architecture")
    require(policy["schema"] == "sr_true_epoch_protocol_v1"
            and train_epochs.recorded_sources_match(policy["sources"]), "Unrecognized epoch training sources")
    require(audit["complete_cv"] and audit["folds_sha256"] == policy["folds_sha256"], "Audit fold identity differs")
    binding = training_state.fingerprint(policy)
    cap, total = policy["schedule"]["epochs"], policy["schedule"]["total_updates"]
    require((done["model"], done["fold"], done["epochs"], done["steps"], done["protocol_binding"])
            == (key, fold, cap, total, binding) and done["early_stopped"] is False,
            "Trajectory is not complete or its identity changed")
    filename = "last.pt" if role == "current" else "best.pt"
    source = folder / filename
    source_hash = models.sha256(source)
    evidence = audit["evidence"][f"{key}-fold-{fold}"]
    require(all(models.sha256(folder / name) == evidence[name] for name in
                ("protocol.json", "COMPLETED.json", filename)), "Checkpoint differs from completed audit")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    epoch = cap if role == "current" else done["best_epoch"]
    if role == "current":
        md = payload["metadata"]
        require(payload["binding"] == binding and md["epoch"] == cap
                and md["cursor"] == 0 and md["step"] == total, "Incomplete current checkpoint")
    else:
        require((payload["model"], payload["fold"], payload["epoch"], payload["step"], payload["protocol_binding"])
                == (key, fold, epoch, done["best_step"], binding)
                and source_hash == done["checkpoint_sha256"], "Selected checkpoint identity differs")
    state = payload["state_dict"]
    require(state and all(isinstance(v, torch.Tensor) and torch.isfinite(v).all().item() for v in state.values()),
            "Empty or nonfinite model weights")
    lock = read(Path(__file__).with_name("models.lock.json"))[key]
    require(policy["official_checkpoint_sha256"] == lock["sha256"]
            and policy["official_source_revision"] == lock["revision"], "Official training initializer changed")
    verify_source(home / "vendor" / lock["directory"], key, lock["revision"])
    require(models.sha256(home / "checkpoints" / lock["checkpoint"]) == lock["sha256"],
            "Installed official initializer changed")
    adapter = models.Upscaler(key, home=home, device="cpu")
    torch.nn.Module.load_state_dict(adapter.model, state, strict=True)
    parent = home / "fine-tuned"
    parent.mkdir(parents=True, exist_ok=True)
    target = parent / variants.bundle_name(key, fold, epoch, role)
    manifest = dict(schema=variants.SCHEMA, model=key, fold=fold, epoch=epoch, role=role,
                    source_checkpoint_sha256=source_hash, protocol_binding=binding,
                    official_checkpoint_sha256=lock["sha256"], audit_sha256=models.sha256(audit_path))
    if target.exists():
        existing = variants.read_bundle(target)
        require(all(existing.get(k) == v for k, v in manifest.items())
                and models.sha256(target / "weights.pt") == existing["weights_sha256"],
                "Live bundle already exists with different contents; choose a different model asset home")
        installed = torch.load(target / "weights.pt", map_location="cpu", weights_only=True)
        require(isinstance(installed, dict) and state.keys() == installed.keys()
                and all(torch.equal(v, installed[k]) for k, v in state.items()),
                "Installed tensors differ from the audited checkpoint")
        return target
    with tempfile.TemporaryDirectory(prefix=".install-", dir=parent) as temporary:
        stage = Path(temporary) / target.name
        stage.mkdir()
        training_state.atomic_weights(stage / "weights.pt", state)
        restored = torch.load(stage / "weights.pt", map_location="cpu", weights_only=True)
        require(state.keys() == restored.keys() and all(torch.equal(v, restored[k]) for k, v in state.items()),
                "Inference-only export changed model tensors")
        manifest["weights_sha256"] = models.sha256(stage / "weights.pt")
        training_state.atomic_json(stage / "manifest.json", manifest)
        variants.read_bundle(stage)
        stage.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--home", type=Path, default=models.project_home())
    parser.add_argument("--models", nargs="+", choices=variants.MODELS, default=list(variants.MODELS))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--role", choices=("current", "best"), default="current")
    args = parser.parse_args()
    import torch
    torch.set_num_threads(1)
    for key in args.models:
        target = export(args.runs / f"{key}-fold-{args.fold}", args.audit, args.home, args.role)
        print(f"INSTALLED {target.name}: {target}", flush=True)


if __name__ == "__main__":
    main()
