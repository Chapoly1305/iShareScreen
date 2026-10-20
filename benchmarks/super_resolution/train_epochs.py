"""Fresh official initialization, frozen folds, true epochs and explicit resume.

Plan-only by default. Each executed invocation owns exactly one model/fold output.
Training uses only the frozen-data, frame-uniform epoch protocol.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback

from benchmarks.super_resolution import epochs, train, training_state
from benchmarks.super_resolution.folds import validate
from benchmarks.super_resolution.training_state import atomic_json, atomic_weights
from benchmarks.super_resolution.training_data import validate_pairs
from isharescreen.enhancement.models import project_home, sha256


def read(path):
    return json.loads(path.read_text())


def make_plan(args):
    folds, data = validate(read(args.folds)), read(args.pairs)
    if data["folds_sha256"] != sha256(args.folds) or data["manifest_sha256"] != folds["manifest_sha256"]:
        raise ValueError("Frozen data, fold and source identities differ")
    validate_pairs(data, folds)
    jobs = train.jobs(folds, [args.model], [args.fold])
    training, validation = epochs.split_indices(data, args.fold)
    if {data["pairs"][i]["capture"] for i in validation} != set(jobs[0]["validation_captures"]):
        raise ValueError("Validation must retain every held-out capture")
    result = dict(job=jobs[0], training_frames=len(training), validation_frames=len(validation),
                  epochs=args.epochs, updates_per_epoch=epochs.budget(len(training), args.batch, 1),
                  total_updates=epochs.budget(len(training), args.batch, args.epochs),
                  initialization="locked official pretrained checkpoint; no legacy continuation",
                  sampling="frame-uniform shuffled without replacement; fresh aligned crop per visit",
                  validation="fixed clean legacy anchors in held-out content groups; development selection only")
    return result, folds, data, training, validation


def source_hashes():
    import isharescreen.enhancement.models as adapters
    root = Path(__file__).parent
    names = ("train_epochs.py", "epochs.py", "train.py", "training_state.py", "training_data.py",
             "folds.py", "setup_models.py", "models.lock.json", "metrics.py", "video_perceptual.py")
    return {**{name: sha256(root / name) for name in names}, "adapters": sha256(Path(adapters.__file__))}


def recorded_sources_match(sources):
    """Accept current sources or the original completed epoch campaign, read-only.

    The archived source-map fingerprint belongs to Git revision aaa72f8ebb5b.
    Ordinary resume requires exact sources. The extension tool may explicitly
    upgrade this recognized source map while preserving the full training state.
    """
    current = source_hashes()
    if sources == current:
        return True
    archived = "a0ba08a71834be7c483a9107f6bf1a1dc92219a53696c66d0d908aaff331bb9a"
    return (training_state.fingerprint(sources) == archived
            and all(sources.get(key) == current[key] for key in (
                "adapters", "models.lock.json", "epochs.py", "folds.py",
                "training_data.py", "setup_models.py",
            )))


def run(args, pause_requested=lambda: False):
    import numpy as np
    import torch
    from benchmarks.super_resolution.setup_models import verify_source

    schedule, folds, data, training, validation = make_plan(args)
    selected = train.selection_indices(data["pairs"], validation, args.selection_per_clip)
    folder, key, seed = args.output, args.model, folds["seed"] + args.fold
    spec = read(Path(__file__).with_name("models.lock.json"))[key]
    verify_source(args.home / "vendor" / spec["directory"], key, spec["revision"])
    official = args.home / "checkpoints" / spec["checkpoint"]
    if sha256(official) != spec["sha256"]:
        raise ValueError("Official initialization checkpoint differs from lock")
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(seed)
    adapter = train.Upscaler(key, device=args.device, home=args.home, precision="fp32")
    model = train.enable_training(adapter)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    perceptual, metric_provenance = train.perceptual_network(adapter.device)
    if args.pixel_metrics == "cuda" and not adapter.device.startswith("cuda"):
        raise ValueError("CUDA metrics require a CUDA device")
    policy = dict(schema="sr_true_epoch_protocol_v1", schedule=schedule, seed=seed,
                  pairs_sha256=sha256(args.pairs), folds_sha256=sha256(args.folds),
                  sources=source_hashes(), official_checkpoint_sha256=spec["sha256"],
                  official_source_revision=spec["revision"], batch=args.batch, patch=args.patch,
                  lr=args.lr, loss="L1 normalized RGB", optimizer="Adam default betas/epsilon; constant LR",
                  precision="fp32", train_tf32=args.train_tf32, validation_tf32=False,
                  pixel_metrics=args.pixel_metrics, metric_provenance=metric_provenance,
                  selection_every_epochs=args.validate_every, full_every_epochs=args.full_every,
                  snapshot_every_epochs=args.validate_every, checkpoint_every_updates=args.checkpoint_every,
                  selection_per_clip=args.selection_per_clip, selected_indices=selected,
                  selection="highest fixed-subset equal-clip RGB PSNR, including epoch zero",
                  torch=str(torch.__version__), cuda=torch.version.cuda,
                  early_stopping=False, drop_last=False)
    binding = training_state.fingerprint(policy)
    if folder.exists():
        if not args.resume or read(folder / "protocol.json") != policy:
            raise ValueError("Existing output requires explicit resume with identical protocol")
        if (folder / "FAILED.json").exists():
            raise ValueError("Failed experiment requires investigation; no automatic retry")
        if (folder / "COMPLETED.json").exists():
            done = read(folder / "COMPLETED.json")
            if done["checkpoint_sha256"] != sha256(folder / "best.pt") or done["protocol_binding"] != binding:
                raise ValueError("Completed output changed")
            return 0
    else:
        if args.resume:
            raise FileNotFoundError("Cannot resume missing output")
        folder.mkdir(parents=True, exist_ok=False)
        atomic_json(folder / "protocol.json", policy)
        atomic_json(folder / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    metadata = dict(step=0, epoch=0, cursor=0, best_step=0, best_epoch=0,
                    best_psnr=None, phase="baseline", attempt=0)
    best_state = training_state.cpu_state(model)
    if args.resume:
        metadata, best_state = training_state.load(folder / "last.pt", model, optimizer, rng, binding)
    metadata["attempt"] += 1
    # The cursor is the exact number of served frames within the current epoch.
    expected_step = metadata["epoch"] * schedule["updates_per_epoch"] + (metadata["cursor"] + args.batch-1)//args.batch
    if (metadata["step"] != expected_step or not 0 <= metadata["epoch"] <= args.epochs
            or not 0 <= metadata["cursor"] <= len(training)
            or (metadata["cursor"] != len(training) and metadata["cursor"] % args.batch)):
        raise ValueError("Checkpoint epoch/cursor/update counters are inconsistent")
    def save():
        training_state.save(folder / "last.pt", model, optimizer, rng, binding, metadata, best_state)
    def status(phase):
        atomic_json(folder / "status.json", dict(model=key, fold=args.fold, phase=phase,
            step=metadata["step"], epoch=metadata["epoch"], cursor=metadata["cursor"],
            budget_epochs=args.epochs, budget_updates=schedule["total_updates"],
            best_epoch=metadata["best_epoch"], best_psnr=metadata["best_psnr"],
            pid=os.getpid(), updated=time.time(), attempt=metadata["attempt"]))
    def weights(state, epoch, step):
        return dict(state_dict=state, model=key, fold=args.fold, epoch=epoch, step=step,
                    protocol_binding=binding, official_checkpoint_sha256=spec["sha256"])
    def export_best():
        atomic_weights(folder / "best.pt", weights(best_state, metadata["best_epoch"], metadata["best_step"]))
    def pause():
        save()
        status("paused-signal")
        return 75
    save()  # Recoverable even during initial data verification/baseline evaluation.
    status("verifying-data")
    store = train.PairStore(args.pairs.parent, data["pairs"], data.get("packed"))
    (folder / "snapshots").mkdir(exist_ok=True)
    with (folder / "attempts.jsonl").open("a") as handle:
        handle.write(json.dumps(dict(time=time.time(), pid=os.getpid(), hostname=socket.gethostname(),
            python=sys.version, gpu=torch.cuda.get_device_name() if adapter.device.startswith("cuda") else adapter.device,
            start_step=metadata["step"], attempt=metadata["attempt"],
            cross_device_bitwise_equivalence=False)) + "\n")
    def evaluate(label, indices, eval_key=key):
        status(label)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        result = train.evaluate(model, eval_key, store, indices, adapter.device, perceptual, args.pixel_metrics)
        atomic_json(folder / f"{label}.json", result)
        return result
    with (folder / "progress.jsonl").open("a") as progress:
        def record(event, **fields):
            progress.write(json.dumps(dict(event=event, step=metadata["step"], epoch=metadata["epoch"],
                cursor=metadata["cursor"], attempt=metadata["attempt"], time=time.time(), **fields), allow_nan=False) + "\n")
            progress.flush()
        record("start" if not args.resume else "resume")
        if pause_requested():
            return pause()
        if metadata["phase"] == "baseline":
            if not (folder / "bicubic.json").exists():
                evaluate("bicubic", validation, "bicubic")
            if pause_requested():
                return pause()
            initial = evaluate("pretrained", validation)
            wanted = {(data["pairs"][i]["capture"], data["pairs"][i]["frame"]) for i in selected}
            scores = train.summarize_samples([r for r in initial["samples"] if (r["capture"], r["frame"]) in wanted])
            metadata.update(best_psnr=scores["psnr"], phase="training")
            atomic_json(folder / "selection-pretrained.json", scores)
            atomic_weights(folder / "snapshots/epoch-0000.pt", weights(best_state, 0, 0))
            export_best()
            record("baseline", selection_validation=scores, full_validation=initial["summary"])
            save()
        while metadata["epoch"] < args.epochs:
            if pause_requested():
                return pause()
            order = epochs.epoch_order(training, seed, metadata["epoch"])
            while metadata["cursor"] < len(order):
                if pause_requested():
                    return pause()
                torch.backends.cudnn.allow_tf32 = args.train_tf32
                torch.backends.cuda.matmul.allow_tf32 = args.train_tf32
                low, reference, served = epochs.batch_at(store, order, metadata["cursor"], args.batch,
                                                         rng, args.patch, adapter.device)
                loss = train.step(model, key, optimizer, low, reference)
                metadata["step"] += 1
                metadata["cursor"] += served
                record("update", loss=loss, batch_frames=served)
                if metadata["step"] % args.checkpoint_every == 0 or pause_requested():
                    save()
                    status("training")
            completed_epoch = metadata["epoch"] + 1
            if completed_epoch % args.validate_every == 0 or completed_epoch == args.epochs:
                atomic_weights(folder / "snapshots" / f"epoch-{completed_epoch:04d}.pt",
                               weights(training_state.cpu_state(model), completed_epoch, metadata["step"]))
                scores = evaluate(f"selection-epoch-{completed_epoch:04d}", selected)["summary"]
                if scores["psnr"] > metadata["best_psnr"]:
                    metadata.update(best_epoch=completed_epoch, best_step=metadata["step"], best_psnr=scores["psnr"])
                    best_state = training_state.cpu_state(model)
                    export_best()
                record("selection", completed_epoch=completed_epoch, selection_validation=scores)
            if completed_epoch % args.full_every == 0 or completed_epoch == args.epochs:
                scores = evaluate(f"full-epoch-{completed_epoch:04d}", validation)["summary"]
                record("full", completed_epoch=completed_epoch, full_validation=scores)
            metadata.update(epoch=completed_epoch, cursor=0)
            save()
            status("training")
            print(f"TRAIN {key} fold={args.fold} epoch={completed_epoch}/{args.epochs} "
                  f"step={metadata['step']}/{schedule['total_updates']} best_epoch={metadata['best_epoch']}", flush=True)
        if pause_requested():
            return pause()
        save()  # Final CURRENT Adam/RNG state, before loading selected weights.
        torch.nn.Module.load_state_dict(model, best_state, strict=True)
        export_best()
        scores = evaluate("best-validation", validation)["summary"]
        atomic_json(folder / "COMPLETED.json", dict(model=key, fold=args.fold, epochs=metadata["epoch"],
            steps=metadata["step"], best_epoch=metadata["best_epoch"], best_step=metadata["best_step"],
            early_stopped=False, checkpoint_sha256=sha256(folder / "best.pt"), protocol_binding=binding,
            validation=scores, full_validation_samples=len(validation), selection_samples=len(selected),
            scope="development CV selection, not an independent final test"))
        record("completed", full_validation=scores)
    status("completed")
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "folds", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.add_argument("--home", type=Path, default=project_home())
    p.add_argument("--model", choices=("edsr", "imdn", "safmn"), required=True)
    p.add_argument("--fold", type=int, required=True)
    for name, default in (("epochs", 1500), ("batch", 32), ("patch", 64), ("validate-every", 10),
                          ("full-every", 50), ("checkpoint-every", 100), ("selection-per-clip", 3), ("threads", 4)):
        p.add_argument(f"--{name}", type=int, default=default)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pixel-metrics", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--train-tf32", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--execute", action="store_true", help="Explicitly launch training; otherwise print plan")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if min(args.epochs, args.batch, args.patch, args.validate_every, args.full_every,
           args.checkpoint_every, args.selection_per_clip, args.threads, args.lr) <= 0:
        p.error("Positive budgets, dimensions, intervals and learning rate required")
    if args.full_every % args.validate_every:
        p.error("Full validation cadence must be a selection-cadence multiple")
    print(json.dumps(make_plan(args)[0], indent=2), flush=True)
    if not args.execute:
        return
    paused = [False]
    def request_pause(signum, frame):
        paused[0] = True
    signal.signal(signal.SIGUSR1, request_pause)
    signal.signal(signal.SIGTERM, request_pause)
    try:
        result = run(args, lambda: paused[0])
    except Exception:
        # Never contaminate an output owned by a different protocol.
        if args.output.is_dir() and (args.output / "status.json").is_file():
            own = read(args.output / "status.json")
            if own["pid"] == os.getpid():
                atomic_json(args.output / "FAILED.json", dict(time=time.time(), traceback=traceback.format_exc()))
        raise
    raise SystemExit(result)


if __name__ == "__main__":
    main()
