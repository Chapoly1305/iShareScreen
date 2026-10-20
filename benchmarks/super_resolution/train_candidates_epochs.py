"""Fine-tune one lightweight candidate/fold with the frozen true-epoch protocol."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import sys
import time
import traceback

from benchmarks.super_resolution import candidate_training, epochs, train, training_state
from benchmarks.super_resolution.folds import partition, validate
from benchmarks.super_resolution.training_state import atomic_json, atomic_weights
from benchmarks.super_resolution.training_data import validate_pairs
from isharescreen.enhancement.models import sha256


def read(path): return json.loads(path.read_text())


def make_plan(args):
    folds, data = validate(read(args.folds)), read(args.pairs)
    if data["folds_sha256"] != sha256(args.folds) or data["manifest_sha256"] != folds["manifest_sha256"]:
        raise ValueError("Frozen data, fold and source identities differ")
    validate_pairs(data, folds)
    training, validation = epochs.split_indices(data, args.fold)
    _, held = partition(folds, args.fold)
    if {data["pairs"][i]["capture"] for i in validation} != {row["id"] for row in held}:
        raise ValueError("Validation must retain every held-out capture")
    return (dict(model=args.model, fold=args.fold, training_frames=len(training), validation_frames=len(validation),
                 epochs=args.epochs, updates_per_epoch=epochs.budget(len(training), args.batch, 1),
                 total_updates=epochs.budget(len(training), args.batch, args.epochs),
                 initialization="official x2 checkpoint or documented deterministic x4-to-x2 transfer",
                 sampling="frame-uniform shuffled without replacement; fresh aligned crop per visit",
                 validation="fixed clean anchors in held-out content groups; development selection only"),
            folds, data, training, validation)


def run(args, pause_requested=lambda: False):
    import numpy as np
    import torch
    from isharescreen.enhancement import candidates

    schedule, folds, data, training, validation = make_plan(args)
    selected = train.selection_indices(data["pairs"], validation, args.selection_per_clip)
    folder, key, seed = args.output, args.model, folds["seed"] + args.fold
    spec = candidates.verify_assets(key, args.home)
    torch.set_num_threads(args.threads); torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
    rng = np.random.default_rng(seed)
    model, initialization = candidate_training.load(key, args.home)
    device = torch.device(args.device)
    model.to(device); model.train().requires_grad_(True)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    perceptual, metric_provenance = train.perceptual_network(str(device))
    if args.pixel_metrics == "cuda" and device.type != "cuda": raise ValueError("CUDA metrics require CUDA")
    policy = dict(schema="sr_candidate_true_epoch_protocol_v1", schedule=schedule, seed=seed,
        pairs_sha256=sha256(args.pairs), folds_sha256=sha256(args.folds), sources=candidate_training.source_hashes(key),
        candidate_spec_sha256=sha256(Path(candidates.__file__).parent / key / "model.json"),
        official_checkpoint_sha256=spec["files"]["weights.pth"]["sha256"], initialization=initialization,
        batch=args.batch, patch=args.patch, lr=args.lr, loss="L1 normalized RGB", optimizer="Adam default betas/epsilon; constant LR",
        precision="fp32", train_tf32=args.train_tf32, validation_tf32=False, pixel_metrics=args.pixel_metrics,
        metric_provenance=metric_provenance, selection_every_epochs=args.validate_every, full_every_epochs=args.full_every,
        snapshot_every_epochs=args.validate_every, checkpoint_every_updates=args.checkpoint_every,
        selection_per_clip=args.selection_per_clip, selected_indices=selected,
        selection="highest fixed-subset equal-clip RGB PSNR, including epoch zero", torch=str(torch.__version__),
        cuda=torch.version.cuda, early_stopping=False, drop_last=False)
    binding = training_state.fingerprint(policy)
    if folder.exists():
        if not args.resume or read(folder / "protocol.json") != policy: raise ValueError("Existing output requires explicit resume with identical protocol")
        if (folder / "FAILED.json").exists(): raise ValueError("Failed experiment requires investigation; no automatic retry")
        if (folder / "COMPLETED.json").exists(): return 0
    else:
        if args.resume: raise FileNotFoundError("Cannot resume missing output")
        folder.mkdir(parents=True, exist_ok=False); atomic_json(folder / "protocol.json", policy)
        atomic_json(folder / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    metadata = dict(step=0, epoch=0, cursor=0, best_step=0, best_epoch=0, best_psnr=None, phase="baseline", attempt=0)
    best_state = training_state.cpu_state(model)
    if args.resume: metadata, best_state = training_state.load(folder / "last.pt", model, optimizer, rng, binding)
    metadata["attempt"] += 1
    expected = metadata["epoch"] * schedule["updates_per_epoch"] + (metadata["cursor"] + args.batch - 1)//args.batch
    if metadata["step"] != expected or not 0 <= metadata["cursor"] <= len(training) or (metadata["cursor"] != len(training) and metadata["cursor"] % args.batch):
        raise ValueError("Checkpoint epoch/cursor/update counters are inconsistent")
    def weights(state, epoch, step): return dict(state_dict=state, model=key, fold=args.fold, epoch=epoch, step=step, protocol_binding=binding, initialization=initialization)
    def save(): training_state.save(folder / "last.pt", model, optimizer, rng, binding, metadata, best_state)
    def status(phase): atomic_json(folder / "status.json", dict(model=key, fold=args.fold, phase=phase, step=metadata["step"], epoch=metadata["epoch"], cursor=metadata["cursor"], budget_epochs=args.epochs, budget_updates=schedule["total_updates"], best_epoch=metadata["best_epoch"], best_psnr=metadata["best_psnr"], pid=os.getpid(), updated=time.time(), attempt=metadata["attempt"]))
    def pause(): save(); status("paused-signal"); return 75
    def export_best(): atomic_weights(folder / "best.pt", weights(best_state, metadata["best_epoch"], metadata["best_step"]))
    def evaluate(label, indices, eval_key=key):
        status(label); torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
        original = train.predict
        train.predict = candidate_training.predict
        try: result = train.evaluate(model, eval_key, store, indices, str(device), perceptual, args.pixel_metrics)
        finally: train.predict = original
        atomic_json(folder / f"{label}.json", result); return result
    save(); status("verifying-data")
    store = train.PairStore(args.pairs.parent, data["pairs"], data.get("packed")); (folder / "snapshots").mkdir(exist_ok=True)
    with (folder / "attempts.jsonl").open("a") as attempts:
        attempts.write(json.dumps(dict(time=time.time(), pid=os.getpid(), hostname=socket.gethostname(), python=sys.version, gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else str(device), start_step=metadata["step"], attempt=metadata["attempt"], cross_device_bitwise_equivalence=False)) + "\n")
    with (folder / "progress.jsonl").open("a") as progress:
        def record(event, **fields):
            progress.write(json.dumps(dict(event=event, step=metadata["step"], epoch=metadata["epoch"], cursor=metadata["cursor"], attempt=metadata["attempt"], time=time.time(), **fields), allow_nan=False) + "\n"); progress.flush()
        record("start" if not args.resume else "resume")
        if pause_requested(): return pause()
        if metadata["phase"] == "baseline":
            if not (folder / "bicubic.json").exists(): evaluate("bicubic", validation, "bicubic")
            initial = evaluate("pretrained", validation); wanted = {(data["pairs"][i]["capture"], data["pairs"][i]["frame"]) for i in selected}
            scores = train.summarize_samples([row for row in initial["samples"] if (row["capture"], row["frame"]) in wanted])
            metadata.update(best_psnr=scores["psnr"], phase="training"); atomic_json(folder / "selection-pretrained.json", scores)
            atomic_weights(folder / "snapshots/epoch-0000.pt", weights(best_state, 0, 0)); export_best(); record("baseline", selection_validation=scores, full_validation=initial["summary"]); save()
        while metadata["epoch"] < args.epochs:
            if pause_requested(): return pause()
            order = epochs.epoch_order(training, seed, metadata["epoch"])
            while metadata["cursor"] < len(order):
                if pause_requested(): return pause()
                torch.backends.cudnn.allow_tf32 = args.train_tf32; torch.backends.cuda.matmul.allow_tf32 = args.train_tf32
                low, reference, served = epochs.batch_at(store, order, metadata["cursor"], args.batch, rng, args.patch, str(device))
                model.train(); optimizer.zero_grad(set_to_none=True); output = candidate_training.predict(model, key, low)
                if output.shape != reference.shape: raise ValueError("Training output geometry differs from paired reference")
                loss = torch.nn.functional.l1_loss(output, reference)
                if not torch.isfinite(loss): raise FloatingPointError("Nonfinite training loss")
                loss.backward(); gradients = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
                if not gradients or not torch.stack(gradients).all().item(): raise FloatingPointError("Nonfinite or missing gradient")
                optimizer.step()
                if not torch.stack([torch.isfinite(p).all() for p in model.parameters()]).all().item(): raise FloatingPointError("Nonfinite updated parameter")
                metadata["step"] += 1; metadata["cursor"] += served; record("update", loss=float(loss.detach()), batch_frames=served)
                if metadata["step"] % args.checkpoint_every == 0: save(); status("training")
            done = metadata["epoch"] + 1
            if done % args.validate_every == 0 or done == args.epochs:
                atomic_weights(folder / "snapshots" / f"epoch-{done:04d}.pt", weights(training_state.cpu_state(model), done, metadata["step"]))
                scores = evaluate(f"selection-epoch-{done:04d}", selected)["summary"]
                if scores["psnr"] > metadata["best_psnr"]: metadata.update(best_epoch=done, best_step=metadata["step"], best_psnr=scores["psnr"]); best_state=training_state.cpu_state(model); export_best()
                record("selection", completed_epoch=done, selection_validation=scores)
            if done % args.full_every == 0 or done == args.epochs: record("full", completed_epoch=done, full_validation=evaluate(f"full-epoch-{done:04d}", validation)["summary"])
            metadata.update(epoch=done, cursor=0); save(); status("training"); print(f"TRAIN {key} fold={args.fold} epoch={done}/{args.epochs} step={metadata['step']}/{schedule['total_updates']} best_epoch={metadata['best_epoch']}", flush=True)
        save(); torch.nn.Module.load_state_dict(model, best_state, strict=True); export_best(); scores=evaluate("best-validation", validation)["summary"]
        atomic_json(folder / "COMPLETED.json", dict(model=key, fold=args.fold, epochs=metadata["epoch"], steps=metadata["step"], best_epoch=metadata["best_epoch"], best_step=metadata["best_step"], early_stopped=False, checkpoint_sha256=sha256(folder / "best.pt"), protocol_binding=binding, validation=scores, full_validation_samples=len(validation), selection_samples=len(selected), scope="development CV selection, not an independent final test")); record("completed", full_validation=scores)
    status("completed"); return 0


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("pairs", "folds", "output"): p.add_argument(f"--{name}", type=Path, required=True)
    from isharescreen.enhancement.models import project_home
    p.add_argument("--home", type=Path, default=project_home()); p.add_argument("--model", choices=candidate_training.IDS, required=True); p.add_argument("--fold", type=int, required=True)
    for name, default in (("epochs",1500),("batch",32),("patch",64),("validate-every",10),("full-every",50),("checkpoint-every",100),("selection-per-clip",3),("threads",4)): p.add_argument(f"--{name}",type=int,default=default)
    p.add_argument("--lr",type=float,default=1e-5); p.add_argument("--device",default="cuda"); p.add_argument("--pixel-metrics",choices=("cpu","cuda"),default="cuda"); p.add_argument("--train-tf32",action="store_true"); p.add_argument("--resume",action="store_true"); p.add_argument("--execute",action="store_true")
    return p


def main():
    args=parser().parse_args()
    if min(args.epochs,args.batch,args.patch,args.validate_every,args.full_every,args.checkpoint_every,args.selection_per_clip,args.threads,args.lr)<=0: raise SystemExit("Positive budgets, dimensions, intervals and learning rate required")
    if args.full_every % args.validate_every: raise SystemExit("Full validation cadence must be a selection-cadence multiple")
    print(json.dumps(make_plan(args)[0],indent=2),flush=True)
    if not args.execute: return
    paused=[False]
    def request_pause(signum, frame): paused[0]=True
    signal.signal(signal.SIGUSR1,request_pause); signal.signal(signal.SIGTERM,request_pause)
    try: result=run(args,lambda:paused[0])
    except Exception:
        if args.output.is_dir() and (args.output/"status.json").is_file() and read(args.output/"status.json")["pid"]==os.getpid(): atomic_json(args.output/"FAILED.json",dict(time=time.time(),traceback=traceback.format_exc()))
        raise
    raise SystemExit(result)


if __name__ == "__main__": main()
