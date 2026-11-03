"""Explicit epoch-boundary recovery of the frozen Hopper scratch experiment.

The original policy and initial checkpoint remain unchanged. Only storage and
execution host may change; optimizer, scheduler and NumPy crop RNG are restored.
The pinned QuickSRNet has no stochastic forward layers or torch-based sampling.
"""

import argparse
import json
import math
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from benchmarks.super_resolution.text_corpus import train as frozen
from benchmarks.super_resolution.training_state import (
    atomic_json,
    atomic_weights,
    cpu_state,
    fingerprint,
)
from isharescreen.enhancement.models import sha256


def restore(path, model, optimizer, scheduler, rng, binding, steps_per_epoch):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["protocol_binding"] != binding:
        raise ValueError("Checkpoint policy differs")
    if payload["step"] != payload["epoch"] * steps_per_epoch:
        raise ValueError("Not an epoch-boundary checkpoint")
    if payload["scheduler"]["last_epoch"] != payload["epoch"]:
        raise ValueError("Scheduler/checkpoint epoch differs")
    model.load_state_dict(payload["state_dict"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    rng.bit_generator.state = payload["numpy_rng"]
    return payload


def run(args):
    out = args.output
    policy = frozen.read(out / "protocol.json")
    binding = fingerprint(policy)
    if (out / "COMPLETED.json").exists():
        raise ValueError("Already completed")
    for name, expected in policy["source_sha256"].items():
        source = Path(name)
        if args.original_root and args.relocated_root:
            source = args.relocated_root / source.relative_to(args.original_root)
        if sha256(source) != expected:
            raise ValueError("Frozen training source changed")
    if sha256(args.packed) != policy["packed_sha256"]:
        raise ValueError("Packed manifest differs")
    if any(
        float(policy[k]) != v
        for k, v in [("batch", 32), ("patch", 96), ("epochs", 300)]
    ):
        raise ValueError("Unexpected frozen experiment")
    torch.set_num_threads(4)
    torch.manual_seed(policy["seed"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    data, arrays = frozen.open_data(args.packed)
    indices = np.array(
        [i for i, r in enumerate(data["rows"]) if r["split"] == "train"], dtype=np.int64
    )
    validation = np.array(
        [i for i, r in enumerate(data["rows"]) if r["split"] == "validation"],
        dtype=np.int64,
    )
    model, initialization = frozen.build(args.home, policy["seed"])
    if frozen.state_identity(model) != policy["initial_tensor_sha256"]:
        raise ValueError("Initial parameters differ")
    if any(
        isinstance(layer, (torch.nn.Dropout, torch.nn.RReLU))
        for layer in model.modules()
    ):
        raise ValueError("Recovery needs missing Torch RNG state for stochastic layers")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=policy["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=policy["epochs"], eta_min=policy["lr"] * 0.01
    )
    rng = np.random.default_rng(policy["seed"])
    payload = restore(
        out / "last.pt",
        model,
        optimizer,
        scheduler,
        rng,
        binding,
        math.ceil(len(indices) / policy["batch"]),
    )
    records = [
        json.loads(line)
        for line in (out / "progress.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if not records or records[-1]["epoch"] != payload["epoch"]:
        raise ValueError(
            "Progress/checkpoint boundary differs; snapshot must be consistent"
        )
    baseline = frozen.read(out / "initial-validation.json")
    best = baseline["loss"]
    best_epoch = 0
    for row in records:
        if row["validation"]["loss"] < best:
            best = row["validation"]["loss"]
            best_epoch = row["epoch"]
    best_payload = torch.load(out / "best.pt", map_location="cpu", weights_only=True)
    if (
        best_payload["epoch"] != best_epoch
        or best_payload["protocol_binding"] != binding
    ):
        raise ValueError("Best checkpoint/history differs")
    recovery = out / "recovery"
    recovery.mkdir(exist_ok=False)
    atomic_json(
        recovery / "host.json",
        dict(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            gpu=torch.cuda.get_device_name(),
            started=time.time(),
            resumed_epoch=payload["epoch"],
            checkpoint_sha256=sha256(out / "last.pt"),
            optimizer_restored=True,
            scheduler_restored=True,
            numpy_rng_restored=True,
            protocol_binding=binding,
            packed_manifest_sha256=sha256(args.packed),
            runner_sha256=sha256(Path(__file__)),
            storage="verified identical packed bytes on host-local storage",
            torch_rng_note="Original QuickSRNet has no stochastic forward layers; crops use restored NumPy PCG64 state.",
        ),
    )
    (recovery / "runner.py").write_bytes(Path(__file__).read_bytes())
    step = payload["step"]
    started = time.time()
    previous_elapsed = records[-1]["elapsed_s"]
    epochs = policy["epochs"]
    patch = policy["patch"]
    batch = policy["batch"]

    def status(epoch, **extra):
        atomic_json(
            out / "status.json",
            dict(
                phase="training",
                epoch=epoch,
                step=step,
                total_epochs=epochs,
                best_epoch=best_epoch,
                pid=os.getpid(),
                hostname=socket.gethostname(),
                resumed=True,
                updated=time.time(),
                **extra,
            ),
        )

    status(payload["epoch"])
    with (out / "progress.jsonl").open("a") as log:
        for epoch in range(payload["epoch"] + 1, epochs + 1):
            order = np.random.default_rng(policy["seed"] + epoch).permutation(indices)
            model.train()
            losses = []
            for cursor in range(0, len(order), batch):
                low, reference, mask = [
                    v.to(device)
                    for v in frozen.crop_batch(
                        arrays, order[cursor : cursor + batch], rng, patch
                    )
                ]
                optimizer.zero_grad(set_to_none=True)
                prediction = model(low)
                if prediction.shape != reference.shape:
                    raise ValueError("Wrong reconstruction shape")
                loss, _ = frozen.objective(
                    prediction,
                    reference,
                    mask,
                    policy["text_weight"],
                    policy["edge_weight"],
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite loss")
                loss.backward()
                if not all(
                    torch.isfinite(p.grad).all().item()
                    for p in model.parameters()
                    if p.grad is not None
                ):
                    raise FloatingPointError("Nonfinite gradient")
                optimizer.step()
                step += 1
                losses.append(float(loss.detach()))
                if step % 100 == 0:
                    status(epoch, cursor=cursor, loss=losses[-1])
            result = frozen.validate(
                model,
                arrays,
                validation,
                device,
                patch,
                batch,
                policy["validation_seed"],
                policy["validation_text_weight"],
                policy["validation_edge_weight"],
            )
            payload = dict(
                state_dict=cpu_state(model),
                epoch=epoch,
                step=step,
                protocol_binding=binding,
                initialization=initialization,
            )
            if result["loss"] < best:
                best = result["loss"]
                best_epoch = epoch
                atomic_weights(out / "best.pt", payload)
            if epoch % 10 == 0 or epoch == epochs:
                atomic_weights(out / f"epoch-{epoch:04d}.pt", payload)
            scheduler.step()
            atomic_weights(
                out / "last.pt",
                dict(
                    payload,
                    optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(),
                    numpy_rng=rng.bit_generator.state,
                ),
            )
            record = dict(
                epoch=epoch,
                step=step,
                training_loss=float(np.mean(losses)),
                validation=result,
                lr=optimizer.param_groups[0]["lr"],
                best_epoch=best_epoch,
                elapsed_s=previous_elapsed + time.time() - started,
                recovery_elapsed_s=time.time() - started,
            )
            log.write(json.dumps(record) + "\n")
            log.flush()
            status(epoch, validation_loss=result["loss"])
            print(json.dumps(record), flush=True)
    atomic_json(
        out / "COMPLETED.json",
        dict(
            epochs=epochs,
            steps=step,
            best_epoch=best_epoch,
            best_validation_loss=best,
            initial_validation_loss=baseline["loss"],
            initial_tensor_sha256=policy["initial_tensor_sha256"],
            last_tensor_sha256=frozen.state_identity(model),
            best_sha256=sha256(out / "best.pt"),
            last_sha256=sha256(out / "last.pt"),
            protocol_binding=binding,
            finished=time.time(),
            recovery_host_sha256=sha256(recovery / "host.json"),
            scope="Fixed-budget scratch experiment completed after explicit epoch-boundary storage recovery.",
        ),
    )
    atomic_json(
        out / "status.json",
        dict(
            phase="completed",
            epoch=epochs,
            step=step,
            total_epochs=epochs,
            best_epoch=best_epoch,
            pid=os.getpid(),
            updated=time.time(),
            resumed=True,
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--home", type=Path, required=True)
    p.add_argument("--original-root", type=Path)
    p.add_argument("--relocated-root", type=Path)
    run(p.parse_args())
