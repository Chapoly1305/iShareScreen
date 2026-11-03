"""Train source-bound QuickSRNet-Small x2 from scratch or official weights.

All new-corpus training views are visited once per epoch. Fresh aligned crops
are biased toward source-known text; validation uses fixed crops in disjoint
source groups. The test split is never used for updates or checkpoint selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from benchmarks.super_resolution.text_corpus.scratch import build
from benchmarks.super_resolution.training_state import (
    atomic_json,
    atomic_weights,
    cpu_state,
    fingerprint,
)
from isharescreen.enhancement.models import project_home, sha256


def read(path):
    return json.loads(path.read_text())


def initialize(home, seed, kind="scratch"):
    if kind == "scratch":
        return build(home, seed)
    if kind != "official":
        raise ValueError("Unknown initialization")
    from benchmarks.super_resolution.text_corpus.official import load

    torch.manual_seed(seed)
    model, spec = load(home)
    model.train().requires_grad_(True)
    return model, dict(
        kind="official_pretrained",
        pretrained_weights_loaded=True,
        architecture="QuickSRNetSmall x2",
        source_revision=spec["revision"],
        checkpoint_sha256=spec["files"]["weights.pth"]["sha256"],
    )


def eligible_indices(data, split, excluded):
    return np.array(
        [
            i
            for i, r in enumerate(data["rows"])
            if r["split"] == split and r["requested_bps"] not in excluded
        ],
        dtype=np.int64,
    )


def state_identity(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def open_data(path, verify=True):
    data = read(path)
    if not data["complete"] or data["schema"] != "isharescreen_text_packed_v1":
        raise ValueError("Incomplete/wrong packed data")
    arrays = {}
    for name, spec in data["files"].items():
        location = path.parent / spec["path"]
        if verify and sha256(location) != spec["sha256"]:
            raise ValueError("Packed data hash differs: " + name)
        arrays[name] = np.load(location, mmap_mode="r", allow_pickle=False)
        if list(arrays[name].shape) != spec["shape"]:
            raise ValueError("Packed data shape differs: " + name)
    if (
        arrays["lr"].dtype != np.uint8
        or arrays["hr"].dtype != np.uint8
        or arrays["boxes"].shape[1] != 5
    ):
        raise ValueError("Wrong packed pixel/label type")
    groups = {}
    aliases = data["group_aliases"]
    for row in data["rows"]:
        group = aliases.get(row["source_group"], row["source_group"])
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Source split leakage")
        groups[group] = row["split"]
    return data, arrays


def crop_batch(arrays, indices, rng, patch=96):
    lows = []
    highs = []
    masks = []
    for index in indices:
        low, high = arrays["lr"][index], arrays["hr"][index]
        start, end = arrays["offsets"][index : index + 2]
        boxes = arrays["boxes"][start:end].astype(np.int32)
        if len(boxes) and rng.random() < 0.8:
            weights = boxes[:, 4].astype(np.float64)
            weights /= weights.sum()
            anchor = boxes[rng.choice(len(boxes), p=weights)]
            center = (anchor[:2] + anchor[2:4]) / 4  # HR corners -> LR center.
            x = int(
                np.clip(
                    center[0] - patch / 2 + rng.integers(-patch // 4, patch // 4 + 1),
                    0,
                    low.shape[1] - patch,
                )
            )
            y = int(
                np.clip(
                    center[1] - patch / 2 + rng.integers(-patch // 4, patch // 4 + 1),
                    0,
                    low.shape[0] - patch,
                )
            )
        else:
            x = int(rng.integers(low.shape[1] - patch + 1))
            y = int(rng.integers(low.shape[0] - patch + 1))
        lows.append(low[y : y + patch, x : x + patch])
        highs.append(high[2 * y : 2 * (y + patch), 2 * x : 2 * (x + patch)])
        mask = np.zeros((patch * 2, patch * 2), np.float32)
        selected = boxes[
            (boxes[:, 0] < 2 * (x + patch))
            & (boxes[:, 2] > 2 * x)
            & (boxes[:, 1] < 2 * (y + patch))
            & (boxes[:, 3] > 2 * y)
        ]
        for a, b, c, d, _ in selected:
            mask[
                max(0, b - 2 * y) : min(patch * 2, d - 2 * y),
                max(0, a - 2 * x) : min(patch * 2, c - 2 * x),
            ] = 1
        masks.append(mask)
    return (
        torch.from_numpy(np.stack(lows)).permute(0, 3, 1, 2).float() / 255,
        torch.from_numpy(np.stack(highs)).permute(0, 3, 1, 2).float() / 255,
        torch.from_numpy(np.stack(masks))[:, None],
    )


def objective(prediction, reference, mask, text_weight=2.0, edge_weight=0.05):
    weight = 1 + text_weight * mask
    pixel = ((prediction - reference).abs() * weight).sum() / (weight.sum() * 3)
    dx = (prediction[..., 1:] - prediction[..., :-1]) - (
        reference[..., 1:] - reference[..., :-1]
    )
    dy = (prediction[..., 1:, :] - prediction[..., :-1, :]) - (
        reference[..., 1:, :] - reference[..., :-1, :]
    )
    edge = (dx.abs().mean() + dy.abs().mean()) / 2
    return pixel + edge_weight * edge, dict(pixel=pixel.detach(), edge=edge.detach())


def validate(
    model, arrays, indices, device, patch, batch, seed, text_weight, edge_weight
):
    # Two reproducible crops per held-out view. Each view has equal weight.
    values = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(indices), batch):
            chosen = indices[offset : offset + batch]
            crops = []
            for index in chosen:
                crops.append(
                    crop_batch(
                        arrays,
                        [index, index],
                        np.random.default_rng(seed + int(index)),
                        patch,
                    )
                )
            low, reference, mask = [
                torch.cat([c[k] for c in crops]).to(device) for k in range(3)
            ]
            output = model(low)
            if not torch.isfinite(output).all():
                raise FloatingPointError("Nonfinite validation output")
            # Per-crop scoring avoids partial-batch or coverage weighting bias.
            for i in range(len(output)):
                loss, _ = objective(
                    output[i : i + 1],
                    reference[i : i + 1],
                    mask[i : i + 1],
                    text_weight,
                    edge_weight,
                )
                values.append(float(loss))
    return dict(
        loss=float(np.mean(values)),
        views=len(indices),
        fixed_crops=len(values),
        selection="equal-view text-weighted pixel L1 plus edge proxy; OCR evaluated separately",
    )


def run(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("This initial full experiment requires an available CUDA GPU")
    if args.memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    data, arrays = open_data(args.packed)
    excluded = set(args.exclude_bps or [])
    train_indices = eligible_indices(data, "train", excluded)
    validation_indices = eligible_indices(data, "validation", excluded)
    if not len(train_indices) or not len(validation_indices):
        raise ValueError("Missing training/validation split")
    model, initialization = initialize(args.home, args.seed, args.initialization)
    initial_tensor_hash = state_identity(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    sources = {
        str(p.resolve()): sha256(p)
        for p in [
            Path(__file__),
            Path(__file__).with_name("scratch.py"),
            Path(__file__).with_name("official.py"),
        ]
    }
    corpus = args.packed.parent.parent
    policy = dict(
        schema="isharescreen_text_training_v2",
        packed_sha256=sha256(args.packed),
        initialization=initialization,
        initial_tensor_sha256=initial_tensor_hash,
        model="quicksrnet",
        scale=2,
        seed=args.seed,
        epochs=args.epochs,
        training_views=len(train_indices),
        validation_views=len(validation_indices),
        test_views=len(eligible_indices(data, "test", excluded)),
        excluded_bps=sorted(excluded),
        training_indices_sha256=hashlib.sha256(train_indices.tobytes()).hexdigest(),
        validation_indices_sha256=hashlib.sha256(
            validation_indices.tobytes()
        ).hexdigest(),
        memory_fraction=args.memory_fraction,
        batch=args.batch,
        patch=args.patch,
        lr=args.lr,
        optimizer="Adam default betas/epsilon",
        scheduler="epoch cosine decay to 1% initial LR",
        sampling="Every training view once per epoch; shuffled; 80% source-known word anchors (3x numeric/punctuation anchor weight), 20% uniform; fresh crop each visit",
        text_weight=args.text_weight,
        edge_weight=args.edge_weight,
        loss="RGB L1 weighted by 1+text_weight*source_box_mask, normalized by weights; plus edge_weight*mean horizontal/vertical derivative L1",
        precision="fp32",
        tf32=False,
        validation="Two fixed crops per validation view, equal view mean; every epoch. No test data used.",
        validation_text_weight=args.validation_text_weight,
        validation_edge_weight=args.validation_edge_weight,
        validation_seed=args.validation_seed,
        checkpoint_selection="Minimum common fixed validation text-weighted loss; subsequent OCR comparisons must declare this proxy selection",
        source_sha256=sources,
        ocr_protocol_sha256=sha256(corpus / "evaluation-protocol/protocol.json"),
        ocr_source_compatibility_sha256=sha256(
            corpus / "evaluation-protocol/source-compatibility.json"
        ),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
    )
    binding = fingerprint(policy)
    out = args.output
    if out.exists():
        raise ValueError("Training requires a fresh output; no implicit resume/retry")
    out.mkdir(parents=True)
    atomic_json(out / "protocol.json", policy)
    for name in ["train.py", "scratch.py", "official.py"]:
        (out / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    atomic_weights(
        out / "initial.pt",
        dict(
            state_dict=cpu_state(model),
            epoch=0,
            step=0,
            protocol_binding=binding,
            initialization=initialization,
        ),
    )
    atomic_json(
        out / "host.json",
        dict(
            pid=os.getpid(),
            hostname=socket.gethostname(),
            gpu=torch.cuda.get_device_name(device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            started=time.time(),
        ),
    )
    step = 0
    best = float("inf")
    best_epoch = 0
    rng = np.random.default_rng(args.seed)

    def status(phase, epoch, **extra):
        atomic_json(
            out / "status.json",
            dict(
                phase=phase,
                epoch=epoch,
                step=step,
                total_epochs=args.epochs,
                best_epoch=best_epoch,
                pid=os.getpid(),
                updated=time.time(),
                **extra,
            ),
        )

    status("baseline", 0)
    baseline = validate(
        model,
        arrays,
        validation_indices,
        device,
        args.patch,
        args.batch,
        args.validation_seed,
        args.validation_text_weight,
        args.validation_edge_weight,
    )
    atomic_json(out / "initial-validation.json", baseline)
    best = baseline["loss"]
    atomic_weights(
        out / "best.pt",
        dict(
            state_dict=cpu_state(model),
            epoch=0,
            step=0,
            protocol_binding=binding,
            initialization=initialization,
        ),
    )
    started = time.time()
    with (out / "progress.jsonl").open("w") as log:
        for epoch in range(1, args.epochs + 1):
            order = np.random.default_rng(args.seed + epoch).permutation(train_indices)
            model.train()
            losses = []
            for cursor in range(0, len(order), args.batch):
                low, reference, mask = [
                    v.to(device)
                    for v in crop_batch(
                        arrays, order[cursor : cursor + args.batch], rng, args.patch
                    )
                ]
                optimizer.zero_grad(set_to_none=True)
                output = model(low)
                if output.shape != reference.shape:
                    raise ValueError("Wrong model geometry")
                loss, _ = objective(
                    output, reference, mask, args.text_weight, args.edge_weight
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite training loss")
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
                    status("training", epoch, cursor=cursor, loss=losses[-1])
            result = validate(
                model,
                arrays,
                validation_indices,
                device,
                args.patch,
                args.batch,
                args.validation_seed,
                args.validation_text_weight,
                args.validation_edge_weight,
            )
            state = cpu_state(model)
            payload = dict(
                state_dict=state,
                epoch=epoch,
                step=step,
                protocol_binding=binding,
                initialization=initialization,
            )
            if result["loss"] < best:
                best = result["loss"]
                best_epoch = epoch
                atomic_weights(out / "best.pt", payload)
            if epoch % 10 == 0 or epoch == args.epochs:
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
                elapsed_s=time.time() - started,
            )
            log.write(json.dumps(record) + "\n")
            log.flush()
            status("training", epoch, validation_loss=result["loss"])
            print(json.dumps(record), flush=True)
    if state_identity(model) == initial_tensor_hash:
        raise ValueError("Training did not change any model parameters")
    atomic_json(
        out / "COMPLETED.json",
        dict(
            epochs=args.epochs,
            steps=step,
            best_epoch=best_epoch,
            best_validation_loss=best,
            initial_validation_loss=baseline["loss"],
            initial_tensor_sha256=initial_tensor_hash,
            last_tensor_sha256=state_identity(model),
            best_sha256=sha256(out / "best.pt"),
            last_sha256=sha256(out / "last.pt"),
            protocol_binding=binding,
            finished=time.time(),
            scope="Completed fixed-budget training; initialization is explicitly bound in protocol.json. Validation proxy loss is not OCR/human readability or live network proof.",
        ),
    )
    status("completed", args.epochs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--packed", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--home", type=Path, default=project_home())
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--initialization", choices=["scratch", "official"], default="scratch"
    )
    p.add_argument("--exclude-bps", type=int, action="append")
    p.add_argument("--memory-fraction", type=float)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patch", type=int, default=96)
    p.add_argument("--seed", type=int, default=682)
    p.add_argument("--validation-seed", type=int, default=682)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--text-weight", type=float, default=2.0)
    p.add_argument("--edge-weight", type=float, default=0.05)
    p.add_argument("--validation-text-weight", type=float, default=2.0)
    p.add_argument("--validation-edge-weight", type=float, default=0.05)
    args = p.parse_args()
    if (
        min(args.epochs, args.batch, args.patch, args.lr) <= 0
        or min(
            args.text_weight,
            args.edge_weight,
            args.validation_text_weight,
            args.validation_edge_weight,
        )
        < 0
    ):
        raise ValueError("Invalid training budget/loss")
    run(args)


if __name__ == "__main__":
    main()
