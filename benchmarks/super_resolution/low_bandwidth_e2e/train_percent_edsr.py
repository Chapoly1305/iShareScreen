"""Fine-tune historical EDSR with frozen numeric-percent anchored paired crops."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import time

import numpy as np
import torch

from benchmarks.super_resolution.low_bandwidth_e2e.evaluate_edsr_gpu import get_model
from benchmarks.super_resolution.text_corpus.train import open_data
from benchmarks.super_resolution.training_state import (
    atomic_json,
    atomic_weights,
    cpu_state,
    fingerprint,
)
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def make_crop(
    arrays: dict,
    index: int,
    rng: np.random.Generator,
    patch: int,
    anchor: list[int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    low, high = arrays["lr"][index], arrays["hr"][index]
    h, w = low.shape[:2]
    if anchor is None:
        x = int(rng.integers(w - patch + 1))
        y = int(rng.integers(h - patch + 1))
    else:
        a, b, c, d = anchor
        center_x = (a + c) / 4
        center_y = (b + d) / 4
        x = int(
            np.clip(
                center_x - patch / 2 + rng.integers(-patch // 4, patch // 4 + 1),
                0,
                w - patch,
            )
        )
        y = int(
            np.clip(
                center_y - patch / 2 + rng.integers(-patch // 4, patch // 4 + 1),
                0,
                h - patch,
            )
        )
    lo = np.asarray(low[y : y + patch, x : x + patch]).copy()
    hi = np.asarray(high[2 * y : 2 * (y + patch), 2 * x : 2 * (x + patch)]).copy()
    if lo.shape != (patch, patch, 3) or hi.shape != (patch * 2, patch * 2, 3):
        raise ValueError("Misaligned paired crop")
    text_mask = np.zeros((patch * 2, patch * 2), np.float32)
    symbol_mask = np.zeros_like(text_mask)
    start, end = arrays["offsets"][index : index + 2]
    for a, b, c, d, _ in np.asarray(arrays["boxes"][start:end]).astype(np.int32):
        left, top = max(0, a - 2 * x), max(0, b - 2 * y)
        right, bottom = min(patch * 2, c - 2 * x), min(patch * 2, d - 2 * y)
        if right > left and bottom > top:
            text_mask[top:bottom, left:right] = 1
    if anchor is not None:
        a, b, c, d = anchor
        left, top = max(0, a - 2 * x), max(0, b - 2 * y)
        right, bottom = min(patch * 2, c - 2 * x), min(patch * 2, d - 2 * y)
        if right <= left or bottom <= top:
            raise ValueError("Chosen symbol anchor outside crop")
        symbol_mask[top:bottom, left:right] = 1
    return lo, hi, text_mask, symbol_mask


def batch(
    arrays: dict,
    train_indices: np.ndarray,
    symbol_rows: list[dict],
    rng: np.random.Generator,
    size: int,
    patch: int,
    symbol_fraction: float,
) -> tuple[torch.Tensor, ...]:
    result = [[], [], [], []]
    for _ in range(size):
        if rng.random() < symbol_fraction:
            entry = symbol_rows[int(rng.integers(len(symbol_rows)))]
            index = entry["index"]
            anchor = entry["boxes"][int(rng.integers(len(entry["boxes"])))]
        else:
            index = int(train_indices[int(rng.integers(len(train_indices)))])
            anchor = None
        for target, value in zip(result, make_crop(arrays, index, rng, patch, anchor)):
            target.append(value)
    lo = torch.from_numpy(np.stack(result[0])).permute(0, 3, 1, 2).float() / 255
    hi = torch.from_numpy(np.stack(result[1])).permute(0, 3, 1, 2).float() / 255
    text = torch.from_numpy(np.stack(result[2]))[:, None]
    symbol = torch.from_numpy(np.stack(result[3]))[:, None]
    return lo, hi, text, symbol


def objective(
    pred: torch.Tensor,
    ref: torch.Tensor,
    text: torch.Tensor,
    symbol: torch.Tensor,
    text_weight: float,
    symbol_weight: float,
    edge_weight: float,
) -> torch.Tensor:
    weight = 1 + text_weight * text + symbol_weight * symbol
    pixel = ((pred - ref).abs() * weight).sum() / (weight.sum() * 3)
    dx = (pred[..., 1:] - pred[..., :-1]) - (ref[..., 1:] - ref[..., :-1])
    dy = (pred[..., 1:, :] - pred[..., :-1, :]) - (ref[..., 1:, :] - ref[..., :-1, :])
    return pixel + edge_weight * (dx.abs().mean() + dy.abs().mean()) / 2


def validation(
    model: torch.nn.Module,
    arrays: dict,
    symbol_rows: list[dict],
    all_val: np.ndarray,
    device: torch.device,
    policy: dict,
) -> float:
    # Fixed source-disjoint crops: all percent-bearing validation views plus
    # deterministic generic validation views. No OCR/test pixels choose a checkpoint.
    rng = np.random.default_rng(policy["validation_seed"])
    chosen = [
        (row["index"], row["boxes"][0])
        for row in symbol_rows
        if row["split"] == "validation"
    ]
    chosen.extend((int(i), None) for i in all_val[: min(16, len(all_val))])
    values = []
    model.eval()
    with torch.inference_mode():
        for index, anchor in chosen:
            lo, hi, text, symbol = make_crop(
                arrays, index, rng, policy["patch"], anchor
            )
            t = torch.from_numpy(lo).permute(2, 0, 1)[None].float().to(device) / 255
            r = torch.from_numpy(hi).permute(2, 0, 1)[None].float().to(device) / 255
            m = torch.from_numpy(text)[None, None].to(device)
            s = torch.from_numpy(symbol)[None, None].to(device)
            values.append(
                float(
                    objective(
                        model(t),
                        r,
                        m,
                        s,
                        policy["text_weight"],
                        policy["symbol_weight"],
                        policy["edge_weight"],
                    )
                )
            )
    return float(np.mean(values))


def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU required for percent fine-tuning")
    device = torch.device("cuda:0")
    data, arrays = open_data(args.packed, verify=True)
    index = read(args.percent_index)
    if index["packed_sha256"] != sha256(args.packed):
        raise ValueError("Percent index belongs to another packed data version")
    train_indices = np.asarray(
        [i for i, row in enumerate(data["rows"]) if row["split"] == "train"],
        dtype=np.int64,
    )
    val_indices = np.asarray(
        [i for i, row in enumerate(data["rows"]) if row["split"] == "validation"],
        dtype=np.int64,
    )
    symbol_rows = [row for row in index["rows"] if row["split"] == "train"]
    if len(symbol_rows) < 200 or not len(val_indices):
        raise ValueError("Insufficient source-bound training/validation symbols")
    for row in index["rows"]:
        original = data["rows"][row["index"]]
        if (
            original["id"] != row["id"]
            or original["split"] != row["split"]
            or original["labels_sha256"] != row["labels_sha256"]
        ):
            raise ValueError("Percent index row identity changed")
    weights = read(args.weights)["models"]
    model, identity = get_model("old_finetuned_edsr_x2", weights, args.home)
    model.to(device).train().requires_grad_(True)
    policy = dict(
        schema="isharescreen_percent_edsr_training_v1",
        model="EDSR baseline x2",
        initial_checkpoint_sha256=identity["checkpoint_sha256"],
        packed_sha256=sha256(args.packed),
        percent_index_sha256=sha256(args.percent_index),
        heldout_protocol_sha256=sha256(args.heldout_protocol),
        runner_sha256=sha256(Path(__file__)),
        seed=args.seed,
        steps=args.steps,
        checkpoint_every=args.checkpoint_every,
        batch=args.batch,
        patch=args.patch,
        lr=args.lr,
        symbol_fraction=args.symbol_fraction,
        text_weight=args.text_weight,
        symbol_weight=args.symbol_weight,
        edge_weight=args.edge_weight,
        validation_seed=args.validation_seed,
        precision="fp32",
        tf32=False,
        optimizer="Adam default betas/epsilon",
        scheduler="Cosine anneal by update to 1% initial learning rate",
        sampling="With replacement: fixed fraction source-known numeric-percent word anchor, rest uniform original training views; no test or external holdout crops.",
        selection="Lowest fixed source-disjoint packed validation pixel/edge loss among saved checkpoints; external percent validation OCR used only after training to decide test roster.",
    )
    binding = fingerprint(policy)
    out = args.output
    if out.exists():
        raise FileExistsError("Fresh output required; explicit verified resume only")
    out.mkdir(parents=True)
    atomic_json(out / "protocol.json", policy)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )
    rng = np.random.default_rng(args.seed)
    atomic_json(
        out / "host.json",
        dict(
            hostname=socket.gethostname(),
            pid=os.getpid(),
            gpu=torch.cuda.get_device_name(device),
            gpu_uuid=None,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            started=time.time(),
        ),
    )
    initial_loss = validation(model, arrays, index["rows"], val_indices, device, policy)
    best, best_step = initial_loss, 0
    atomic_weights(
        out / "best.pt",
        dict(
            state_dict=cpu_state(model),
            step=0,
            protocol_binding=binding,
            validation_loss=initial_loss,
        ),
    )
    started = time.time()
    with (out / "progress.jsonl").open("w") as log:
        for step in range(1, args.steps + 1):
            model.train()
            lo, hi, text, symbol = [
                x.to(device)
                for x in batch(
                    arrays,
                    train_indices,
                    symbol_rows,
                    rng,
                    args.batch,
                    args.patch,
                    args.symbol_fraction,
                )
            ]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(lo)
            loss = objective(
                prediction,
                hi,
                text,
                symbol,
                args.text_weight,
                args.symbol_weight,
                args.edge_weight,
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite percent training loss")
            loss.backward()
            if not all(
                torch.isfinite(p.grad).all().item()
                for p in model.parameters()
                if p.grad is not None
            ):
                raise FloatingPointError("Nonfinite gradient")
            optimizer.step()
            scheduler.step()
            if step % args.checkpoint_every == 0 or step == args.steps:
                value = validation(
                    model, arrays, index["rows"], val_indices, device, policy
                )
                payload = dict(
                    state_dict=cpu_state(model),
                    step=step,
                    protocol_binding=binding,
                    validation_loss=value,
                )
                atomic_weights(out / f"step-{step:04d}.pt", payload)
                if value < best:
                    best, best_step = value, step
                    atomic_weights(out / "best.pt", payload)
                atomic_weights(
                    out / "last.pt",
                    dict(
                        payload,
                        optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),
                        numpy_rng=rng.bit_generator.state,
                        torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state_all(),
                    ),
                )
                record = dict(
                    step=step,
                    train_loss=float(loss.detach()),
                    validation_loss=value,
                    best_step=best_step,
                    elapsed_s=time.time() - started,
                )
                log.write(json.dumps(record) + "\n")
                log.flush()
                atomic_json(
                    out / "status.json",
                    dict(
                        phase="running", **record, pid=os.getpid(), updated=time.time()
                    ),
                )
                print("PERCENT TRAIN", json.dumps(record), flush=True)
    atomic_json(
        out / "COMPLETED.json",
        dict(
            protocol_binding=binding,
            steps=args.steps,
            initial_validation_loss=initial_loss,
            best_validation_loss=best,
            best_step=best_step,
            best_sha256=sha256(out / "best.pt"),
            last_sha256=sha256(out / "last.pt"),
            elapsed_s=time.time() - started,
            finished=time.time(),
        ),
    )
    atomic_json(
        out / "status.json",
        dict(
            phase="completed",
            step=args.steps,
            best_step=best_step,
            finished=time.time(),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--percent-index", type=Path, required=True)
    parser.add_argument("--heldout-protocol", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=682)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patch", type=int, default=96)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--symbol-fraction", type=float, default=0.8)
    parser.add_argument("--text-weight", type=float, default=2)
    parser.add_argument("--symbol-weight", type=float, default=4)
    parser.add_argument("--edge-weight", type=float, default=0.05)
    parser.add_argument("--validation-seed", type=int, default=781)
    args = parser.parse_args()
    if (
        args.steps <= 0
        or args.checkpoint_every <= 0
        or args.steps % args.checkpoint_every
    ):
        raise ValueError("Steps must divide into fixed checkpoint intervals")
    if args.batch <= 0 or args.patch <= 0 or not 0 <= args.symbol_fraction <= 1:
        raise ValueError("Invalid batch, patch or sampler fraction")
    run(args)


if __name__ == "__main__":
    main()
