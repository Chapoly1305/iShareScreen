"""Serial, resumable Mac-codec EDSR ablations with immutable data bindings.

The GPU job exports each development reconstruction without selecting a model
by PSNR. Frozen OCR scoring and checkpoint selection run separately on H100.
No historical test file is opened. All arms start from official EDSR weights.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from isharescreen.enhancement import models
from benchmarks.super_resolution.training_state import (
    atomic_json,
    atomic_weights,
    cpu_state,
    fingerprint,
)


GPU_UUID = "GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a"
ARMS = ["matched_codec", "native_glyph_mix", "glyph_edges", "glyph_conservative"]
DATASETS = [
    "mac-videotoolbox-720p-600k-420-pilot-v1",
    "mac-native-glyph-720p-600k-420-v1",
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def area_4_to_3(x):
    b, c, h, w = x.shape
    assert h % 4 == 0 and w % 4 == 0
    weights = x.new_tensor([[0.75, 0.25, 0, 0], [0, 0.5, 0.5, 0], [0, 0, 0.25, 0.75]])
    blocks = x.reshape(b, c, h // 4, 4, w // 4, 4)
    return torch.einsum("bchxwy,ix,jy->bchiwj", blocks, weights, weights).reshape(
        b, c, h // 4 * 3, w // 4 * 3
    )


def configure_parameters(model, arm):
    for name, param in model.named_parameters():
        param.requires_grad_(
            name.startswith("tail.")
            if arm == "glyph_conservative"
            else not name.startswith(("sub_mean.", "add_mean."))
        )
    return [p for p in model.parameters() if p.requires_grad]


def verify_device():
    info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    if GPU_UUID not in info:
        raise RuntimeError("Wrong GPU host for this frozen queue")
    users = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    if any(v.strip().isdigit() and int(v) != os.getpid() for v in users.splitlines()):
        raise RuntimeError(
            "Another GPU compute process is active; refuse contaminated timings"
        )
    return info


def training_data(project):
    rows = []
    bindings = []
    for dataset in DATASETS:
        root = project / "datasets/derived" / dataset
        data = read(root / "DATA.json")
        frozen = read(root / "FROZEN.json")
        assert (
            data["complete"]
            and data["role"] == "train_only"
            and sha(root / "DATA.json") == frozen["data_sha256"]
        )
        bindings.append(
            dict(
                dataset=dataset,
                data_sha256=frozen["data_sha256"],
                counts=data["counts"],
            )
        )
        for row in data["rows"]:
            assert row["split"] == "train"
            for role in ["lr", "hr"]:
                assert sha(root / row[role]) == row[role + "_sha256"]
            rows.append(dict(row, root=root, dataset=dataset))
    assert len(rows) == 512 and len({r["id"] for r in rows}) == 512
    for key in ["lr_pixel_sha256", "hr_pixel_sha256"]:
        assert len({r[key] for r in rows}) == 512
    return rows, bindings


class PairCache:
    def __init__(self, rows):
        # uint8 cache is about4.6GB, shared across arms. No GPU data allocation.
        self.rows = rows
        self.pairs = []
        for row in rows:
            low = np.asarray(Image.open(row["root"] / row["lr"]).convert("RGB")).copy()
            high = np.asarray(Image.open(row["root"] / row["hr"]).convert("RGB")).copy()
            assert low.shape == (720, 1280, 3) and high.shape == (1080, 1920, 3)
            self.pairs.append((low, high))

    def batch(self, arm, attempt, size=16, seed=20261103):
        rng = np.random.default_rng(seed + attempt)
        eligible = np.arange(384 if arm == "matched_codec" else len(self.rows))
        lows = []
        highs = []
        ids = []
        for i in rng.choice(eligible, size=size, replace=True):
            row = self.rows[i]
            box = row["transform"]["content_box"]
            lines = row["transform"]["line_boxes"]
            if lines and rng.random() < 0.8:
                box = lines[int(rng.integers(len(lines)))]
            cx = rng.uniform(box[0], box[2]) / 1.5
            cy = rng.uniform(box[1], box[3]) / 1.5
            x = int(np.clip(cx - 64, 0, 1280 - 128)) // 2 * 2
            y = int(np.clip(cy - 64, 0, 720 - 128)) // 2 * 2
            low, high = self.pairs[i]
            rx, ry = x * 3 // 2, y * 3 // 2
            lows.append(low[y : y + 128, x : x + 128])
            highs.append(high[ry : ry + 192, rx : rx + 192])
            ids.append(row["id"])
        tensor = lambda values: (
            torch.from_numpy(np.stack(values).transpose(0, 3, 1, 2).copy()).float()
            / 255
        )
        return tensor(lows), tensor(highs), ids


def update(model, optimizer, scaler, arm, low, high):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    rgb_sum = edge_sum = 0.0
    for offset in range(0, len(low), 4):
        x = low[offset : offset + 4].to("cuda:0")
        ref = high[offset : offset + 4].to("cuda:0")
        with torch.autocast("cuda", dtype=torch.float16):
            native = model(x * 255)
        pred = area_4_to_3(native.float() / 255)
        rgb = F.l1_loss(pred, ref)
        edge = (
            F.l1_loss(pred[:, :, 1:] - pred[:, :, :-1], ref[:, :, 1:] - ref[:, :, :-1])
            + F.l1_loss(
                pred[:, :, :, 1:] - pred[:, :, :, :-1],
                ref[:, :, :, 1:] - ref[:, :, :, :-1],
            )
        ) * 0.5
        loss = rgb + (0.1 * edge if arm == "glyph_edges" else 0)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite reconstruction loss")
        fraction = len(x) / len(low)
        scaler.scale(loss * fraction).backward()
        rgb_sum += float(rgb.detach()) * fraction
        edge_sum += float(edge.detach()) * fraction
    scaler.unscale_(optimizer)
    finite = all(
        p.grad is None or bool(torch.isfinite(p.grad).all())
        for group in optimizer.param_groups
        for p in group["params"]
    )
    old_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    applied = scaler.get_scale() >= old_scale
    if applied and not finite:
        raise FloatingPointError("Optimizer accepted nonfinite gradients")
    return dict(
        rgb_l1=rgb_sum,
        gradient_l1=edge_sum,
        loss=rgb_sum + (0.1 * edge_sum if arm == "glyph_edges" else 0),
        update_applied=applied,
        gradients_finite=finite,
        scaler_scale=scaler.get_scale(),
    )


def timing(model):
    verify_device()
    model.eval()
    x = torch.linspace(0, 255, 720 * 1280 * 3, device="cuda:0").reshape(1, 3, 720, 1280)
    result = {}
    for amp in [False, True]:
        values = []
        torch.cuda.reset_peak_memory_stats()
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.float16, enabled=amp),
        ):
            for _ in range(5):
                model(x)
            for _ in range(20):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                out = model(x)
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end))
            assert torch.isfinite(out).all()
        result["amp_fp16" if amp else "fp32"] = dict(
            median_ms=float(np.median(values)),
            p95_ms=float(np.percentile(values, 95)),
            samples_ms=values,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        )
    return dict(
        scope="Same lab RTX4090, batch1 EDSR720p ->1440p CUDA forward; excludes codec, transfers, resize and rendering",
        modes=result,
    )


def validation_inputs(inputs, reference):
    listing = read(inputs / "INPUTS.json")
    truth = read(reference / "manifest.json")
    assert listing["source_sha256"] == sha(reference / "manifest.json")
    assert (
        len(listing["images"]) == 32
        and truth["role"] == "development_validation_only_never_train_or_final_test"
    )
    rows = []
    for row in listing["images"]:
        state = next(s for s in truth["states"] if s["state_id"] == row["state_id"])
        low = inputs / row["input_path"]
        high = reference / state["reference"]["path"]
        assert (
            sha(low) == row["decoded_sha256"]
            and sha(high) == state["reference"]["sha256"]
        )
        rows.append(
            dict(
                row,
                low=np.asarray(Image.open(low).convert("RGB")),
                high=np.asarray(Image.open(high).convert("RGB")),
                state=state,
            )
        )
    return rows, listing["source_sha256"]


def evaluate(
    wrapper, cases, source_hash, out, arm, step, *, smoke=False, baseline=None
):
    out.mkdir(parents=True, exist_ok=True)
    receipt = out / "RESULTS.json"
    if receipt.exists():
        prior = read(receipt)
        assert prior["complete"] and prior["step"] == step
        for row in prior["images"]:
            assert sha(out / row["path"]) == row["sha256"]
        return prior
    wrapper.model.eval()
    images = []
    qualities = []
    parity = []
    chosen = [cases[0], cases[24]] if smoke else cases
    for row in chosen:
        native, latency = wrapper(row["low"])
        display = cv2.resize(native, (1920, 1080), interpolation=cv2.INTER_AREA)
        name = row["id"] + ".png"
        path = out / name
        Image.fromarray(display).save(path)
        mse = float(np.mean((display.astype(np.float64) - row["high"]) ** 2))
        qualities.append(
            dict(
                state_id=row["state_id"],
                group=row["group"],
                psnr=10 * math.log10(255**2 / mse),
            )
        )
        images.append(
            {
                k: row[k]
                for k in [
                    "group",
                    "category",
                    "state_id",
                    "config",
                    "source_age_s",
                    "height",
                    "codec_frame_count_valid",
                ]
            }
            | dict(
                id=row["id"],
                method=arm + f"-step{step:06d}",
                path=name,
                sha256=sha(path),
                input_sha256=row["decoded_sha256"],
                timing=latency,
            )
        )
        if step == 0 and baseline:
            old = next(
                v
                for v in baseline["images"]
                if v["state_id"] == row["state_id"] and v["method"] == "edsr-official"
            )
            target = baseline["root"] / old["path"]
            assert sha(target) == old["sha256"]
            error = np.abs(
                display.astype(np.int16)
                - np.asarray(Image.open(target).convert("RGB")).astype(np.int16)
            )
            assert error.max() <= 1, (row["id"], error.max())
            parity.append(
                dict(
                    state_id=row["state_id"],
                    maximum_rgb_difference=int(error.max()),
                    mean_rgb_difference=float(error.mean()),
                    exact=bool(not error.any()),
                )
            )
    quality_groups = defaultdict(list)
    for row in qualities:
        quality_groups[row["group"]].append(row["psnr"])
    result = dict(
        schema="mac_edsr_checkpoint_validation_v1",
        complete=True,
        arm=arm,
        step=step,
        source_sha256=source_hash,
        images=images,
        quality_rows=qualities,
        official_parity=parity,
        macro_psnr=float(np.mean([np.mean(v) for v in quality_groups.values()])),
        selection="PSNR is diagnostic only. Exported images require frozen OCR before checkpoint selection.",
        performance=timing(wrapper.model) if not smoke else None,
    )
    atomic_json(receipt, result)
    return result


def frozen_tensors(model):
    return {
        name: p.detach().cpu().clone()
        for name, p in model.named_parameters()
        if not p.requires_grad
    }


def assert_frozen(model, expected):
    parameters = dict(model.named_parameters())
    assert all(
        torch.equal(parameters[name].detach().cpu(), value)
        for name, value in expected.items()
    )


def curves(out, history, validations):
    if history:
        with (out / "training-curve.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
    if validations:
        rows = [
            dict(
                step=v["step"],
                macro_psnr=v["macro_psnr"],
                fp32_forward_ms=v["performance"]["modes"]["fp32"]["median_ms"],
                amp_forward_ms=v["performance"]["modes"]["amp_fp16"]["median_ms"],
            )
            for v in validations
        ]
        with (out / "validation-performance-curve.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def smoke(cache, cases, source_hash, out, policy):
    receipt = out / "SMOKE-PASSED.json"
    if receipt.exists():
        assert read(receipt)["binding"] == fingerprint(policy)
        return
    checks = []
    for arm in ARMS:
        torch.manual_seed(policy["seed"])
        wrapper = models.Upscaler("edsr", device="cuda:0", precision="amp-fp16")
        model = wrapper.model
        params = configure_parameters(model, arm)
        frozen = frozen_tensors(model)
        optimizer = torch.optim.Adam(params, lr=policy["lr"])
        scaler = torch.amp.GradScaler("cuda", init_scale=1024)
        before = [p.detach().clone() for p in params]
        low, high, ids = cache.batch(arm, 0)
        result = update(model, optimizer, scaler, arm, low, high)
        assert result["update_applied"] and result["gradients_finite"]
        assert any(not torch.equal(a, b) for a, b in zip(before, params))
        assert_frozen(model, frozen)
        path = out / "smoke" / arm
        path.mkdir(parents=True, exist_ok=True)
        checkpoint = path / "checkpoint.pt"
        atomic_weights(
            checkpoint,
            dict(
                state_dict=cpu_state(model),
                optimizer=optimizer.state_dict(),
                scaler=scaler.state_dict(),
            ),
        )
        model.eval()
        with torch.inference_mode():
            expected = model(low[:1].to("cuda:0") * 255).detach().cpu()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        torch.nn.Module.load_state_dict(model, payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scaler.load_state_dict(payload["scaler"])
        with torch.inference_mode():
            actual = model(low[:1].to("cuda:0") * 255).detach().cpu()
        assert torch.equal(actual, expected)
        validation = evaluate(
            wrapper, cases, source_hash, path / "validation", arm, 1, smoke=True
        )
        checks.append(
            dict(
                arm=arm,
                finite_update=True,
                parameters_changed=True,
                frozen_parameters_unchanged=True,
                exact_checkpoint_reload=True,
                validation_images=len(validation["images"]),
                sample_ids=ids,
                loss=result,
            )
        )
        del wrapper, model, optimizer, params, before
        torch.cuda.empty_cache()
        print("SMOKE", arm, "passed", flush=True)
    atomic_json(receipt, dict(binding=fingerprint(policy), passed=True, checks=checks))


def run_arm(cache, cases, source_hash, out, arm, policy, status, baseline):
    root = out / "models" / arm
    root.mkdir(parents=True, exist_ok=True)
    binding = fingerprint(dict(policy, arm=arm))
    if (root / "COMPLETED.json").exists():
        assert read(root / "COMPLETED.json")["binding"] == binding
        return
    torch.manual_seed(policy["seed"])
    wrapper = models.Upscaler("edsr", device="cuda:0", precision="amp-fp16")
    model = wrapper.model
    params = configure_parameters(model, arm)
    frozen = frozen_tensors(model)
    optimizer = torch.optim.Adam(params, lr=policy["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, policy["updates"], eta_min=policy["end_lr"]
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=1024)
    step = attempt = 0
    history = []
    validations = []
    elapsed = 0.0
    skipped = 0
    if (root / "last.pt").exists():
        payload = torch.load(root / "last.pt", map_location="cpu", weights_only=True)
        assert payload["binding"] == binding
        torch.nn.Module.load_state_dict(model, payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        step = payload["step"]
        attempt = payload["attempt"]
        history = payload["history"]
        validations = payload["validations"]
        elapsed = payload["elapsed"]
        skipped = payload["skipped"]
        torch.set_rng_state(payload["torch_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
        assert_frozen(model, frozen)
    started = time.monotonic()
    segment = started
    losses = []

    def checkpoint():
        assert_frozen(model, frozen)
        atomic_weights(
            root / "last.pt",
            dict(
                binding=binding,
                step=step,
                attempt=attempt,
                state_dict=cpu_state(model),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                history=history,
                validations=validations,
                skipped=skipped,
                elapsed=elapsed + time.monotonic() - started,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
            ),
        )

    def validate():
        atomic_json(
            status,
            dict(
                state="validating",
                arm=arm,
                step=step,
                pid=os.getpid(),
                updated=time.time(),
            ),
        )
        path = root / f"checkpoint-{step:06d}.pt"
        if not path.exists():
            atomic_weights(
                path,
                dict(
                    binding=binding,
                    step=step,
                    arm=arm,
                    state_dict=cpu_state(model),
                    initial_checkpoint_sha256=wrapper.provenance["checkpoint_sha256"],
                ),
            )
        v = evaluate(
            wrapper,
            cases,
            source_hash,
            root / f"validation-{step:06d}",
            arm,
            step,
            baseline=baseline,
        )
        validations.append(
            dict(
                step=step,
                macro_psnr=v["macro_psnr"],
                performance=v["performance"],
                checkpoint_sha256=sha(path),
            )
        )
        checkpoint()
        curves(root, history, validations)

    if step % 1000 == 0 and step not in {v["step"] for v in validations}:
        validate()
    while step < policy["updates"]:
        low, high, _ = cache.batch(arm, attempt, seed=policy["seed"])
        attempt += 1
        result = update(model, optimizer, scaler, arm, low, high)
        if not result["update_applied"]:
            skipped += 1
            if skipped > 20:
                raise FloatingPointError("Repeated AMP gradient overflow")
            continue
        scheduler.step()
        step += 1
        losses.append(result)
        if step % 100 == 0:
            now = time.monotonic()
            history.append(
                dict(
                    step=step,
                    attempt=attempt,
                    rgb_l1=float(np.mean([r["rgb_l1"] for r in losses])),
                    gradient_l1=float(np.mean([r["gradient_l1"] for r in losses])),
                    loss=float(np.mean([r["loss"] for r in losses])),
                    lr=optimizer.param_groups[0]["lr"],
                    elapsed_s=elapsed + now - started,
                    segment_seconds=now - segment,
                    successful_updates_per_second=len(losses) / (now - segment),
                    skipped_updates=skipped,
                )
            )
            losses = []
            segment = now
            atomic_json(
                status,
                dict(
                    state="training",
                    arm=arm,
                    step=step,
                    updates=policy["updates"],
                    pid=os.getpid(),
                    updated=time.time(),
                    last=history[-1],
                ),
            )
            print(arm, "step", step, "loss", history[-1]["loss"], flush=True)
            curves(root, history, validations)
        if step % 500 == 0:
            checkpoint()
        if step % 1000 == 0:
            validate()
            segment = time.monotonic()
    atomic_json(
        root / "COMPLETED.json",
        dict(
            binding=binding,
            updates=step,
            attempts=attempt,
            skipped_updates=skipped,
            last_sha256=sha(root / "last.pt"),
            checkpoints=validations,
            finished=time.time(),
            selection_pending="Frozen development OCR, not PSNR",
        ),
    )
    wrapper = model = optimizer = params = None
    torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = args.output / "STATUS.json"
    gpu = verify_device()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    rows, bindings = training_data(args.project)
    cases, source_hash = validation_inputs(args.inputs, args.reference)
    assert not {r["group_id"] for r in rows} & {r["group"] for r in cases}
    for train_key, source_key in [
        ("group_id", "group_id"),
        ("repository", "repository"),
        ("source_text_sha256", "source_sha256"),
    ]:
        heldout = {
            r["state"]["origin"][source_key]
            for r in cases
            if source_key in r["state"]["origin"]
        }
        assert not {r[train_key] for r in rows} & heldout, (
            "Training/development source overlap",
            train_key,
        )
    plan = read(args.plan)
    assert [r["id"] for r in plan["arms"]] == ARMS
    policy = dict(
        schema="mac_edsr_ablation_queue_v1",
        runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(args.plan),
        models_runner_sha256=sha(Path(models.__file__)),
        data=bindings,
        validation_inputs_sha256=sha(args.inputs / "INPUTS.json"),
        source_sha256=source_hash,
        seed=20261103,
        updates=6000,
        lr=2e-5,
        end_lr=2e-6,
        batch=16,
        microbatch=4,
        amp_initial_scale=1024,
        sampler="Uniform images with replacement; deterministicseed+attempt;80% code/text line anchor; even128px LR origins; no flips/rotations",
        loss_geometry="Float32 exact area4to3 after AMP native output, before loss; no quantization or clamp in differentiable loss",
        checkpoint_steps=list(range(0, 6001, 1000)),
        selection="Frozen development CER only; no historical test access",
    )
    if (args.output / "PROTOCOL.json").exists():
        assert read(args.output / "PROTOCOL.json") == policy, "Queue binding changed"
    else:
        atomic_json(args.output / "PROTOCOL.json", policy)
    atomic_json(
        args.output / "HOST.json",
        dict(
            hostname=platform.node(),
            gpu=gpu,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            pid=os.getpid(),
            started=time.time(),
        ),
    )
    try:
        atomic_json(
            status,
            dict(state="loading_train_pairs", pid=os.getpid(), updated=time.time()),
        )
        cache = PairCache(rows)
        baseline = read(args.baseline / "RESULTS.json")
        baseline["root"] = args.baseline
        smoke(cache, cases, source_hash, args.output, policy)
        if args.smoke_only:
            atomic_json(
                status, dict(state="smoke_passed", pid=os.getpid(), updated=time.time())
            )
            return
        for arm in ARMS:
            run_arm(
                cache, cases, source_hash, args.output, arm, policy, status, baseline
            )
        atomic_json(
            status,
            dict(
                state="training_completed",
                arms=ARMS,
                updated=time.time(),
                ocr_selection_pending=True,
            ),
        )
    except BaseException as error:
        atomic_json(
            status,
            dict(
                state="failed",
                error_type=type(error).__name__,
                error=str(error),
                pid=os.getpid(),
                updated=time.time(),
            ),
        )
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
