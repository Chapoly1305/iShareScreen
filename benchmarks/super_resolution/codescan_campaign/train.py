"""Resumable source-bound training and matched video validation curves."""

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import time

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from benchmarks.super_resolution.codescan_campaign.data import read
from benchmarks.super_resolution.legacy_validation_compare import adapter
from benchmarks.super_resolution.training_state import (
    atomic_json,
    atomic_weights,
    cpu_state,
    fingerprint,
)
from benchmarks.super_resolution.validation_gpu_quality import quality
from benchmarks.super_resolution.train import perceptual_network
from benchmarks.super_resolution.video_perceptual import normalized_tensor
from benchmarks.super_resolution.candidate_training import predict as candidate_predict
from isharescreen.enhancement.models import sha256


def area_4_to_3(x):
    """Exact pixel-area resize (OpenCV INTER_AREA geometry), differentiable."""
    b, c, h, w = x.shape
    assert h % 4 == 0 and w % 4 == 0
    weights = x.new_tensor([[0.75, 0.25, 0, 0], [0, 0.5, 0.5, 0], [0, 0, 0.25, 0.75]])
    blocks = x.reshape(b, c, h // 4, 4, w // 4, 4)
    return torch.einsum("bchxwy,ix,jy->bchiwj", blocks, weights, weights).reshape(
        b, c, h // 4 * 3, w // 4 * 3
    )


def crop_coordinates(row, seed):
    rng = np.random.default_rng(seed)
    box = row["transform"]["content_box"]
    lines = row["transform"]["line_boxes"]
    if lines and rng.random() < 0.8:
        box = lines[int(rng.integers(len(lines)))]
    # Sample a point in real content and jitter the window around it; no blank-padding-only crops.
    cx = rng.uniform(box[0], box[2]) / 1.5
    cy = rng.uniform(box[1], box[3]) / 1.5
    x = int(np.clip(cx - 64, 0, 1280 - 128)) // 2 * 2
    y = int(np.clip(cy - 64, 0, 720 - 128)) // 2 * 2
    return x, y


class Crops(Dataset):
    def __init__(self, root, rows, epoch=0, seed=682):
        self.root, self.rows, self.epoch, self.seed = root, rows, epoch, seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        identity = int(row["id"][:12], 16)
        x, y = crop_coordinates(
            row, self.seed + self.epoch * 100000000000000 + identity
        )
        with Image.open(self.root / row["lr"]) as image:
            low = np.asarray(image.convert("RGB").crop((x, y, x + 128, y + 128))).copy()
        rx, ry = x * 3 // 2, y * 3 // 2
        with Image.open(self.root / row["hr"]) as image:
            high = np.asarray(
                image.convert("RGB").crop((rx, ry, rx + 192, ry + 192))
            ).copy()
        return torch.from_numpy(low).permute(2, 0, 1).float() / 255, torch.from_numpy(
            high
        ).permute(2, 0, 1).float() / 255


def forward(model, key, low):
    if key == "plainusr" and model.training and torch.is_grad_enabled():
        return candidate_predict(model, key, low)
    out = model(low * 255 if key == "edsr" else low)
    return out / 255 if key == "edsr" else out


def trainable_parameters(model):
    model.requires_grad_(True)
    # Preserve EDSR's fixed RGB normalization; only learned reconstruction layers train.
    for module in model.modules():
        if module.__class__.__name__ == "MeanShift":
            module.requires_grad_(False)
    return [p for p in model.parameters() if p.requires_grad]


def aggregate_crop_quality(mses, ssims):
    # Exact reconstruction has infinite PSNR. Preserve that case as JSON null;
    # aggregate equal-size crop MSE before the logarithm rather than cap PSNR.
    mse = float(np.mean(mses))
    return dict(
        psnr=10 * math.log10(65025 / mse) if mse else None,
        ssim=float(np.mean(ssims)),
        views=len(mses),
        exact_crops=sum(v == 0 for v in mses),
        aggregation="PSNR of equal-view mean squared error",
    )


def performance(model, key, device, smoke=False):
    """CUDA-event forward only; fixed shapes, no decoding/upload/download in FPS."""
    values = {}
    model.eval()
    x = torch.linspace(0, 1, 1280 * 720 * 3, device=device).reshape(1, 3, 720, 1280)
    for amp in (False, True):
        samples = []
        torch.cuda.reset_peak_memory_stats()
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.float16, enabled=amp),
        ):
            for _ in range(2 if smoke else 5):
                y = forward(model, key, x)
                assert torch.isfinite(y).all()
            for _ in range(5 if smoke else 20):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                y = forward(model, key, x)
                end.record()
                end.synchronize()
                assert torch.isfinite(y).all()
                samples.append(start.elapsed_time(end))
        median = float(np.median(samples))
        values["amp_fp16" if amp else "fp32"] = dict(
            median_ms=median,
            p95_ms=float(np.percentile(samples, 95)),
            forward_fps=1000 / median,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            samples_ms=samples,
            scope="Batch1 CUDA forward720p ->1440p; no codec, transfer, FHD resize or rendering. Not Mac client performance.",
        )
    return values


def mac_cases(base):
    cases = []
    for run in read(base / "INPUTS.json")["runs"]:
        if run["config"] != "720p-600k-420":
            continue
        for sample in run["samples"]:
            for role in ("lr", "hr"):
                assert sha256(base / sample[role]) == sample[role + "_sha256"]
            cases.append(
                dict(
                    clip=run["clip"],
                    frame=sample["frame"],
                    lr=base / sample["lr"],
                    hr=base / sample["hr"],
                )
            )
    assert len(cases) == 72 and len({c["clip"] for c in cases}) == 24
    return cases


def validate(wrapper, key, root, rows, cases, device, perceptual, smoke=False, epoch=0):
    model = wrapper.model
    model.eval()
    scored = []
    with torch.inference_mode():
        for case in cases:
            low = np.asarray(Image.open(case["lr"]).convert("RGB"))
            high = np.asarray(Image.open(case["hr"]).convert("RGB"))
            native, _ = wrapper(low)
            out = cv2.resize(native, (1920, 1080), interpolation=cv2.INTER_AREA)
            metrics = quality(high, out)
            metrics["lpips"] = float(
                perceptual(
                    normalized_tensor(high, device), normalized_tensor(out, device)
                ).item()
            )
            scored.append(dict(clip=case["clip"], frame=case["frame"], **metrics))
    grouped = {}
    for case in scored:
        grouped.setdefault(case["clip"], []).append(case)
    mac = {
        k: float(
            np.mean([np.mean([r[k] for r in group]) for group in grouped.values()])
        )
        for k in ("psnr", "ssim", "lpips")
    }
    # Fixed held-out crops, independent of training epoch; these do not replace full-video validation.
    errors = []
    ssims = []
    loader = DataLoader(
        Crops(root, rows, epoch=0, seed=1682),
        batch_size=16,
        shuffle=False,
        num_workers=2,
    )
    with torch.inference_mode():
        for lo, hi in loader:
            out = area_4_to_3(forward(model, key, lo.to(device))).clamp(0, 1)
            out = (out * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
            refs = (hi * 255).round().byte().permute(0, 2, 3, 1).numpy()
            for ref, pred in zip(refs, out):
                q = quality(ref, pred)
                errors.append(float(np.mean((ref.astype(np.float64) - pred) ** 2)))
                ssims.append(q["ssim"])
    return dict(
        epoch=epoch,
        mac=mac,
        mac_frames=scored,
        codescan_fixed_crops=aggregate_crop_quality(errors, ssims),
        performance=performance(model, key, device, smoke=smoke),
    )


def curves(out, history, validations):
    """Machine-readable curves and standalone PNG/SVG figures on each epoch."""
    with (out / "training-curve.csv").open("w") as f:
        fields = [
            "epoch",
            "step",
            "train_l1",
            "lr",
            "epoch_seconds",
            "samples_per_second",
            "elapsed_seconds",
            "peak_gpu_bytes",
            "amp_skipped_updates",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in history)
    records = []
    for r in validations:
        records.append(
            dict(
                epoch=r["epoch"],
                mac_psnr=r["mac"]["psnr"],
                mac_ssim=r["mac"]["ssim"],
                mac_lpips=r["mac"]["lpips"],
                codescan_crop_psnr=r["codescan_fixed_crops"]["psnr"],
                codescan_crop_ssim=r["codescan_fixed_crops"]["ssim"],
                fp32_forward_ms=r["performance"]["fp32"]["median_ms"],
                amp_fp16_forward_ms=r["performance"]["amp_fp16"]["median_ms"],
            )
        )
    if records:
        with (out / "validation-performance-curve.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    if history:
        axes[0, 0].plot([r["epoch"] for r in history], [r["train_l1"] for r in history])
        axes[0, 0].set_title("Training RGB L1")
    if records:
        axes[0, 1].plot(
            [r["epoch"] for r in records], [r["mac_psnr"] for r in records], marker="o"
        )
        axes[0, 1].set_title("Mac validation PSNR (dB)")
        axes[1, 0].plot(
            [r["epoch"] for r in records], [r["mac_lpips"] for r in records], marker="o"
        )
        axes[1, 0].set_title("Mac validation LPIPS")
        for key, label in [
            ("fp32_forward_ms", "FP32"),
            ("amp_fp16_forward_ms", "AMP FP16"),
        ]:
            axes[1, 1].plot(
                [r["epoch"] for r in records],
                [r[key] for r in records],
                marker="o",
                label=label,
            )
        axes[1, 1].legend()
        axes[1, 1].set_title("4090 forward latency at720p (ms)")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(out / f"curves.{suffix}", dpi=160)
    plt.close(fig)


def run(
    project, campaign, data_path, key, epochs=100, smoke=False, stop_after_epoch=None
):
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.manual_seed(682)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = "cuda:0"
    assert torch.cuda.is_available()
    root = data_path.parent
    data = read(data_path)
    assert data["complete"]
    if not smoke:
        gate = read(campaign / "SMOKE-PASSED.json")
        assert gate["train_runner_sha256"] == sha256(Path(__file__)), (
            "Training code changed since smoke gate"
        )
    train = [r for r in data["rows"] if r["split"] == "train"]
    val = [r for r in data["rows"] if r["split"] == "validation"]
    assert not ({r["group_id"] for r in train} & {r["group_id"] for r in val})
    assert len(train) == (128 if smoke else 9600) and len(val) == (
        16 if smoke else 1200
    )
    out = campaign / ("smoke" if smoke else "models") / key
    out.mkdir(parents=True, exist_ok=True)
    wrapper = adapter(key, device)
    model = wrapper.model
    parameters = trainable_parameters(model)
    policy = dict(
        schema="codescan_training_v1",
        model=key,
        epochs=epochs,
        smoke=smoke,
        seed=682,
        lr=1e-4,
        batch=32,
        microbatch=8,
        data_sha256=sha256(data_path),
        runner_sha256=sha256(Path(__file__)),
        data_runner_sha256=sha256(Path(__file__).with_name("data.py")),
        initial_checkpoint_sha256=wrapper.provenance["checkpoint_sha256"],
        loss="RGB L1 exact area4to3 at display reference resolution",
        frozen_normalization="EDSR MeanShift weights/bias remain fixed",
        crop="One128px LR crop per source per epoch; 80% valid code-line anchor, else content; even origins; no flips/rotations",
        selection="Highest macro24-clip Mac PSNR at epochs0,5,...100; CodeSCAN test excluded",
        optimizer="Adam cosine1e-4 to1e-6; AMP FP16 with dynamic scaler",
        validation="Mac72 full frames; CodeSCAN one deterministic192px reference crop per validation sample; all FP32",
    )
    binding = fingerprint(policy)
    if (out / "PROTOCOL.json").exists():
        assert read(out / "PROTOCOL.json") == policy, "Training configuration changed"
    else:
        atomic_json(out / "PROTOCOL.json", policy)
    if (out / "COMPLETED.json").exists():
        done = read(out / "COMPLETED.json")
        assert (
            done["binding"] == binding
            and sha256(out / "best.pt") == done["best_sha256"]
        )
        return done
    optimizer = torch.optim.Adam(parameters, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    scaler = torch.amp.GradScaler("cuda")
    history = []
    validations = []
    start_epoch = 0
    best = -math.inf
    best_epoch = 0
    step = 0
    elapsed = 0
    if (out / "last.pt").exists():
        payload = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        assert payload["binding"] == binding
        torch.nn.Module.load_state_dict(model, payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        start_epoch = payload["epoch"]
        history = payload["history"]
        validations = payload["validations"]
        best = payload["best"]
        best_epoch = payload["best_epoch"]
        step = payload["step"]
        elapsed = payload["elapsed"]
        torch.set_rng_state(payload["torch_rng"])
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    cases = mac_cases(
        project
        / "benchmarks/super_resolution/experiments/validation-baselines-2026-10-29"
    )
    perceptual, lpips_identity = perceptual_network(device)
    atomic_json(
        out / "HOST.json",
        dict(
            hostname=platform.node(),
            gpu=torch.cuda.get_device_name(),
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            pid=os.getpid(),
            gpu_uuid=os.environ.get("CUDA_VISIBLE_DEVICES"),
            lpips=lpips_identity,
            started=time.time(),
            resumed_epoch=start_epoch,
        ),
    )
    started = time.time()

    def checkpoint(epoch):
        atomic_weights(
            out / "last.pt",
            dict(
                binding=binding,
                epoch=epoch,
                state_dict=cpu_state(model),
                optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(),
                scaler=scaler.state_dict(),
                history=history,
                validations=validations,
                best=best,
                best_epoch=best_epoch,
                step=step,
                elapsed=elapsed + time.time() - started,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
            ),
        )

    def score(epoch):
        nonlocal best, best_epoch
        result = validate(
            wrapper, key, root, val, cases, device, perceptual, smoke, epoch
        )
        validations.append(result)
        atomic_json(out / f"validation-{epoch:03d}.json", result)
        if epoch == 0:
            official = read(
                project
                / "benchmarks/super_resolution/experiments/validation-baselines-2026-10-29/COVERAGE-AUDIT.json"
            )
            original = (
                project
                / f"benchmarks/super_resolution/experiments/validation-baselines-2026-10-29/evaluations/{key}/records.jsonl"
            )
            assert sha256(original) == official["results_sha256"][key]
            expected = {
                (r["clip"], r["frame"]): r
                for r in map(json.loads, original.read_text().splitlines())
                if r["config"] == "720p-600k-420"
            }
            for actual in result["mac_frames"]:
                prior = expected[actual["clip"], actual["frame"]]
                for metric in ("psnr", "ssim", "lpips"):
                    assert abs(actual[metric] - prior[metric]) < 1e-5, (
                        key,
                        metric,
                        actual[metric],
                        prior[metric],
                    )
            atomic_json(
                out / "OFFICIAL-PARITY.json",
                dict(passed=True, frames=72, tolerance=1e-5),
            )
        if result["mac"]["psnr"] > best:
            best = result["mac"]["psnr"]
            best_epoch = epoch
            atomic_weights(
                out / "best.pt",
                dict(
                    binding=binding,
                    epoch=epoch,
                    state_dict=cpu_state(model),
                    mac_psnr=best,
                ),
            )
        return result

    if not validations:
        score(0)
        checkpoint(0)
        curves(out, history, validations)
    for epoch in range(start_epoch + 1, epochs + 1):
        model.train()
        torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()
        loss_sum = 0
        seen = 0
        skipped = 0
        generator = torch.Generator().manual_seed(682 + epoch)
        loader = DataLoader(
            Crops(root, train, epoch),
            batch_size=32,
            shuffle=True,
            generator=generator,
            num_workers=4,
            pin_memory=True,
        )
        for lo, hi in loader:
            optimizer.zero_grad(set_to_none=True)
            n = len(lo)
            batch_loss = 0
            for offset in range(0, n, 8):
                low = lo[offset : offset + 8].to(device, non_blocking=True)
                ref = hi[offset : offset + 8].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    native = forward(model, key, low)
                prediction = area_4_to_3(native.float())
                loss = torch.nn.functional.l1_loss(prediction, ref)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss {key} epoch{epoch}")
                scaler.scale(loss * (len(low) / n)).backward()
                batch_loss += float(loss.detach()) * len(low)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            skipped += int(scaler.get_scale() < old_scale)
            step += 1
            loss_sum += batch_loss
            seen += n
            if step % 20 == 0:
                atomic_json(
                    out / "STATUS.json",
                    dict(
                        phase="training",
                        epoch=epoch,
                        total_epochs=epochs,
                        step=step,
                        seen=seen,
                        total=len(train),
                        loss=loss_sum / seen,
                        best_epoch=best_epoch,
                        best_mac_psnr=best,
                        updated=time.time(),
                    ),
                )
        if skipped > max(2, len(loader) // 20):
            raise FloatingPointError(f"Too many AMP skipped updates: {skipped}")
        torch.cuda.synchronize()
        seconds = time.time() - epoch_start
        history.append(
            dict(
                epoch=epoch,
                step=step,
                train_l1=loss_sum / seen,
                lr=optimizer.param_groups[0]["lr"],
                epoch_seconds=seconds,
                samples_per_second=seen / seconds,
                elapsed_seconds=elapsed + time.time() - started,
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                amp_skipped_updates=skipped,
            )
        )
        scheduler.step()
        if epoch % 5 == 0 or epoch == epochs or smoke:
            score(epoch)
        checkpoint(epoch)
        if epoch % 5 == 0 or epoch == epochs:
            atomic_weights(
                out / f"epoch-{epoch:03d}.pt",
                dict(binding=binding, epoch=epoch, state_dict=cpu_state(model)),
            )
        curves(out, history, validations)
        atomic_json(
            out / "STATUS.json",
            dict(
                phase="epoch_complete",
                epoch=epoch,
                total_epochs=epochs,
                step=step,
                best_epoch=best_epoch,
                best_mac_psnr=best,
                updated=time.time(),
            ),
        )
        print(
            "EPOCH",
            key,
            epoch,
            "/",
            epochs,
            "L1",
            loss_sum / seen,
            "best",
            best_epoch,
            best,
            flush=True,
        )
        if stop_after_epoch == epoch and epoch < epochs:
            atomic_json(
                out / "INTERRUPTION-TEST.json",
                dict(epoch=epoch, step=step, last_sha256=sha256(out / "last.pt")),
            )
            return dict(stopped_at_epoch=epoch, step=step)
    done = dict(
        binding=binding,
        epochs=epochs,
        steps=step,
        best_epoch=best_epoch,
        best_mac_psnr=best,
        best_sha256=sha256(out / "best.pt"),
        finished=time.time(),
    )
    atomic_json(out / "COMPLETED.json", done)
    return done
