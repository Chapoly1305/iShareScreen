"""Combine mirrored per-model curves into exportable comparison artifacts."""

import csv
from pathlib import Path
import os
import json


def device_curves(out, history, validations, gpu_label):
    """Reproduce numeric exports and label plotting hardware without changing frozen trainer."""
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
    with (out / "training-curve.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in history)
    records = [
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
        for r in validations
    ]
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
        for ax, key, title in [
            (axes[0, 1], "mac_psnr", "Mac validation PSNR (dB)"),
            (axes[1, 0], "mac_lpips", "Mac validation LPIPS"),
        ]:
            ax.plot(
                [r["epoch"] for r in records], [r[key] for r in records], marker="o"
            )
            ax.set_title(title)
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
        axes[1, 1].set_title(f"{gpu_label} forward latency at720p (ms)")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(out / f"curves.{ext}", dpi=160)
    plt.close(fig)


def report(root, models):
    records = []
    for model in models:
        path = root / "models" / model / "validation-performance-curve.csv"
        if not path.exists():
            continue
        identity = json.loads((path.parent / "HOST.json").read_text())
        with path.open() as f:
            for row in csv.DictReader(f):
                records.append(
                    dict(
                        model=model,
                        training_gpu=identity["gpu"],
                        host=identity["hostname"],
                        **{k: float(v) if v else None for k, v in row.items()},
                    )
                )
    if not records:
        return
    temporary = root / "COMPARISON.csv.tmp"
    with temporary.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(root / "COMPARISON.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    keys = [
        ("mac_psnr", "Mac validation PSNR (dB)"),
        ("mac_ssim", "Mac validation SSIM"),
        ("mac_lpips", "Mac validation LPIPS"),
        ("amp_fp16_forward_ms", "AMP forward latency on each training GPU (ms)"),
    ]
    for model in models:
        rows = [r for r in records if r["model"] == model]
        if not rows:
            continue
        label = (
            model
            + " ("
            + ("H100" if "H100" in rows[0]["training_gpu"] else "4090")
            + ")"
        )
        for ax, (key, title) in zip(axes.flat, keys):
            ax.plot(
                [r["epoch"] for r in rows],
                [r[key] for r in rows],
                marker=".",
                label=label,
            )
            ax.set_title(title)
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    for ext in ("png", "svg"):
        temp = root / f"COMPARISON.tmp.{ext}"
        fig.savefig(temp, dpi=160)
        temp.replace(root / f"COMPARISON.{ext}")
    plt.close(fig)
