"""Render audited epoch histories without smoothing or new model inference."""
import argparse
import json
import math
from pathlib import Path

from isharescreen.enhancement.models import sha256

METRICS = ("psnr", "ssim", "lpips")
LABELS = {"psnr": "PSNR (dB; higher is better)", "ssim": "SSIM (higher is better)",
          "lpips": "LPIPS (lower is better)"}
COLORS = {"edsr": "#2563EB", "imdn": "#D97706", "safmn": "#16865A"}
NAMES = ("full_curves", "selection_curves", "loss_curves")


def validate(result, continuation_epoch):
    if result.get("schema") != "sr_epoch_results_v1" or not result.get("complete_cv"):
        raise ValueError("Use a completed saved-artifact audit, not raw partial logs")
    keys = list(result["summary"])
    caps = {j["epochs"] for j in result["jobs"]}
    if len(caps) != 1:
        raise ValueError("A shared epoch axis requires the same completed ceiling")
    cap = caps.pop()
    if continuation_epoch is not None and not 0 < continuation_epoch < cap:
        raise ValueError("Continuation boundary must be inside the completed epoch range")
    for tier in ("full", "selection"):
        grids = []
        for key in keys:
            rows = sorted((r for r in result["curves"] if r["model"] == key and r["tier"] == tier),
                          key=lambda r: r["epoch"])
            grid = [r["epoch"] for r in rows]
            if (not grid or grid[0] != 0 or grid[-1] != cap or len(grid) != len(set(grid))
                    or continuation_epoch is not None and continuation_epoch not in grid
                    or not all(math.isfinite(r[m]) for r in rows for m in METRICS)):
                raise ValueError("Incomplete, duplicate or nonfinite validation curve")
            grids.append(grid)
        if any(g != grids[0] for g in grids[1:]):
            raise ValueError("Models must share the same observation epochs")
    for job in result["jobs"]:
        rows = sorted((r for r in result["losses"] if r["model"] == job["model"] and r["fold"] == job["fold"]),
                      key=lambda r: r["epoch"])
        if ([r["epoch"] for r in rows] != list(range(1, cap+1))
                or not all(math.isfinite(r["loss"]) and r["loss"] >= 0 for r in rows)):
            raise ValueError("Incomplete or nonfinite epoch loss history")
    return keys, cap


def render(result, output, continuation_epoch=None, audit_path=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter

    keys, cap = validate(result, continuation_epoch)
    output = Path(output)
    paths = [output / f"{n}.{ext}" for n in NAMES for ext in ("png", "svg")]
    paths.append(output / "plot_manifest.json")
    if any(p.exists() for p in paths):
        raise FileExistsError("Plot outputs already exist; use a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    ranges = [0] if continuation_epoch is None else [0, continuation_epoch]

    def decorate(ax, start):
        ax.set_xlim(start, cap)
        ax.set_xlabel("Completed epoch")
        ax.xaxis.set_major_locator(MaxNLocator(6, integer=True))
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.grid(alpha=.20, linewidth=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        if continuation_epoch is not None:
            ax.axvline(continuation_epoch, color="#6B7280", linestyle=":", linewidth=1.4,
                       label=f"Budget extension at epoch {continuation_epoch:,}")

    def save(fig, name):
        for ext in ("png", "svg"):
            fig.savefig(output / f"{name}.{ext}", dpi=180, facecolor="white")
        plt.close(fig)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none"}):
        for tier in ("full", "selection"):
            fig, axes = plt.subplots(len(ranges), 3, figsize=(15.5, 4.0*len(ranges)+.8), squeeze=False)
            for row_number, start in enumerate(ranges):
                for metric, ax in zip(METRICS, axes[row_number]):
                    for key in keys:
                        rows = sorted((r for r in result["curves"] if r["model"] == key
                                       and r["tier"] == tier and r["epoch"] >= start), key=lambda r:r["epoch"])
                        ax.plot([r["epoch"] for r in rows], [r[metric] for r in rows], label=key.upper(),
                                color=COLORS.get(key), linewidth=1.8, marker="o" if tier == "full" else None,
                                markersize=2.8)
                    if tier == "full" and start == 0:
                        ax.axhline(result["summary"][keys[0]]["bicubic"][metric], color="#9CA3AF",
                                   linestyle="--", linewidth=1.1, label="Bicubic")
                    decorate(ax, start)
                    ax.set_ylabel(LABELS[metric])
                    ax.set_title(metric.upper() if start == 0 else f"{metric.upper()} (Epochs {start:,}–{cap:,})",
                                 fontsize=11, loc="left", pad=10)
            title = "Validation Performance" if tier == "full" else "Checkpoint Selection Metrics"
            fig.suptitle(title, x=.065, ha="left", y=.975,
                         fontsize=16, fontweight="bold")
            handles, names = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, names, loc="upper left", bbox_to_anchor=(.06, .93), ncol=len(names),
                       frameon=False, fontsize=10)
            # Use the audited inventory, not a hard-coded dataset size.
            if tier == "full":
                summary = result["summary"][keys[0]]
                scope = f"All {summary['samples']:,} held-out frames from {summary['clips']} recordings"
                fig.text(.065, .025, f"{scope}. Equal-recording means; raw observations, no smoothing. Development CV, not an independent test.",
                         fontsize=9, color="#4B5563")
            fig.subplots_adjust(left=.065, right=.985, top=.84, bottom=.11, wspace=.30, hspace=.52)
            save(fig, f"{tier}_curves")

        fig, axes = plt.subplots(len(ranges), len(keys), figsize=(5.15*len(keys), 4.0*len(ranges)+.8), squeeze=False)
        for row_number, start in enumerate(ranges):
            for key, ax in zip(keys, axes[row_number]):
                folds = sorted({j["fold"] for j in result["jobs"] if j["model"] == key})
                for fold in folds:
                    rows = sorted((r for r in result["losses"] if r["model"] == key and r["fold"] == fold
                                   and r["epoch"] >= max(1, start)), key=lambda r:r["epoch"])
                    ax.plot([r["epoch"] for r in rows], [r["loss"] for r in rows], label=f"Fold {fold}",
                            linewidth=.85, alpha=.8)
                decorate(ax, start)
                ax.set_ylabel("Training L1 (frame-weighted epoch mean)")
                ax.set_title(key.upper() if start == 0 else f"{key.upper()} (Epochs {start:,}–{cap:,})",
                             fontsize=11, loc="left", pad=10)
        fig.suptitle("Training Loss", x=.065, ha="left", y=.975,
                     fontsize=16, fontweight="bold")
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper left", bbox_to_anchor=(.06, .93), ncol=len(names), frameon=False, fontsize=10)
        fig.text(.065, .025, "Raw epoch means, no smoothing. Each epoch uses fresh random aligned crops. This is training loss, not validation loss.",
                 fontsize=9, color="#4B5563")
        fig.subplots_adjust(left=.065, right=.985, top=.84, bottom=.11, wspace=.30, hspace=.52)
        save(fig, "loss_curves")

    manifest = dict(schema="sr_epoch_plots_v1", generator_sha256=sha256(Path(__file__)),
        source_audit_sha256=sha256(audit_path) if audit_path else None,
        epoch_ceiling=cap, continuation_epoch=continuation_epoch, smoothing=False,
        plotted_quality="current snapshots; not a running maximum or selected-model score",
        quality_aggregation="frame means within each recording, then equal recording weights",
        loss_aggregation="frame-weighted mean of observed training batch losses per epoch, separately per fold",
        plotted_artifacts_sha256={p.name: sha256(p) for p in paths if p.suffix != ".json"})
    (output / "plot_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--continuation-epoch", type=int)
    args = p.parse_args()
    result = json.loads(args.audit.read_text())
    print(json.dumps(render(result, args.output, args.continuation_epoch, args.audit), indent=2))


if __name__ == "__main__":
    main()
