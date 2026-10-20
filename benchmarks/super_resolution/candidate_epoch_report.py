"""Audit and summarize completed candidate true-epoch training trajectories.

The tool is deliberately read-only with respect to experiments: it consumes
their saved receipts and curve events and will fail instead of silently
averaging an incomplete or duplicate fold.  It does not rerun inference.
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from benchmarks.super_resolution import candidate_training, training_state


METRICS = ("psnr", "ssim", "lpips")


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_scores(scores, context):
    require(set(METRICS) <= set(scores), f"Missing quality metric: {context}")
    require(all(math.isfinite(scores[name]) for name in METRICS), f"Nonfinite quality metric: {context}")


def write_csv(path, rows):
    rows = list(rows)
    require(rows, f"No rows to write: {path.name}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mean_rows(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for identity, group in sorted(grouped.items()):
        require(len(group) == 5, f"Expected five folds for {identity}, found {len(group)}")
        result = dict(zip(keys, identity), folds=len(group))
        for metric in METRICS + ("loss",):
            if metric in group[0]:
                result[metric] = sum(row[metric] for row in group) / len(group)
        output.append(result)
    return output


def events(folder, protocol, done):
    schedule = protocol["schedule"]
    cap, updates = schedule["epochs"], schedule["updates_per_epoch"]
    require(cap == done["epochs"] == 1500, f"Incorrect epoch budget: {folder}")
    require(done["steps"] == schedule["total_updates"] == cap * updates, f"Incorrect update budget: {folder}")
    curve_rows = []
    losses = defaultdict(lambda: [0.0, 0])
    observed, baseline, update_count = {"selection": [], "full": []}, 0, 0
    with (folder / "progress.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            event = row["event"]
            if event == "baseline":
                baseline += 1
            elif event == "update":
                update_count += 1
                require(row["step"] == update_count and row["batch_frames"] > 0 and math.isfinite(row["loss"]),
                        f"Invalid update history: {folder}")
                total = losses[(update_count - 1) // updates + 1]
                total[0] += row["loss"] * row["batch_frames"]
                total[1] += row["batch_frames"]
            elif event in observed:
                observed[event].append(row)
    require(baseline == 1, f"Missing or duplicate baseline: {folder}")
    require(update_count == done["steps"], f"Missing update history: {folder}")
    for tier, period, field in (("selection", protocol["selection_every_epochs"], "selection_validation"),
                                ("full", protocol["full_every_epochs"], "full_validation")):
        expected = list(range(period, cap + 1, period))
        actual = observed[tier]
        require([row["completed_epoch"] for row in actual] == expected, f"Incomplete {tier} curve: {folder}")
        for row in actual:
            require(row["step"] == row["completed_epoch"] * updates, f"Incorrect {tier} step: {folder}")
            finite_scores(row[field], f"{folder}/{tier}/{row['completed_epoch']}")
            curve_rows.append(dict(tier=tier, epoch=row["completed_epoch"], **row[field]))
            require((folder / f"{tier}-epoch-{row['completed_epoch']:04d}.json").is_file(),
                    f"Missing saved {tier} metrics: {folder}")
    loss_rows = [dict(epoch=epoch, loss=total / frames) for epoch, (total, frames) in sorted(losses.items())]
    return curve_rows, loss_rows


def audit(root):
    expected = {(model, fold) for model in candidate_training.IDS for fold in range(5)}
    found = {}
    for receipt in root.rglob("COMPLETED.json"):
        done = read(receipt)
        identity = (done.get("model"), done.get("fold"))
        require(identity in expected, f"Unexpected completed run: {receipt}")
        if identity in found:
            raise ValueError(f"Duplicate authoritative run for {identity}: {receipt} and {found[identity]}")
        found[identity] = receipt.parent
    require(set(found) == expected, f"Missing completed candidate folds: {sorted(expected - set(found))}")
    curves, losses, final = [], [], []
    for (model, fold), folder in sorted(found.items()):
        done, protocol = read(folder / "COMPLETED.json"), read(folder / "protocol.json")
        require(protocol["schema"] == "sr_candidate_true_epoch_protocol_v1", f"Incorrect protocol: {folder}")
        require(done["model"] == model and done["fold"] == fold and not done["early_stopped"], f"Invalid receipt: {folder}")
        require(done["protocol_binding"] == training_state.fingerprint(protocol), f"Unbound receipt: {folder}")
        require((folder / "best.pt").is_file() and (folder / "last.pt").is_file(), f"Missing model artifact: {folder}")
        finite_scores(done["validation"], f"final {folder}")
        run_curves, run_losses = events(folder, protocol, done)
        curves.extend(dict(model=model, fold=fold, **row) for row in run_curves)
        losses.extend(dict(model=model, fold=fold, **row) for row in run_losses)
        final.append(dict(model=model, fold=fold, best_epoch=done["best_epoch"], best_step=done["best_step"], **done["validation"]))
    return curves, losses, final


def plot(rows, output, value, suffix):
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True)
    for axis, model in zip(axes.flat, candidate_training.IDS):
        series = [row for row in rows if row["model"] == model]
        for tier in sorted({row.get("tier", "training") for row in series}):
            points = [row for row in series if row.get("tier", "training") == tier]
            axis.plot([row["epoch"] for row in points], [row[value] for row in points], label=tier)
        axis.set_title(model); axis.grid(alpha=0.25); axis.legend()
    figure.supxlabel("Epoch"); figure.supylabel(value); figure.tight_layout()
    figure.savefig(output / suffix, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(args.run_root.is_dir(), f"Run root does not exist: {args.run_root}")
    require(not args.output.exists(), f"Refusing to overwrite report directory: {args.output}")
    curves, losses, final = audit(args.run_root)
    args.output.mkdir(parents=True)
    curve_means = mean_rows(curves, ("model", "tier", "epoch"))
    loss_means = mean_rows(losses, ("model", "epoch"))
    write_csv(args.output / "curve-folds.csv", curves)
    write_csv(args.output / "curves.csv", curve_means)
    write_csv(args.output / "loss-folds.csv", losses)
    write_csv(args.output / "losses.csv", loss_means)
    write_csv(args.output / "final-folds.csv", final)
    write_csv(args.output / "final.csv", mean_rows(final, ("model",)))
    (args.output / "report.json").write_text(json.dumps(dict(
        models=list(candidate_training.IDS), folds=5, epochs=1500,
        curves="current held-out validation snapshots; not running-best weights",
        final="selected-checkpoint development-CV scores; not an independent final test"), indent=2) + "\n")
    for metric in METRICS:
        plot(curve_means, args.output, metric, f"curves-{metric}.png")
    plot(loss_means, args.output, "loss", "loss-curves.png")


if __name__ == "__main__":
    main()
