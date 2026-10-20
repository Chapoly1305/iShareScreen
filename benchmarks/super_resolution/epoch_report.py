"""Audit completed true-epoch trajectories and summarize matched development scores.

This reads saved metrics and checkpoints. It does not rerun inference, train a
model, or treat checkpoint-selection validation as an independent final test.
"""
import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

from benchmarks.super_resolution import epochs, train, training_state, train_epochs
from benchmarks.super_resolution.folds import validate
from isharescreen.enhancement.models import sha256

METRICS = ("psnr", "ssim", "lpips")
PHASES = ("bicubic", "pretrained", "best-validation")


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected):
    require(all(math.isclose(actual[m], expected[m], rel_tol=1e-10, abs_tol=1e-10) for m in METRICS),
            "Stale or incorrect metric summary")


def identities(rows):
    return [(r["capture"], r["frame"]) for r in rows]


def metric_result(path, expected):
    result = read(path)
    rows = result["samples"]
    ids = identities(rows)
    require(len(ids) == len(set(ids)) and set(ids) == expected,
            f"Missing, duplicate or incorrect held-out samples: {path.name}")
    require(all(math.isfinite(r[m]) for r in rows for m in METRICS), "Nonfinite quality metric")
    equal(result["summary"], train.summarize_samples(rows))
    return result


def expected_metrics():
    from benchmarks.super_resolution.video_perceptual import BACKBONE_SHA256, LINEAR_SHA256
    return dict(backbone_sha256=BACKBONE_SHA256, linear_sha256=LINEAR_SHA256)


def finite_state(state):
    import torch
    require(bool(state) and all(torch.isfinite(v).all().item() for v in state.values()), "Nonfinite or empty weights")


def audit(root, split, pairs, models=("edsr", "imdn", "safmn")):
    import torch
    torch.set_num_threads(1)
    plan, data = validate(read(split)), read(pairs)
    schedule = train.jobs(plan, list(models), list(range(plan["n_folds"])))
    lock = read(Path(__file__).with_name("models.lock.json"))
    expected_folders = {f"{j['model']}-fold-{j['fold']}" for j in schedule}
    actual_folders = {p.name for p in root.iterdir() if p.is_dir()}
    require(actual_folders == expected_folders, "Unexpected or missing trajectory directories; choose authoritative copies only")
    samples = {key: {phase: [] for phase in PHASES} for key in models}
    curves, clip_rows, frame_rows, job_rows, losses, evidence = {}, [], [], [], [], {}
    shared_policies = set()
    for job in schedule:
        key, fold = job["model"], job["fold"]
        folder = root / f"{key}-fold-{fold}"
        require(not (folder / "FAILED.json").exists(), "Failed trajectory cannot enter completed report")
        policy, done = read(folder / "protocol.json"), read(folder / "COMPLETED.json")
        require(policy["schema"] == "sr_true_epoch_protocol_v1", "Not a true-epoch protocol")
        args = SimpleNamespace(pairs=pairs, folds=split, model=key, fold=fold,
                               epochs=policy["schedule"]["epochs"], batch=policy["batch"])
        expected_schedule, _, _, training, validation = train_epochs.make_plan(args)
        require(policy["schedule"] == expected_schedule and policy["seed"] == plan["seed"] + fold,
                "Training/validation schedule or seed changed")
        selected = train.selection_indices(data["pairs"], validation, policy["selection_per_clip"])
        require(policy["selected_indices"] == selected, "Selection subset changed")
        require(train_epochs.recorded_sources_match(policy["sources"]), "Pinned training source changed")
        require(policy["pairs_sha256"] == sha256(pairs) and policy["folds_sha256"] == sha256(split), "Data identity changed")
        require(policy["official_checkpoint_sha256"] == lock[key]["sha256"]
                and policy["official_source_revision"] == lock[key]["revision"], "Incorrect official initialization identity")
        require(policy["metric_provenance"] == expected_metrics() and policy["validation_tf32"] is False,
                "Incorrect metric provenance or validation arithmetic")
        require(policy["drop_last"] is False and policy["early_stopping"] is False, "Epoch coverage policy changed")
        shared_policies.add(training_state.fingerprint({k: v for k, v in policy.items() if k not in
            {"schedule", "seed", "selected_indices", "official_checkpoint_sha256", "official_source_revision"}}))
        binding = training_state.fingerprint(policy)
        cap, total, per_epoch = (expected_schedule[k] for k in ("epochs", "total_updates", "updates_per_epoch"))
        require((done["model"], done["fold"], done["epochs"], done["steps"], done["protocol_binding"])
                == (key, fold, cap, total, binding), "Incomplete or incorrectly bound completion marker")
        require(done["early_stopped"] is False, "Unexpected early stop")
        for filename in ("best.pt", "last.pt", "snapshots/epoch-0000.pt"):
            require((folder / filename).is_file(), f"Missing checkpoint: {filename}")
        require(sha256(folder / "best.pt") == done["checkpoint_sha256"], "Selected checkpoint changed")
        best = torch.load(folder / "best.pt", map_location="cpu", weights_only=True)
        initial = torch.load(folder / "snapshots/epoch-0000.pt", map_location="cpu", weights_only=True)
        last = torch.load(folder / "last.pt", map_location="cpu", weights_only=True)
        for payload, epoch, step in ((initial, 0, 0), (best, done["best_epoch"], done["best_step"])):
            require((payload["model"], payload["fold"], payload["epoch"], payload["step"], payload["protocol_binding"])
                    == (key, fold, epoch, step, binding), "Checkpoint payload identity differs")
            require(payload["official_checkpoint_sha256"] == lock[key]["sha256"], "Checkpoint official identity differs")
            finite_state(payload["state_dict"])
        md = last["metadata"]
        require(last["binding"] == binding and md["epoch"] == cap and md["cursor"] == 0 and md["step"] == total,
                "Incomplete current recovery state")
        require(md["best_epoch"] == done["best_epoch"] and md["best_step"] == done["best_step"], "Best-state metadata differs")
        finite_state(last["state_dict"])
        require(last["best_state"].keys() == best["state_dict"].keys()
                and all(torch.equal(v, best["state_dict"][k]) for k, v in last["best_state"].items()),
                "Embedded selected weights differ from exported checkpoint")
        require(last["optimizer"]["state"] and all(int(v["step"]) == total for v in last["optimizer"]["state"].values()),
                "Adam state/update count differs")
        require(last["rng"] and last["torch_rng"].numel(), "Missing RNG state")
        expected = set(identities([data["pairs"][i] for i in validation]))
        subset = set(identities([data["pairs"][i] for i in selected]))
        require(done["full_validation_samples"] == len(expected) and done["selection_samples"] == len(subset),
                "Incorrect final validation counts")
        phases = {phase: metric_result(folder / f"{phase}.json", expected) for phase in PHASES}
        equal(done["validation"], phases["best-validation"]["summary"])
        baseline_subset = train.summarize_samples([r for r in phases["pretrained"]["samples"] if (r["capture"], r["frame"]) in subset])
        equal(read(folder / "selection-pretrained.json"), baseline_subset)
        history = [json.loads(line) for line in (folder / "progress.jsonl").read_text().splitlines()]
        updates = [r for r in history if r["event"] == "update"]
        require([r["step"] for r in updates] == list(range(1, total+1)), "Missing or duplicated training update")
        epoch_loss = {}
        for row in updates:
            epoch = (row["step"]-1)//per_epoch
            position = (row["step"]-1) % per_epoch
            served = min(policy["batch"], len(training)-position*policy["batch"])
            cursor = min((position+1)*policy["batch"], len(training))
            require((row["epoch"], row["cursor"], row["batch_frames"]) == (epoch, cursor, served)
                    and math.isfinite(row["loss"]) and row["loss"] >= 0, "Invalid epoch/crop exposure or loss")
            epoch_loss[epoch+1] = epoch_loss.get(epoch+1, 0.) + row["loss"]*served/len(training)
        losses.extend(dict(model=key, fold=fold, epoch=e, loss=v) for e, v in epoch_loss.items())
        baseline_events = [r for r in history if r["event"] == "baseline"]
        require(len(baseline_events) == 1, "Missing or duplicate epoch-zero baseline")
        equal(baseline_events[0]["selection_validation"], baseline_subset)
        equal(baseline_events[0]["full_validation"], phases["pretrained"]["summary"])
        selection_scores = [(0, baseline_subset["psnr"])]
        for tier, indices, first in (("selection", subset, [r for r in phases["pretrained"]["samples"] if (r["capture"], r["frame"]) in subset]),
                                     ("full", expected, phases["pretrained"]["samples"])):
            interval = policy["selection_every_epochs" if tier == "selection" else "full_every_epochs"]
            wanted_epochs = list(range(interval, cap+1, interval))
            if cap not in wanted_epochs:
                wanted_epochs.append(cap)
            events = [r for r in history if r["event"] == tier]
            require([r["completed_epoch"] for r in events] == wanted_epochs, "Incomplete or duplicated validation curve")
            curves.setdefault((key, tier, 0), []).extend(first)
            for event in events:
                e = event["completed_epoch"]
                require(event["step"] == e*per_epoch, "Validation epoch/update mapping differs")
                result = metric_result(folder / f"{tier}-epoch-{e:04d}.json", indices)
                equal(event[f"{tier}_validation"], result["summary"])
                curves.setdefault((key, tier, e), []).extend(result["samples"])
                if tier == "selection":
                    selection_scores.append((e, result["summary"]["psnr"]))
        best_epoch, best_psnr = max(selection_scores, key=lambda item: item[1])
        require(done["best_epoch"] == best_epoch and done["best_step"] == best_epoch*per_epoch
                and math.isclose(md["best_psnr"], best_psnr, abs_tol=1e-10), "Selected checkpoint does not follow fixed-subset PSNR rule")
        for phase, result in phases.items():
            samples[key][phase].extend(result["samples"])
        for capture in job["validation_captures"]:
            row = dict(model=key, fold=fold, capture=capture,
                       content=next(c.get("content", "unknown") for c in plan["clips"] if c["id"] == capture))
            for phase in PHASES:
                row[phase] = train.summarize_samples([r for r in phases[phase]["samples"] if r["capture"] == capture])
            row["delta_vs_pretrained"] = {m: row["best-validation"][m]-row["pretrained"][m] for m in METRICS}
            clip_rows.append(row)
        keyed = {phase: {(r["capture"], r["frame"]): r for r in phases[phase]["samples"]} for phase in PHASES}
        for capture, frame in sorted(expected):
            item = dict(model=key, fold=fold, capture=capture, frame=frame)
            for phase in PHASES:
                item.update({f"{phase}_{m}": keyed[phase][capture, frame][m] for m in METRICS})
            frame_rows.append(item)
        job_rows.append(dict(model=key, fold=fold, epochs=cap, steps=total, best_epoch=best_epoch,
                             best_step=done["best_step"], training_frames=len(training), validation_frames=len(expected),
                             attempts=md["attempt"], checkpoint_sha256=done["checkpoint_sha256"], **done["validation"]))
        evidence[folder.name] = {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob("*")) if p.is_file()}
    require(len(shared_policies) == 1, "Campaign mixes training or evaluation protocols")
    summary = {}
    for key, phases in samples.items():
        summary[key] = {phase: train.summarize_samples(rows) for phase, rows in phases.items()}
        require(len(set(identities(phases["best-validation"]))) == len(phases["best-validation"]), "Held-out frame counted across multiple folds")
        for reference in ("pretrained", "bicubic"):
            summary[key][f"delta_vs_{reference}"] = {m: summary[key]["best-validation"][m]-summary[key][reference][m] for m in METRICS}
        clips = [r for r in clip_rows if r["model"] == key]
        summary[key]["clip_wins_vs_pretrained"] = {m: sum(r["delta_vs_pretrained"][m] > 0 if m != "lpips" else r["delta_vs_pretrained"][m] < 0 for r in clips) for m in METRICS}
        frames = [r for r in frame_rows if r["model"] == key]
        summary[key]["frame_regressions_vs_pretrained"] = {
            m: sum((r[f"best-validation_{m}"]-r[f"pretrained_{m}"])*(1 if m != "lpips" else -1) < 0 for r in frames)
            for m in METRICS}
        summary[key].update(clips=len(clips), samples=len(phases["best-validation"]),
            lpips_reduction_percent=100*(1-summary[key]["best-validation"]["lpips"]/summary[key]["pretrained"]["lpips"]))
    # Bicubic may differ at FP32 LPIPS round-off across GPU types, never in coverage.
    control = {identity: r for identity, r in zip(identities(samples[models[0]]["bicubic"]), samples[models[0]]["bicubic"])}
    deviation = dict.fromkeys(METRICS, 0.)
    for key in models:
        for row in samples[key]["bicubic"]:
            reference = control[row["capture"], row["frame"]]
            for m in METRICS:
                error = abs(row[m]-reference[m])
                deviation[m] = max(deviation[m], error)
                require(error <= (1e-6 if m == "lpips" else 1e-8), "Matched bicubic control differs beyond numerical tolerance")
    curve_rows = [dict(model=key, tier=tier, epoch=e, clips=len({r["capture"] for r in rows}), samples=len(rows),
                       **train.summarize_samples(rows)) for (key, tier, e), rows in sorted(curves.items())]
    return dict(schema="sr_epoch_results_v1", summary=summary, jobs=job_rows, per_clip=clip_rows,
                per_frame=frame_rows, curves=curve_rows, losses=losses, evidence=evidence,
                complete_cv=True, folds_sha256=sha256(split), pairs_sha256=sha256(pairs),
                bicubic_max_absolute_deviation=deviation, report_generator_sha256=sha256(Path(__file__)),
                scope="Equal-clip development CV; validation used for checkpoint selection, not independent final testing")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(result, output):
    # Keep report-generated and standalone figures on the same maintained path.
    # Existing reports and their recorded generator hashes remain unchanged.
    from benchmarks.super_resolution.epoch_plots import render
    render(result, output, audit_path=output / "audit.json")


def write_report(result, output, plots=True):
    output.mkdir(parents=True, exist_ok=False)
    (output / "audit.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    write_csv(output / "jobs.csv", result["jobs"])
    write_csv(output / "per_frame.csv", result["per_frame"])
    write_csv(output / "curves.csv", result["curves"])
    write_csv(output / "losses.csv", result["losses"])
    clips = []
    for row in result["per_clip"]:
        item = {k: row[k] for k in ("model", "fold", "capture", "content")}
        for phase in (*PHASES, "delta_vs_pretrained"):
            item.update({f"{phase}_{m}": row[phase][m] for m in METRICS})
        clips.append(item)
    write_csv(output / "per_clip.csv", clips)
    regressions = [r for r in clips if r["delta_vs_pretrained_psnr"] < 0 or r["delta_vs_pretrained_ssim"] < 0 or r["delta_vs_pretrained_lpips"] > 0]
    if regressions:
        write_csv(output / "regressions.csv", regressions)
    frame_regressions = [r for r in result["per_frame"] if any(
        (r[f"best-validation_{m}"]-r[f"pretrained_{m}"])*(1 if m != "lpips" else -1) < 0 for m in METRICS)]
    if frame_regressions:
        write_csv(output / "frame_regressions.csv", frame_regressions)
    keys = list(result["summary"])
    lines = ["# Fine-tuning results", "", "## Matched development scores", "",
        f"All {len(result['jobs'])} trajectories completed their epoch budgets. Each model covers "
        f"{result['summary'][keys[0]]['samples']} paired held-out frames from {result['summary'][keys[0]]['clips']} captures.",
        "Each capture is averaged over its frames, then captures receive equal weight; folds and frame counts are not equally weighted.", "",
        "| Method | PSNR (dB) higher | SSIM higher | LPIPS lower |", "|---|---:|---:|---:|"]
    baseline = result["summary"][keys[0]]["bicubic"]
    lines.append(f"| Bicubic | {baseline['psnr']:.3f} | {baseline['ssim']:.5f} | {baseline['lpips']:.5f} |")
    for key, item in result["summary"].items():
        for phase, label in (("pretrained", "official pretrained"), ("best-validation", "fine-tuned, selected")):
            s = item[phase]
            lines.append(f"| {key.upper()} {label} | {s['psnr']:.3f} | {s['ssim']:.5f} | {s['lpips']:.5f} |")
    lines += ["", "## Paired changes from official pretrained", "",
        "| Model | PSNR delta (dB) | SSIM delta | LPIPS delta | LPIPS reduction | Clip wins: PSNR / SSIM / LPIPS |",
        "|---|---:|---:|---:|---:|---|"]
    for key, item in result["summary"].items():
        d, wins = item["delta_vs_pretrained"], item["clip_wins_vs_pretrained"]
        lines.append(f"| {key.upper()} | {d['psnr']:+.3f} | {d['ssim']:+.5f} | {d['lpips']:+.5f} | {item['lpips_reduction_percent']:.1f}% | "
                     f"{wins['psnr']} / {wins['ssim']} / {wins['lpips']} of {item['clips']} |")
    lines += ["", "LPIPS percentage reduction is the ratio of aggregate means, not an average of per-frame percentages.",
        f"{len(regressions)} model/capture pairs regress on at least one metric. See per_clip.csv and regressions.csv when present.",
        f"{len(frame_regressions)} model/frame pairs regress on at least one metric; these are retained in frame_regressions.csv when present.",
        "", "| Model | Frame regressions: PSNR / SSIM / LPIPS |", "|---|---|"]
    for key, item in result["summary"].items():
        counts = item["frame_regressions_vs_pretrained"]
        lines.append(f"| {key.upper()} | {counts['psnr']} / {counts['ssim']} / {counts['lpips']} of {item['samples']} |")
    lines += [
        "", "## Checkpoint selection", "",
        "Highest fixed-subset equal-clip PSNR, including epoch zero, selects each fold independently. Other metrics do not change that rule.",
        "The reported selected model is therefore a five-fold collection, not one all-data deployment model.", "",
        "| Model | Selected epochs in fold order |", "|---|---|"]
    for key in keys:
        epochs_used = [r["best_epoch"] for r in result["jobs"] if r["model"] == key]
        lines.append(f"| {key.upper()} | {', '.join(map(str, epochs_used))} |")
    lines += ["", "## Curves and evidence", "",
        "full_curves.png: full held-out validation at epoch zero and each full-validation checkpoint; these are CURRENT snapshots, not best-so-far weights.",
        "selection_curves.png: the smaller fixed selection subset. Its scores are not interchangeable with full-set scores.",
        "loss_curves.png: frame-weighted mean training-batch L1 per epoch. Model weights change within the epoch; this is not validation loss.",
        "The score table uses selected checkpoints, so it can differ from the full-curve endpoint at the final epoch.",
        "audit.json binds every copied artifact hash and records policy, coverage, recovery-state and selection-rule checks.",
        "jobs.csv, per_clip.csv, per_frame.csv, curves.csv and losses.csv retain unrounded values.", "",
        "## Interpretation limits", "",
        "These folds were used for checkpoint selection. This is development CV, not an independent final evaluation or a confidence claim about new sessions.",
        "The audit recomputes summaries from saved per-frame scores; it does not rerun inference or independently recompute PSNR/SSIM/LPIPS from pixels.",
        "No extra training, uniform-selection comparison, OCR, inference-latency benchmark or temporal-quality test was run.",
        "Do not compare this cleaned validation inventory directly with the historical 704-frame protocol.",
        "Hardware differs by trajectory, and SAFMN folds 2/3 migrated mid-run with state preserved; cross-device bitwise equality is not claimed.",
        "A fixed training ceiling does not establish convergence or prove that a longer budget has no benefit.",
        f"Maximum per-frame bicubic control discrepancies across model runs: {result['bicubic_max_absolute_deviation']}."]
    (output / "REPORT.md").write_text("\n".join(lines)+"\n")
    if plots:
        plot_curves(result, output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "folds", "pairs", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    args = p.parse_args()
    result = audit(args.run, args.folds, args.pairs)
    write_report(result, args.output)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
