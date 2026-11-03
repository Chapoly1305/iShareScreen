"""Audit and analyze all six completed, frozen text-corpus experiments.

No checkpoint selection or training is performed. Rate slices contain different
source lines, and document bootstrap intervals condition on the three seeds.
"""

import argparse
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
import time

import numpy as np

from benchmarks.super_resolution.base_text_quality import distance, normalized
from benchmarks.super_resolution.text_corpus.evaluate import summarize
from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement.models import sha256


def read(path):
    return json.loads(path.read_text())


def paired_interval(delta, groups, seed=682, repetitions=20000):
    clusters = defaultdict(list)
    for i, group in enumerate(groups):
        clusters[group].append(i)
    members = [clusters[key] for key in sorted(clusters)]
    counts = np.array([len(indices) for indices in members])
    totals = np.array([np.asarray(delta)[indices].sum() for indices in members])
    means = totals / counts
    draws = np.random.default_rng(seed).integers(
        0, len(members), size=(repetitions, len(members))
    )
    line = totals[draws].sum(1) / counts[draws].sum(1) * 100
    group = means[draws].mean(1) * 100
    return dict(
        line_mean_delta_pp=float(np.mean(delta) * 100),
        line_cluster_percentile_95=np.quantile(line, [0.025, 0.975]).tolist(),
        equal_source_delta_pp=float(means.mean() * 100),
        equal_source_cluster_percentile_95=np.quantile(group, [0.025, 0.975]).tolist(),
        source_groups=len(members),
        repetitions=repetitions,
        seed=seed,
        scope="Exploratory paired source-cluster percentile bootstrap, conditional on these three training seeds; not a seed or population-wide confidence interval. No multiplicity-adjusted hypothesis test.",
    )


def run(args):
    root = args.root.resolve()
    out = root / "hopper-results/analysis-v1"
    out.mkdir(exist_ok=False)
    protocol = read(root / "evaluation-protocol/protocol.json")
    cases = {r["id"]: r for r in protocol["cases"]}
    test_ids = sorted(k for k, c in cases.items() if c["split"] == "test")
    groups = [cases[k]["source_group"] for k in test_ids]
    runs = []
    all_rows = {}
    baselines = {}
    initials = {}
    files_checked = 0
    provenance = []
    collection = read(root / "hopper-results/COMPLETED.json")
    assert (
        sha256(root / "hopper-results/all-run-scores.json")
        == collection["all_run_scores_sha256"]
    )
    for loss, text_weight, edge_weight in [("l1", 0.0, 0.0), ("text-edge", 2.0, 0.05)]:
        for seed in (682, 683, 684):
            name = f"{loss}-seed-{seed}"
            training = root / "hopper-results/runs" / name
            evaluation = root / "evaluations" / name
            policy = read(training / "protocol.json")
            done = read(training / "COMPLETED.json")
            seal = read(evaluation / "COMPLETED.json")
            rendered = read(evaluation / "rendered.json")
            assert done["protocol_binding"] == fingerprint(policy)
            assert done["epochs"] == 300 and done["steps"] == 75900
            assert (
                policy["seed"] == seed
                and policy["text_weight"] == text_weight
                and policy["edge_weight"] == edge_weight
            )
            assert (
                policy["validation_seed"] == 682
                and policy["validation_text_weight"] == 2.0
                and policy["validation_edge_weight"] == 0.05
            )
            assert (
                policy["initialization"]["kind"] == "from_scratch"
                and not policy["initialization"]["pretrained_weights_loaded"]
            )
            assert done["initial_tensor_sha256"] != done["last_tensor_sha256"]
            assert (
                sha256(training / "best.pt") == done["best_sha256"]
                and sha256(training / "last.pt") == done["last_sha256"]
            )
            assert policy["ocr_protocol_sha256"] == sha256(
                root / "evaluation-protocol/protocol.json"
            )
            assert rendered["training_policy_sha256"] == sha256(
                training / "protocol.json"
            ) and rendered["training_receipt_sha256"] == sha256(
                training / "COMPLETED.json"
            )
            assert rendered["ocr_protocol_sha256"] == sha256(
                root / "evaluation-protocol/protocol.json"
            )
            assert (
                rendered["identities"]["trained"]["checkpoint_sha256"]
                == done["best_sha256"]
            )
            assert rendered["identities"]["trained"]["epoch"] == done["best_epoch"]
            for field, file in [
                ("rows_sha256", "ocr-rows.json"),
                ("summary_sha256", "summary.json"),
                ("predictions_sha256", "rendered.json"),
            ]:
                assert sha256(evaluation / file) == seal[field]
            history = [
                json.loads(line)
                for line in (training / "progress.jsonl").read_text().splitlines()
                if line.strip()
            ]
            assert [r["epoch"] for r in history] == list(range(1, 301))
            best = min(
                [(0, read(training / "initial-validation.json")["loss"])]
                + [(r["epoch"], r["validation"]["loss"]) for r in history],
                key=lambda x: x[1],
            )
            assert (
                best[0] == done["best_epoch"]
                and best[1] == done["best_validation_loss"]
            )
            if seed in initials:
                assert initials[seed] == done["initial_tensor_sha256"]
            initials[seed] = done["initial_tensor_sha256"]
            rows = read(evaluation / "ocr-rows.json")
            assert len(rows) == 960
            for row in rows:
                case = cases[row["id"]]
                assert row["text_gt"] == case["text"]
                assert sha256(root / row["crop"]) == row["sha256"]
                files_checked += 1
                a, b = normalized(row["text_gt"]), normalized(row["predicted"])
                edits = distance(a, b)
                assert (
                    edits == row["edit_distance"]
                    and edits / len(a) == row["cer"]
                    and (a == b) == row["exact"]
                )
                if row["method"] in ["bilinear", "bicubic", "pretrained", "reference"]:
                    key = (row["method"], row["id"])
                    identity = (row["sha256"], row["predicted"], row["text_gt"])
                    if key in baselines:
                        assert baselines[key] == identity
                    baselines[key] = identity
            for summary in read(evaluation / "summary.json"):
                selected = [
                    r
                    for r in rows
                    if r["split"] == summary["split"]
                    and r["method"] == summary["method"]
                ]
                for key, value in summarize(selected).items():
                    assert np.isclose(value, summary[key])
            all_rows[name] = rows
            trained = [
                r for r in rows if r["split"] == "test" and r["method"] == "trained"
            ]
            runs.append(
                dict(
                    name=name,
                    loss=loss,
                    seed=seed,
                    best_epoch=done["best_epoch"],
                    best_validation_proxy=done["best_validation_loss"],
                    **summarize(trained),
                )
            )
            provenance.append(
                dict(
                    run=name,
                    policy_sha256=sha256(training / "protocol.json"),
                    training_receipt_sha256=sha256(training / "COMPLETED.json"),
                    ocr_receipt_sha256=sha256(evaluation / "COMPLETED.json"),
                    history_sha256=sha256(training / "progress.jsonl"),
                )
            )
            print("AUDITED", name, flush=True)
    lookup = {
        name: {
            r["id"]: r
            for r in rows
            if r["split"] == "test" and r["method"] == "trained"
        }
        for name, rows in all_rows.items()
    }
    baseline = {
        method: {
            r["id"]: r
            for r in all_rows["l1-seed-682"]
            if r["split"] == "test" and r["method"] == method
        }
        for method in ["bilinear", "bicubic", "pretrained", "reference"]
    }
    means = {
        loss: np.array(
            [
                [lookup[f"{loss}-seed-{seed}"][id]["cer"] for id in test_ids]
                for seed in (682, 683, 684)
            ]
        ).mean(0)
        for loss in ["l1", "text-edge"]
    }
    contrasts = {
        name: paired_interval(delta, groups)
        for name, delta in {
            "text-edge-minus-l1": means["text-edge"] - means["l1"],
            "text-edge-minus-bilinear": means["text-edge"]
            - np.array([baseline["bilinear"][id]["cer"] for id in test_ids]),
            "l1-minus-bilinear": means["l1"]
            - np.array([baseline["bilinear"][id]["cer"] for id in test_ids]),
        }.items()
    }
    slices = []
    for category in ["requested_bps", "kind"]:
        values = sorted({lookup["l1-seed-682"][id][category] for id in test_ids})
        for value in values:
            ids = [
                id for id in test_ids if lookup["l1-seed-682"][id][category] == value
            ]
            record = dict(
                category=category,
                value=value,
                lines=len(ids),
                source_groups=len({cases[id]["source_group"] for id in ids}),
                metrics={},
            )
            for method, rows in baseline.items():
                record["metrics"][method] = dict(
                    mean_line_cer_percent=np.mean([rows[id]["cer"] for id in ids]) * 100
                )
            for loss in ["l1", "text-edge"]:
                values = np.array(
                    [
                        np.mean(
                            [lookup[f"{loss}-seed-{seed}"][id]["cer"] for id in ids]
                        )
                        * 100
                        for seed in (682, 683, 684)
                    ]
                )
                record["metrics"][loss] = dict(
                    mean_line_cer_percent=float(values.mean()),
                    seed_sd=float(values.std(ddof=1)),
                )
            slices.append(record)
    atomic_json(out / "runs.json", runs)
    atomic_json(out / "slices.json", slices)
    atomic_json(out / "paired-contrasts.json", contrasts)
    atomic_json(
        out / "audit.json",
        dict(
            passed=True,
            runs=6,
            epochs_each=300,
            steps_each=75900,
            scored_crops_checked=files_checked,
            per_run_history_epochs_complete=True,
            validation_selected_checkpoints_verified=True,
            baseline_pixels_and_ocr_identical_across_runs=True,
            initial_tensors_matched_by_seed_across_losses=True,
            ocr_distances_recomputed=True,
            provenance=provenance,
            finished=time.time(),
        ),
    )
    if args.replay:
        from benchmarks.super_resolution.text_corpus.ocr_protocol import score_crop
        from concurrent.futures import ThreadPoolExecutor

        selected = sorted(
            cases, key=lambda id: hashlib.sha256(id.encode()).hexdigest()
        )[:16]
        tasks = [
            next(
                r
                for r in all_rows[
                    "l1-seed-682" if method != "text-edge" else "text-edge-seed-682"
                ]
                if r["id"] == id
                and r["method"]
                == ("trained" if method in ["l1", "text-edge"] else method)
            )
            for id in selected
            for method in ["bilinear", "l1", "text-edge"]
        ]

        def repeat(row):
            actual = score_crop(root / row["crop"], row["text_gt"])
            assert actual["predicted"] == row["predicted"]
            return dict(id=row["id"], crop=row["crop"], prediction_exact_match=True)

        with ThreadPoolExecutor(max_workers=4) as pool:
            replays = list(pool.map(repeat, tasks))
        atomic_json(
            out / "ocr-replay.json",
            dict(
                cases=16,
                scorings=len(replays),
                all_exact=True,
                selection="SHA256 order over all 160 frozen cases, independent of outcome",
                records=replays,
            ),
        )
    plot(out, slices)
    gallery(root, out, cases, all_rows)
    atomic_json(
        out / "COMPLETED.json",
        dict(
            audit_sha256=sha256(out / "audit.json"),
            runs_sha256=sha256(out / "runs.json"),
            slices_sha256=sha256(out / "slices.json"),
            contrasts_sha256=sha256(out / "paired-contrasts.json"),
            runner_sha256=sha256(Path(__file__)),
            finished=time.time(),
        ),
    )
    print(json.dumps(dict(runs=runs, contrasts=contrasts, slices=slices), indent=2))


def plot(out, slices):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in slices if r["category"] == "requested_bps"]
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for offset, method, color, label in [
        (-1.5, "bilinear", "#999999", "Bilinear"),
        (-0.5, "pretrained", "#c7b8ad", "Official pretrained"),
        (0.5, "l1", "#487fa7", "Scratch L1 (3 seeds)"),
        (1.5, "text-edge", "#197c6a", "Scratch text + edge (3 seeds)"),
    ]:
        ax.bar(
            x + offset * 0.2,
            [r["metrics"][method]["mean_line_cer_percent"] for r in rows],
            0.19,
            label=label,
            color=color,
            yerr=[r["metrics"][method].get("seed_sd", 0) for r in rows],
            capsize=2,
        )
    ax.set_xticks(x, [f"{r['value'] // 1000}\n(n={r['lines']})" for r in rows])
    ax.set_xlabel("Requested HEVC video bitrate (Kbps)")
    ax.set_ylabel("Mean line CER (%)")
    ax.set_title("Frozen test OCR: 96 English lines, 50 source groups")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    fig.text(
        0.06,
        0.025,
        "Different source lines at each rate: not a controlled rate sweep. Error bars are seed SD, not confidence intervals.",
        fontsize=8,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out / "rate-comparison.png", dpi=180)
    plt.close(fig)


def gallery(root, out, cases, all_rows):
    methods = [
        ("reference", "Clean reference", "l1-seed-682"),
        ("bilinear", "Bilinear", "l1-seed-682"),
        ("pretrained", "Official pretrained", "l1-seed-682"),
        ("trained", "Scratch L1, seed 682", "l1-seed-682"),
        ("trained", "Scratch text + edge, seed 682", "text-edge-seed-682"),
    ]
    rows = {
        name: {(r["id"], r["method"]): r for r in values}
        for name, values in all_rows.items()
    }
    page = [
        '<!doctype html><html><meta charset="utf-8"><title>Six-run text benchmark</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 18px}img{max-width:100%}pre{white-space:pre-wrap}details{margin:20px 0;padding:10px;border:1px solid #ddd}h3{font-size:16px}</style><h1>Same-source OCR comparisons</h1><p>All 96 test lines. Seed 682 fixed for both trained losses; no best-test seed selection. Clean OCR failures and model regressions are retained.</p>'
    ]
    for id in sorted(k for k, c in cases.items() if c["split"] == "test"):
        c = cases[id]
        row = rows["l1-seed-682"][(id, "trained")]
        page.append(
            f"<details><summary>{html.escape(c['kind'])} / {row['requested_bps'] // 1000} Kbps / {html.escape(id)}</summary><pre>GT: {html.escape(c['text'])}</pre>"
        )
        for method, label, name in methods:
            row = rows[name][(id, method)]
            path = html.escape(os.path.relpath(root / row["crop"], out), quote=True)
            page.append(
                f'<h3>{label}: CER {row["cer"] * 100:.2f}%</h3><img loading="lazy" src="{path}"><pre>{html.escape(row["predicted"])}</pre>'
            )
        page.append("</details>")
    page.append("</html>")
    (out / "index.html").write_text("\n".join(page))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--replay", action="store_true")
    run(p.parse_args())
