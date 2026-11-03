"""Collect this transfer-learning comparison and score only eligible frozen OCR."""

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from benchmarks.super_resolution.text_corpus import remote
from benchmarks.super_resolution.text_corpus.collect_hopper import copy_tree
from benchmarks.super_resolution.text_corpus.evaluate import score, summarize
from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement.models import sha256


def read(path):
    return json.loads(path.read_text())


def remote_json(sftp, path):
    with sftp.file(path) as stream:
        return json.loads(stream.read())


def analyze(out):
    protocol = read(out / "data/evaluation-protocol/protocol.json")
    plan = read(out / "campaign.json")
    rowsets = {
        name: read(out / "data/evaluations" / name / "ocr-rows.json")
        for name in ["official-finetuned", "scratch-control"]
    }
    audits = {}
    baselines = {}
    policies = []
    for name, rows in rowsets.items():
        folder = out / "runs" / name
        p = read(folder / "protocol.json")
        done = read(folder / "COMPLETED.json")
        policies.append(p)
        progress = [
            json.loads(line)
            for line in (folder / "progress.jsonl").read_text().splitlines()
        ]
        assert [r["epoch"] for r in progress] == list(range(1, plan["epochs"] + 1))
        assert done["epochs"] == plan["epochs"] and done["steps"] == plan["epochs"] * (
            (p["training_views"] + p["batch"] - 1) // p["batch"]
        )
        assert (
            fingerprint(p) == done["protocol_binding"]
            and sha256(folder / "best.pt") == done["best_sha256"]
        )
        history = [(0, read(folder / "initial-validation.json")["loss"])] + [
            (r["epoch"], r["validation"]["loss"]) for r in progress
        ]
        selected = min(history, key=lambda item: item[1])
        assert (
            selected[0] == done["best_epoch"]
            and selected[1] == done["best_validation_loss"]
        )
        assert p["ocr_protocol_sha256"] == sha256(
            out / "data/evaluation-protocol/protocol.json"
        )
        assert done["initial_tensor_sha256"] != done["last_tensor_sha256"]
        rendered = read(out / "data/evaluations" / name / "rendered.json")
        assert rendered["training_policy_sha256"] == sha256(
            folder / "protocol.json"
        ) and rendered["training_receipt_sha256"] == sha256(folder / "COMPLETED.json")
        for r in rendered["records"]:
            assert (
                r["requested_bps"] in plan["allowed_bps"]
                and sha256(out / "data" / r["crop"]) == r["sha256"]
            )
        baselines[name] = {
            (r["id"], r["method"]): r["sha256"]
            for r in rendered["records"]
            if r["method"] in ["bilinear", "bicubic", "pretrained"]
        }
        audits[name] = dict(
            epochs=done["epochs"],
            steps=done["steps"],
            best_epoch=done["best_epoch"],
            initialization=p["initialization"],
            initial_tensor_sha256=done["initial_tensor_sha256"],
            initial_validation=done["initial_validation_loss"],
            best_validation=done["best_validation_loss"],
            checkpoint_sha256=done["best_sha256"],
            lr=p["lr"],
        )
    assert baselines["official-finetuned"] == baselines["scratch-control"]
    for key in [
        "training_indices_sha256",
        "validation_indices_sha256",
        "packed_sha256",
        "seed",
        "epochs",
        "batch",
        "patch",
        "text_weight",
        "edge_weight",
        "ocr_protocol_sha256",
    ]:
        assert policies[0][key] == policies[1][key]
    assert (
        audits["official-finetuned"]["initial_tensor_sha256"]
        == read(out / "official-state.json")["tensor_sha256"]
    )
    assert audits["official-finetuned"]["initialization"]["pretrained_weights_loaded"]
    assert not audits["scratch-control"]["initialization"]["pretrained_weights_loaded"]
    methods = {
        "bilinear": [
            r for r in rowsets["official-finetuned"] if r["method"] == "bilinear"
        ],
        "official": [
            r for r in rowsets["official-finetuned"] if r["method"] == "pretrained"
        ],
        "fine-tuned": [
            r for r in rowsets["official-finetuned"] if r["method"] == "trained"
        ],
        "scratch": [r for r in rowsets["scratch-control"] if r["method"] == "trained"],
        "reference": [
            r for r in rowsets["official-finetuned"] if r["method"] == "reference"
        ],
    }
    summaries = []
    for split in ["validation", "test"]:
        for rate in [None] + plan["allowed_bps"]:
            for method, rows in methods.items():
                # Reference records inherit their rate from the source case's predictions.
                rates = {r["id"]: r["requested_bps"] for r in methods["official"]}
                chosen = [
                    r
                    for r in rows
                    if r["split"] == split and (rate is None or rates[r["id"]] == rate)
                ]
                if chosen:
                    summaries.append(
                        dict(
                            split=split,
                            requested_bps=rate,
                            method=method,
                            **summarize(chosen),
                        )
                    )
    contrasts = []
    rng = np.random.default_rng(682)
    for split in ["validation", "test"]:
        for rate in [None] + plan["allowed_bps"]:
            a = {
                r["id"]: r
                for r in methods["fine-tuned"]
                if r["split"] == split and (rate is None or r["requested_bps"] == rate)
            }
            for comparator in ["official", "scratch"]:
                b = {r["id"]: r for r in methods[comparator] if r["id"] in a}
                groups = defaultdict(list)
                for identifier, r in a.items():
                    groups[r["source_group"]].append(r["cer"] - b[identifier]["cer"])
                means = np.array([np.mean(v) for v in groups.values()])
                boot = rng.choice(means, (5000, len(means)), replace=True).mean(1)
                contrasts.append(
                    dict(
                        split=split,
                        requested_bps=rate,
                        comparison="fine-tuned minus " + comparator,
                        cases=len(a),
                        source_groups=len(means),
                        mean_line_delta=float(
                            np.mean([r["cer"] - b[i]["cer"] for i, r in a.items()])
                        ),
                        equal_source_delta=float(means.mean()),
                        equal_source_bootstrap_95_percentile_ci=np.quantile(
                            boot, [0.025, 0.975]
                        ).tolist(),
                        replicates=5000,
                    )
                )
    atomic_json(out / "summary.json", summaries)
    atomic_json(out / "paired-contrasts.json", contrasts)
    atomic_json(
        out / "audit.json",
        dict(
            runs=audits,
            baseline_pixels_identical=True,
            matched_training_data=True,
            official_initial_tensors_identical=True,
            all_prediction_hashes_verified=True,
            ocr_counts=protocol["counts"],
            single_seed=True,
            learning_rates_differ=True,
            finished=time.time(),
        ),
    )
    lines = [
        "# Official QuickSRNet transfer experiment",
        "",
        "## Material Passport",
        "",
        "- Type: code experiment; status: completed and checkpoint/prediction identities verified.",
        "- Inputs: source-held-out text corpus; allowed bitrate subset; identical training/validation views and crop RNG for both runs.",
        "- Models: pinned official QuickSRNet-Small x2; the same weights after text training; random initialized matched control.",
        f"- Budget: seed 682, {plan['epochs']} epochs, batch32, LR patch96, FP32, TF32 off. Official fine-tuning LR 1e-4; scratch LR 1e-3; Adam cosine to 1% LR.",
        "- Selection: minimum fixed validation text/edge pixel proxy, including epoch0. Test OCR never selects the checkpoint.",
        "- Metrics: frozen source-derived English lines, unchanged Tesseract model/version and PSM7 preprocessing; lower CER is better.",
        "- Scope: single-seed offline exploratory comparison. Different LR recipes mean this is not a pure initialization ablation. No full-screen OCR, temporal or live network conclusion.",
        "",
        "| Test setting | Method | Lines | Mean line CER % | Equal-source CER % | Exact lines |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in summaries:
        if r["split"] == "test":
            lines.append(
                f"| {str(r['requested_bps'] // 1000) + ' Kbps' if r['requested_bps'] else 'All allowed rates'} | {r['method']} | {r['cases']} | {r['mean_line_cer'] * 100:.3f} | {r['equal_source_group_cer'] * 100:.3f} | {r['exact_lines']} |"
            )
    lines += ["", "## Checkpoint selection", ""]
    for name, a in audits.items():
        lines.append(
            f"- {name}: best epoch {a['best_epoch']}; {a['steps']} updates; validation proxy {a['initial_validation']:.6f} → {a['best_validation']:.6f}."
        )
    lines += [
        "",
        "[Audit](audit.json) · [All metrics](summary.json) · [Paired source bootstrap estimates](paired-contrasts.json) · [Fine-tuned predictions](data/evaluations/official-finetuned/index.html) · [Scratch predictions](data/evaluations/scratch-control/index.html)",
        "",
        "The earlier 300-epoch scratch campaign used a different rate mixture. It is retained as historical evidence and is not substituted for this matched 100-epoch control.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    # One common gallery uses every fixed test case, without selecting successes.
    page = [
        '<!doctype html><meta charset="utf-8"><title>Official vs transfer vs scratch</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}img{max-width:100%}pre{white-space:pre-wrap}details{margin:20px 0}</style><h1>Official / fine-tuned / matched scratch</h1><p><a href="REPORT.md">Results and scope</a> · <a href="summary.json">Metrics</a></p>'
    ]
    for case in protocol["cases"]:
        if case["split"] != "test":
            continue
        page.append(
            "<details><summary>"
            + html.escape(case["id"] + " / " + case["kind"])
            + "</summary><pre>GT: "
            + html.escape(case["text"])
            + "</pre>"
        )
        for method in ["reference", "bilinear", "official", "scratch", "fine-tuned"]:
            r = next(r for r in methods[method] if r["id"] == case["id"])
            page.append(
                f'<p>{method}: CER {r["cer"] * 100:.2f}%</p><img loading="lazy" src="data/{html.escape(r["crop"], quote=True)}"><pre>{html.escape(r["predicted"])}</pre>'
            )
        page.append("</details>")
    (out / "index.html").write_text("\n".join(page))
    atomic_json(
        out / "COMPLETED.json",
        dict(
            runs=list(rowsets),
            report_sha256=sha256(out / "REPORT.md"),
            summary_sha256=sha256(out / "summary.json"),
            audit_sha256=sha256(out / "audit.json"),
            finished=time.time(),
        ),
    )
    print(
        json.dumps(
            [
                r
                for r in summaries
                if r["split"] == "test" and r["requested_bps"] in (600000, 1000000)
            ],
            indent=2,
        ),
        flush=True,
    )


def run(args):
    root = args.root.resolve()
    out = root / "official-transfer-v1"
    out.mkdir(exist_ok=True)
    if (out / "COMPLETED.json").exists():
        raise ValueError("Already collected")
    started = time.time()
    completed = []
    client = remote.connect(args)
    sftp = client.open_sftp()

    def status(phase, **extra):
        atomic_json(
            out / "status.json",
            dict(phase=phase, completed=completed, updated=time.time(), **extra),
        )

    def wait():
        if time.time() - started > args.timeout_hours * 3600:
            raise TimeoutError("Collection deadline; training was not stopped")
        time.sleep(30)

    try:
        status("waiting_for_inputs")
        while True:
            try:
                plan = remote_json(sftp, args.remote_root + "/campaign.json")
                break
            except FileNotFoundError:
                wait()
        atomic_json(out / "campaign.json", plan)
        protocol = remote_json(
            sftp, args.remote_root + "/data/evaluation-protocol/protocol.json"
        )
        folder = out / "data/evaluation-protocol"
        if folder.exists() and not args.resume_prepared:
            raise ValueError("Prepared collection exists; explicit resume required")
        folder.mkdir(parents=True, exist_ok=args.resume_prepared)
        if (folder / "protocol.json").exists():
            assert read(folder / "protocol.json") == protocol
        else:
            atomic_json(folder / "protocol.json", protocol)
        assert sha256(folder / "protocol.json") == plan["ocr_protocol_sha256"]
        compatibility = remote_json(
            sftp,
            args.remote_root + "/data/evaluation-protocol/source-compatibility.json",
        )
        atomic_json(folder / "source-compatibility.json", compatibility)
        for case in protocol["cases"]:
            assert sha256(root / case["clean_crop"]) == case["clean_crop_sha256"]
            destination = out / "data" / case["clean_crop"]
            if destination.exists():
                assert sha256(destination) == case["clean_crop_sha256"]
            else:
                destination.symlink_to(root / case["clean_crop"])
        while len(completed) < len(plan["runs"]):
            for spec in plan["runs"]:
                name = spec["name"]
                if name in completed:
                    continue
                try:
                    exit = remote_json(
                        sftp, args.remote_root + "/logs/" + name + "-exit.json"
                    )
                except FileNotFoundError:
                    continue
                if exit["exit_code"]:
                    if (
                        args.resume_prepared
                        and name == "official-finetuned"
                        and exit["phase"] == "render"
                    ):
                        try:
                            exit = remote_json(
                                sftp,
                                args.remote_root
                                + "/logs/official-finetuned-render-recovered.json",
                            )
                        except FileNotFoundError:
                            continue
                    if exit["exit_code"]:
                        raise RuntimeError(
                            "Training/render failed; no automatic retry: "
                            + name
                            + " "
                            + str(exit)
                        )
                status("collecting", current=name)
                copy_tree(sftp, args.remote_root + "/runs/" + name, out / "runs" / name)
                copy_tree(
                    sftp,
                    args.remote_root + "/data/evaluations/" + name,
                    out / "data/evaluations" / name,
                )
                score(
                    SimpleNamespace(
                        root=out / "data",
                        output=out / "data/evaluations" / name,
                        workers=4,
                    )
                )
                completed.append(name)
            if len(completed) < len(plan["runs"]):
                status("waiting_for_training")
                wait()
        status("analyzing")
        analyze(out)
        status("completed")
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        sftp.close()
        client.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument(
        "--remote-root", default="/home/chen/isharescreen-official-transfer.20261021"
    )
    p.add_argument("--host", default="10.193.124.90")
    p.add_argument("--user", default="chen")
    p.add_argument("--key", type=Path, default=Path.home() / ".ssh/id_ed25519")
    p.add_argument("--known-hosts", type=Path, default=Path.home() / ".ssh/known_hosts")
    p.add_argument("--timeout-hours", type=float, default=6)
    p.add_argument(
        "--resume-prepared",
        action="store_true",
        help="Reuse a verified prepared collection after explicit environment-only render recovery; never resume training.",
    )
    run(p.parse_args())
