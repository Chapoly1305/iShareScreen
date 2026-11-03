"""Report causal resolution controls, measured payload fairness, and regressions."""

import argparse
import csv
import html
import json
import math
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.mac_readability.quality import save, sha, summarize


def read(path):
    return json.loads(path.read_text())


def one(rows, config, method, age):
    return [
        r
        for r in rows
        if r["config"] == config
        and r["method"] == method
        and r["source_age_s"] == age
        and r["unit"] == "line"
    ]


def paired(a, b):
    x = {r["group"]: r["cer"] for r in a["groups"]}
    y = {r["group"]: r["cer"] for r in b["groups"]}
    assert x.keys() == y.keys()
    groups = sorted(x)
    d = np.array([y[g] - x[g] for g in groups])
    draw = np.random.default_rng(20261102).integers(0, len(d), (10000, len(d)))
    return dict(
        scope="Positive values favor first method; document-group paired bootstrap,10000 draws,seed20261102",
        mean_difference=float(d.mean()),
        difference_ci95=np.quantile(d[draw].mean(axis=1), [0.025, 0.975]).tolist(),
        first_wins=int((d > 0).sum()),
        second_wins=int((d < 0).sum()),
        ties=int((d == 0).sum()),
        groups=[
            dict(group=g, first_cer=x[g], second_cer=y[g], difference=y[g] - x[g])
            for g in groups
        ],
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    source = read(a.source / "manifest.json")
    states = {s["state_id"]: s for s in source["states"]}
    datasets = [
        "dense-clean-v1",
        "dense-native-quality-v1",
        "dense-model-quality-v1",
        "precodec-quality-v1",
        "precodec-model-quality-v1",
        "rate-match-quality-v1",
        "temporal-quality-v1",
    ]
    all_rows = {n: read(root / n / "ROWS.json") for n in datasets}
    stages = [
        ("Clean source", "dense-clean-v1", "clean", "reference", None),
        (
            "720p RGB resize, no codec",
            "precodec-quality-v1",
            "720p-rgb-resize",
            "bicubic",
            None,
        ),
        (
            "720p +420, no codec",
            "precodec-quality-v1",
            "720p-420-no-codec",
            "bicubic",
            None,
        ),
        ("720p HEVC, bicubic", "dense-native-quality-v1", "720p-600k", "bicubic", 2.8),
        (
            "720p +420, no codec, EDSR",
            "precodec-model-quality-v1",
            "720p-420-no-codec",
            "edsr-official",
            None,
        ),
        (
            "720p HEVC +EDSR",
            "dense-model-quality-v1",
            "720p-600k",
            "edsr-official",
            2.8,
        ),
        (
            "1080p +420, no codec",
            "precodec-quality-v1",
            "1080p-420-no-codec",
            "native",
            None,
        ),
        (
            "1080p HEVC, 600k request",
            "dense-native-quality-v1",
            "1080p-600k",
            "native",
            2.8,
        ),
        (
            "1080p HEVC, matched payload",
            "rate-match-quality-v1",
            "1080p-matched-payload",
            "native",
            2.8,
        ),
    ]
    table = []
    for name, dataset, config, method, age in stages:
        rows = one(all_rows[dataset], config, method, age)
        s = summarize(rows)
        assert len(s) == 1 and s[0]["units"] == 1046
        table.append(
            dict(
                name=name,
                dataset=dataset,
                summary=s[0],
                fonts={
                    str(f): summarize([r for r in rows if r["font_pixels"] == f])[0]
                    for f in [14, 18]
                },
                categories={
                    c: summarize([r for r in rows if r["category"] == c])[0]
                    for c in ["code", "table", "english", "chinese"]
                },
                blank_ocr_lines=sum(not r["text_predicted"].strip() for r in rows),
            )
        )
    by_name = {r["name"]: r for r in table}
    baseline = by_name["720p HEVC +EDSR"]["summary"]
    matched = by_name["1080p HEVC, matched payload"]["summary"]
    comparison = paired(baseline, matched)
    comparison["relative_cer_reduction"] = (
        1 - baseline["macro_cer"] / matched["macro_cer"]
    )
    matched_manifest = read(root / "rate-match-v1/MANIFEST.json")
    rates = []
    for group in matched_manifest["groups"]:
        run = next(r for r in group["attempts"] if r["id"] == group["selected_id"])
        assert group["matched"]
        rates.append(
            dict(
                group=group["group"],
                rate720=group["target_kbps"],
                rate1080=run["actual_kbps"],
                relative_error=run["relative_rate_error"],
                requested1080=run["requested_kbps"],
                frames1080=int(run["probe"]["streams"][0]["nb_read_frames"]),
                attempts=len(group["attempts"]),
            )
        )
    codec = read(a.source.parent / "codec-v1/MANIFEST.json")
    original_rates = {
        str(h): float(
            np.mean(
                [
                    sum(int(p["size"]) for p in r["probe"]["packets"]) * 8 / 6000
                    for r in codec["runs"]
                    if r["height"] == h
                ]
            )
        )
        for h in [720, 1080]
    }
    temporal = summarize(
        [r for r in all_rows["temporal-quality-v1"] if r["unit"] == "line"]
    )
    temporal += [
        by_name[n]["summary"]
        for n in ["720p HEVC, bicubic", "1080p HEVC, 600k request"]
    ]
    cells = []
    for folder in [
        "dense-clean-v1",
        "dense-native-quality-v1",
        "dense-model-quality-v1",
        "rate-match-full-quality-v1",
    ]:
        for row in read(root / folder / "SUMMARY.json"):
            if row["unit"] == "cell" and (
                folder != "dense-native-quality-v1" or row["method"] != "reference"
            ):
                cells.append(row)
    cell720 = next(x for x in cells if x["method"] == "edsr-official")
    cell1080 = next(x for x in cells if x["config"] == "1080p-matched-payload")
    # A paired line transition audit prevents interpreting small net CER changes
    # as an additive fraction of physical information loss.
    transitions = []
    for label, left, right in [
        ("RGB resize vs source", stages[0], stages[1]),
        ("Codec added after420", stages[2], stages[3]),
        ("EDSR after codec", stages[3], stages[5]),
    ]:
        l = one(all_rows[left[1]], *left[2:])
        r = one(all_rows[right[1]], *right[2:])
        lm = {(x["state_id"], x["label_id"]): x for x in l}
        delta = [x["edits"] - lm[x["state_id"], x["label_id"]]["edits"] for x in r]
        transitions.append(
            dict(
                label=label,
                lines_worse=sum(x > 0 for x in delta),
                lines_better=sum(x < 0 for x in delta),
                unchanged=sum(x == 0 for x in delta),
                net_edits=sum(delta),
            )
        )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    plt.rcParams.update({"font.size": 10})
    names = [
        "Clean source",
        "720p RGB resize, no codec",
        "720p +420, no codec",
        "720p HEVC, bicubic",
        "720p HEVC +EDSR",
    ]
    axes[0].barh(
        ["Source", "Resize only", "Resize +420", "+ HEVC", "+ EDSR"],
        [by_name[n]["summary"]["macro_cer"] * 100 for n in names],
        color=["#61758b", "#ca7838", "#ba843a", "#a95340", "#247c87"],
    )
    axes[0].invert_yaxis()
    axes[0].set(
        xlabel="Document-macro character error (%)",
        title="Separate the processing losses",
    )
    for config, method, label, color in [
        ("720p-600k", "bicubic", "720p bicubic", "#a95340"),
        ("1080p-600k", "native", "1080p native", "#446aab"),
    ]:
        rows = sorted(
            [r for r in temporal if r["config"] == config and r["method"] == method],
            key=lambda r: r["source_age_s"],
        )
        axes[1].plot(
            [r["source_age_s"] for r in rows],
            [r["macro_cer"] * 100 for r in rows],
            marker="o",
            label=label,
            color=color,
        )
    axes[1].set(
        xlabel="Media PTS after source change (s)",
        ylabel="Character error (%)",
        title="Settled quality differs from early quality",
    )
    axes[1].legend(fontsize=8)
    for rate, name, label, color in [
        (original_rates["720"], "720p HEVC, bicubic", "720p bicubic", "#a95340"),
        (original_rates["720"], "720p HEVC +EDSR", "720p EDSR", "#247c87"),
        (
            original_rates["1080"],
            "1080p HEVC, 600k request",
            "1080p, 600 Kbps request",
            "#446aab",
        ),
        (
            float(np.mean([r["rate1080"] for r in rates])),
            "1080p HEVC, matched payload",
            "1080p, matched",
            "#9560a9",
        ),
    ]:
        y = by_name[name]["summary"]["macro_cer"] * 100
        axes[2].scatter([rate], [y], s=75, label=label, color=color)
    axes[2].set(
        xlabel="Measured mean video payload (Kbps)",
        ylabel="Character error (%)",
        title="Same requested rate is not same payload",
    )
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(root / "resolution-causal-curves.png", dpi=170)
    fig.savefig(root / "resolution-causal-curves.svg")
    plt.close(fig)
    # Select illustrative groups by disclosed measured advantage/regression.
    ordered = sorted(comparison["groups"], key=lambda r: r["difference"])
    pictures = []
    old_images = read(root / "dense-codec-quality-v1/IMAGES.json")["images"]
    old_models = read(root / "dense-models-v1/IMAGES-LOCAL.json")["images"]
    pre_images = read(root / "precodec-images-v1/IMAGES.json")["images"]
    match_images = read(root / "rate-match-images-v1/IMAGES.json")["images"]
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 16)
    for kind, group in [
        ("advantage", ordered[-1]["group"]),
        ("regression", ordered[0]["group"]),
    ]:
        state = next(
            s for s in source["states"] if s["group"] == group and s["font_points"] == 7
        )
        sid = state["state_id"]
        base_rows = one(
            all_rows["dense-model-quality-v1"], "720p-600k", "edsr-official", 2.8
        )
        matched_rows = one(
            all_rows["rate-match-quality-v1"], "1080p-matched-payload", "native", 2.8
        )
        bm = {r["label_id"]: r for r in base_rows if r["state_id"] == sid}
        mr = [r for r in matched_rows if r["state_id"] == sid]
        chosen = max(
            mr,
            key=lambda r: (
                (r["edits"] - bm[r["label_id"]]["edits"])
                * (1 if kind == "advantage" else -1)
            ),
        )
        label = next(l for l in state["labels"] if l["label_id"] == chosen["label_id"])
        x, y, _, _ = label["xywh"]
        x = max(0, min(x - 4, 1920 - 700))
        y = max(0, min(y - 6, 1080 - 110))
        box = (x, y, x + 700, y + 110)

        def pick(rows, config, method):
            return next(
                r["path"]
                for r in rows
                if r["state_id"] == sid
                and r["config"] == config
                and r["method"] == method
            )

        panels = [
            ("Clean source", str(a.source / state["reference"]["path"])),
            ("720p resize only", pick(pre_images, "720p-rgb-resize", "bicubic")),
            ("720p HEVC + bicubic", pick(old_images, "720p-600k", "bicubic")),
            (
                "720p HEVC + official EDSR",
                pick(old_models, "720p-600k", "edsr-official"),
            ),
            (
                "1080p HEVC, matched payload",
                pick(match_images, "1080p-matched-payload", "native"),
            ),
            (
                "1080p HEVC, higher actual payload",
                pick(old_images, "1080p-600k", "native"),
            ),
        ]
        canvas = Image.new("RGB", (1424, 3 * 150 + 12), "#edf1f5")
        draw = ImageDraw.Draw(canvas)
        for i, (title, path) in enumerate(panels):
            xx = 8 + (i % 2) * 708
            yy = 6 + (i // 2) * 150
            draw.text((xx, yy), title, font=font, fill="#17283d")
            canvas.paste(Image.open(path).convert("RGB").crop(box), (xx, yy + 28))
        name = f"resolution-{kind}.png"
        canvas.save(root / name)
        pictures.append(
            dict(
                path=name,
                group=group,
                state_id=sid,
                box=list(box),
                selection="Group with largest macroCER advantage/regression;14px variant; largest paired line error difference; native pixels, no sharpening",
                label_id=chosen["label_id"],
            )
        )
    training = read(root / "training-ocr-v1/CURVES.json")
    assert training["complete"]
    best = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    for arm in sorted({r["arm"] for r in training["checkpoints"]}):
        values = sorted(
            [r for r in training["checkpoints"] if r["arm"] == arm and r["step"] > 0],
            key=lambda r: r["step"],
        )
        chosen = min(values, key=lambda r: r["hard_macro_cer"])
        best.append(chosen)
        axes[0, 0].plot(
            [r["step"] for r in values],
            [r["hard_macro_cer"] * 100 for r in values],
            marker="o",
            label=arm,
        )
        folder = root / "training-v1/models" / arm
        with (folder / "training-curve.csv").open() as f:
            history = list(csv.DictReader(f))
        axes[0, 1].plot(
            [int(r["step"]) for r in history],
            [float(r["loss"]) for r in history],
            label=arm,
        )
        done = read(folder / "COMPLETED.json")
        assert done["updates"] == 6000 and done["skipped_updates"] == 0
        vals = done["checkpoints"]
        axes[1, 0].plot(
            [r["step"] for r in vals],
            [r["macro_psnr"] for r in vals],
            marker="o",
            label=arm,
        )
        axes[1, 1].plot(
            [r["step"] for r in vals],
            [r["performance"]["modes"]["amp_fp16"]["p95_ms"] for r in vals],
            marker="o",
            label=arm,
        )
    axes[0, 0].axhline(
        training["comparator"]["hard_macro_cer"] * 100,
        color="black",
        ls="--",
        label="Official EDSR",
    )
    axes[0, 0].axhline(
        training["comparator"]["target_hard_cer"] * 100,
        color="green",
        ls=":",
        label="25% reduction target",
    )
    axes[0, 0].set(
        title="Hard15-group CER: lower is better", ylabel="Character error (%)"
    )
    axes[0, 1].set(title="Training loss", ylabel="Loss (different objectives)")
    axes[1, 0].set(
        title="Full-frame PSNR can rise while CER worsens", ylabel="Mean PSNR (dB)"
    )
    axes[1, 1].set(
        title="Same lab RTX4090, AMP forward only", ylabel="p95 forward (ms)"
    )
    for ax in axes.flat:
        ax.set_xlabel("Successful optimizer updates")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.savefig(root / "training-v1-curves.png", dpi=170)
    fig.savefig(root / "training-v1-curves.svg")
    plt.close(fig)
    outcome = dict(
        schema="mac_edsr_ablation_outcome_v1",
        complete=True,
        comparator_sha256=sha(root / "COMPARATOR-FROZEN.json"),
        arms=4,
        updates_per_arm=6000,
        scored_checkpoints=25,
        best_per_arm=best,
        selected_new_model=None,
        target_met=False,
        decision="Retain official EDSR; no trained checkpoint improves frozen hard-set CER. Preserve all curves and checkpoints.",
        test_used=False,
    )
    save(root / "TRAINING-V1-OUTCOME.json", outcome)
    result = dict(
        schema="resolution_causal_report_v1",
        script_sha256=sha(Path(__file__)),
        source_sha256=sha(a.source / "manifest.json"),
        inputs={
            str(root / n / "ROWS.json"): sha(root / n / "ROWS.json") for n in datasets
        },
        scope="Actual M1 HEVC VideoToolbox420 controls on32 native Mac references in16 development document groups;1920x1080 display",
        table=table,
        matched_comparison=comparison,
        rates=rates,
        original_mean_kbps=original_rates,
        temporal=temporal,
        table_cells=cells,
        line_transitions=transitions,
        pictures=pictures,
        training_outcome=outcome,
        conclusions=[
            "At the original600k request, steady720p OCR loss is largely present before video compression.",
            "The original600k requests produced different actual mean video payloads, so their quality ordering was not an equal-rate comparison.",
            "At matched per-document mean payload, official EDSR720 improves aggregate CER relative to native1080; it loses on some groups and table-cell metrics.",
            "OCR results on early media frames differ markedly from2.8s settled samples. MediaPTS is not delivery latency.",
        ],
        limitations=[
            "Six-second elementary-video means; not wire bandwidth, packet pacing, physicalWAN or equal burst envelopes.",
            "Each1080 clip was matched within3% of its corresponding720 clip by rate only; all encoding attempts are retained.",
            "Real encoder frame drops retained; scored frames must exist at exactPTS and match both state markers.",
            "Fixed OCR produces errors and some blank outputs even on readable original images; no human-preference or physical-ceiling claim.",
            "420-only controls; the RGB diagnostic isolates preprocessing and is not a444 encoder comparison.",
            "Independent development groups only; final test untouched. Small group count and one Mac encoder limit generalization.",
            "The39% matched-rate result compares resolution pipelines, not a new model against the frozen official EDSR goal.",
        ],
    )
    save(root / "RESOLUTION-CAUSES.json", result)
    body = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Why resolution changes low-bandwidth readability</title><style>body{font:16px/1.55 system-ui;max-width:1150px;margin:36px auto;padding:0 20px;color:#17283d}table{border-collapse:collapse;width:100%;margin:18px 0}th,td{border-bottom:1px solid #d3dbe3;padding:9px;text-align:left}img{max-width:100%;height:auto}small{color:#47586a}.callout{background:#edf4f5;padding:16px;border-left:4px solid #247c87}</style>",
        "<h1>Why resolution changes low-bandwidth readability</h1>",
        '<p class="callout">Two effects compete: downsampling removes small text strokes, while encoding fewer pixels can preserve more useful detail per transmitted byte. The winning pipeline depends on actual bytes, content, reconstruction, and time since the source changed.</p>',
        "<p>Measured on32 native Mac reference images,16 document groups,14/18px fonts, actual M1 HEVC VideoToolbox,4:2:0. All results use the same1920×1080 display size and frozen OCR. Lower CER is better.</p>",
        "<h2>1. Separate resolution loss from codec loss</h2><table><tr><th>Processing stage</th><th>Macro CER</th><th>Exact lines</th><th>14px CER</th><th>18px CER</th></tr>",
    ]
    for row in table:
        s = row["summary"]
        body.append(
            f"<tr><td>{html.escape(row['name'])}</td><td>{s['macro_cer'] * 100:.3f}%</td><td>{s['exact_units']}/{s['units']}</td><td>{row['fonts']['14']['macro_cer'] * 100:.3f}%</td><td>{row['fonts']['18']['macro_cer'] * 100:.3f}%</td></tr>"
        )
    body += [
        "</table><p>At the original600Kbps request and2.8s source age, most720p bicubic error was already present in the uncompressed resize. A14px source font becomes approximately9.3px at720p;18px becomes12px. The sampling reduction is exact geometry; its observed OCR effect is measured above. Small positive/negative changes after conversion or compression are not additive amounts of physical information loss.</p>",
        "<h2>2. Correct the unequal actual-payload comparison</h2>",
        f"<p>Original requests were both600Kbps, but mean video payload was {original_rates['720']:.2f}Kbps for720p and {original_rates['1080']:.2f}Kbps for1080p. We re-encoded1080p separately for every document, selecting only by rate error, within3% of its corresponding720p clip. Matched1080p mean: {np.mean([r['rate1080'] for r in rates]):.2f}Kbps. All attempts are retained.</p>",
        f"<p><strong>At matched mean payload:720p +official EDSR CER {baseline['macro_cer'] * 100:.3f}% versus1080p native {matched['macro_cer'] * 100:.3f}%, a {comparison['relative_cer_reduction'] * 100:.1f}% relative reduction.</strong> Across16 documents: {comparison['first_wins']} wins, {comparison['second_wins']} losses, {comparison['ties']} tie. Paired document bootstrap95% interval for the absolute advantage: {comparison['difference_ci95'][0] * 100:.3f} to {comparison['difference_ci95'][1] * 100:.3f} percentage points. This is an average-payload comparison; bursts and wire overhead are not equalized.</p>",
        "<table><tr><th>Content</th><th>720p +official EDSR CER</th><th>1080p matched-payload CER</th></tr>",
    ]
    for c in ["code", "table", "english", "chinese"]:
        left = by_name["720p HEVC +EDSR"]["categories"][c]
        right = by_name["1080p HEVC, matched payload"]["categories"][c]
        body.append(
            f"<tr><td>{c.title()}</td><td>{left['macro_cer'] * 100:.3f}%</td><td>{right['macro_cer'] * 100:.3f}%</td></tr>"
        )
    body += [
        "</table>",
        f"<p><strong>Table exception:</strong> cell-level CER favors native1080p ({cell1080['macro_cer'] * 100:.3f}%) over720p +EDSR ({cell720['macro_cer'] * 100:.3f}%). Exact cells: {cell1080['exact_units']} versus {cell720['exact_units']}, both out of1806. A single aggregate cannot choose the best pipeline for every content type.</p>",
        "<h2>3. Static-frame sampling changes what the result means</h2><p>The original primary metric samples2.8s after a source change. The0.2s and1.0s controls below show that early frames behave differently. These timestamps are media positions, not wall-clock encoding, network, or presentation latency. A settled screenshot does not establish responsive readability.</p>",
        '<img src="resolution-causal-curves.png" alt="Resolution, temporal and actual-payload diagnostics">',
        "<h2>Pixel-preserving examples</h2><p>The examples deliberately show the largest measured group advantage and regression. Each is a disclosed14px crop at native pixels, without sharpening. Click to view the original-size artifact.</p>",
    ]
    for pic in pictures:
        body.append(
            f'<h3>{html.escape(pic["group"])}</h3><a href="{pic["path"]}"><img src="{pic["path"]}" alt="{html.escape(pic["group"])} comparison"></a>'
        )
    body += [
        "<h2>Training remains an independent question</h2><p>All four new EDSR ablations completed6000 updates and25 validation snapshots including initialization were scored. None improved the frozen hard-set official EDSR comparator; no new model is selected. PSNR/loss improvements do not establish improved text reading. The project quality goal remains unmet.</p>",
        '<img src="training-v1-curves.png" alt="Training, quality and GPU forward curves">',
        "<h2>Limits and reproducibility</h2><ul>",
    ]
    body += ["<li>" + html.escape(x) + "</li>" for x in result["limitations"]]
    body += [
        '</ul><p><a href="RESOLUTION-CAUSES.json">Raw report and evidence hashes</a> · <a href="rate-match-v1/MANIFEST.json">All matched-rate attempts</a> · <a href="TRAINING-V1-OUTCOME.json">Training outcome</a> · <a href="NETWORK-AND-DENSE.html">Earlier baseline and link report</a></p></html>'
    ]
    document = "\n".join(body)
    # Format prose only; never rewrite paths, attributes, or evidence numbers.
    parts = re.split("(<[^>]*>)", document)
    for i in range(0, len(parts), 2):
        text = re.sub(r"(?<=[A-Za-z:;,])(?=\d)", " ", parts[i])
        text = re.sub(r"(?<=\d)(?=Kbps|px|s\b)", " ", text)
        text = (
            text.replace("+EDSR", "+ EDSR")
            .replace("+official", "+ official")
            .replace("+420", "+ 420")
        )
        parts[i] = text
    (root / "RESOLUTION-CAUSES.html").write_text("".join(parts))


if __name__ == "__main__":
    main()
