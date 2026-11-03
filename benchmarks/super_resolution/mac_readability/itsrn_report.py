"""Audit official screen-content SR against the frozen Mac development baseline."""

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.mac_readability.quality import save, sha, summarize


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    a = p.parse_args()
    r = a.root.resolve()
    source = a.source.resolve()
    freeze = read(r / "COMPARATOR-FROZEN.json")
    candidate = read(r / "itsrn-quality-v1/ROWS.json")
    protocol = read(r / "itsrn-quality-v1/PROTOCOL.json")
    frozen_ocr = read(r / "dense-clean-v1/PROTOCOL.json")
    assert protocol["complete"]
    for k in ["tesseract", "tessdata", "preprocessing"]:
        assert protocol[k] == frozen_ocr[k]
    assert len(candidate) == 2852
    datasets = [
        (
            "Official EDSR",
            [
                x
                for x in read(r / "dense-model-quality-v1/ROWS.json")
                if x["method"] == "edsr-official"
            ],
        ),
        ("Best trained EDSR", read(r / "training-v2-selected-full-quality/ROWS.json")),
        ("Official ITSRN", candidate),
    ]
    base = {
        (x["state_id"], x["label_id"]): x for x in datasets[0][1] if x["unit"] == "line"
    }
    table = []
    for name, rows in datasets:
        lines = [x for x in rows if x["unit"] == "line"]
        assert len(lines) == 1046
        summary = summarize(lines)[0]
        groups = {x["group"]: x["cer"] for x in summary["groups"]}
        hard = float(np.mean([groups[g] for g in freeze["hard_groups"]]))
        table.append(
            dict(
                name=name,
                hard_cer=hard,
                relative_reduction=1 - hard / freeze["hard_macro_cer"],
                lines=summary,
                cells=summarize([x for x in rows if x["unit"] == "cell"])[0],
                categories={
                    c: summarize([x for x in lines if x["category"] == c])[0]
                    for c in ["code", "chinese", "table", "english"]
                },
                new_wrong_lines=[
                    x
                    for x in lines
                    if base[x["state_id"], x["label_id"]]["exact_nonspace"]
                    and not x["exact_nonspace"]
                ],
                repaired_lines=[
                    x
                    for x in lines
                    if not base[x["state_id"], x["label_id"]]["exact_nonspace"]
                    and x["exact_nonspace"]
                ],
                zero_error_group_edits=sum(
                    x["edits"]
                    for x in lines
                    if x["group"] in freeze["zero_error_groups"]
                ),
            )
        )
    delta = np.array(
        [
            next(x["cer"] for x in table[0]["lines"]["groups"] if x["group"] == g)
            - next(x["cer"] for x in table[2]["lines"]["groups"] if x["group"] == g)
            for g in freeze["hard_groups"]
        ]
    )
    draws = np.random.default_rng(20261102).integers(0, len(delta), (10000, len(delta)))
    manifest = read(r / "itsrn-official-v1/RESULTS.json")
    smoke = read(r / "itsrn-smoke-v1/RESULTS.json")
    assert manifest["complete"] and len(manifest["images"]) == 32 and smoke["complete"]
    timings = np.array([x["timing"]["total_ms"] for x in manifest["images"][1:]])
    amp = np.array([x["total_ms"] for x in smoke["timings"]])
    timing = dict(
        fp32_warm_samples=len(timings),
        fp32_warm_p50_ms=float(np.median(timings)),
        fp32_warm_p95_ms=float(np.quantile(timings, 0.95)),
        fp32_cold_ms=manifest["images"][0]["timing"]["total_ms"],
        amp_warm_samples=len(amp),
        amp_warm_p50_ms=float(np.median(amp)),
        amp_warm_p95_ms=float(np.quantile(amp, 0.95)),
        scope=manifest["timing_scope"],
        host=manifest["host"],
        gpu=manifest["gpu"],
    )
    states = {x["state_id"]: x for x in read(source / "manifest.json")["states"]}
    indexes = [
        read(r / "dense-models-v1/IMAGES-LOCAL.json"),
        read(r / "TRAINING-V2-SELECTED-IMAGES.json"),
        read(r / "itsrn-official-v1/IMAGES-LOCAL.json"),
    ]
    images = [
        {
            x["state_id"]: x
            for x in indexes[0]["images"]
            if x["method"] == "edsr-official"
        },
        {x["state_id"]: x for x in indexes[1]["images"]},
        {x["state_id"]: x for x in indexes[2]["images"]},
    ]
    # Identical frozen source-defined text masks; no OCR-dependent crop selection.
    roi = []
    for sid, state in states.items():
        ref = np.asarray(
            Image.open(source / state["reference"]["path"]).convert("RGB")
        ).astype(np.float32)
        mask = np.zeros(ref.shape[:2], bool)
        for label in state["labels"]:
            x, y, w, h = label["xywh"]
            mask[y : y + h, x : x + w] = True
        for (name, _), index in zip(datasets, images):
            item = index[sid]
            assert sha(item["path"]) == item["sha256"]
            out = np.asarray(Image.open(item["path"]).convert("RGB")).astype(np.float32)
            mse = float(np.square(ref[mask] - out[mask]).mean())
            roi.append(
                dict(
                    method=name,
                    state_id=sid,
                    group=state["group"],
                    mse=mse,
                    psnr=10 * np.log10(255**2 / mse),
                )
            )
    for t in table:
        values = [x for x in roi if x["method"] == t["name"]]
        t["text_roi_psnr"] = float(
            np.mean(
                [
                    np.mean([x["psnr"] for x in values if x["group"] == g])
                    for g in {x["group"] for x in values}
                ]
            )
        )
    output = dict(
        schema="screen_sr_comparison_v1",
        complete=True,
        script_sha256=sha(Path(__file__)),
        comparator_sha256=sha(r / "COMPARATOR-FROZEN.json"),
        plan_sha256=sha(r / "SCREEN-SR-PLAN-V1.json"),
        rows_sha256=sha(r / "itsrn-quality-v1/ROWS.json"),
        table=table,
        timing=timing,
        text_roi_rows=roi,
        paired_hard_gain_ci95=np.quantile(
            delta[draws].mean(1), [0.025, 0.975]
        ).tolist(),
        hard_target_cer=freeze["target_hard_cer"],
        quality_target_met=table[2]["hard_cer"] <= freeze["target_hard_cer"],
        latency_target_met=False,
        production_selected=False,
        test_used=False,
        limitations=[
            "Official ITSRN, not ITSRN++; no Mac-codec fine-tuning",
            "Source-age 2.8s offline replay; no network or display timing claim",
            "FP32 quality, AMP only numerical and warm-timing diagnostic",
            "OCR CER is not human preference; indentation excluded from nonspace exactness",
            "Repeatedly used development set, not final held-out test",
        ],
    )
    save(r / "SCREEN-SR-RESULTS.json", output)
    # Fixed illustrations: first three source lines in code, table and Chinese.
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    examples = []
    for sid in [1, 9, 25]:
        state = states[sid]
        box = (36, 78, 600, 200)
        canvas = Image.new("RGB", (1144, 304), "#e9edf2")
        draw = ImageDraw.Draw(canvas)
        paths = [source / state["reference"]["path"]] + [
            Path(index[sid]["path"]) for index in images
        ]
        for i, (label, path) in enumerate(
            zip(
                ["Source", "Official EDSR", "Best trained EDSR", "Official ITSRN"],
                paths,
            )
        ):
            x = (i % 2) * 572
            y = (i // 2) * 152
            draw.text((x + 4, y + 3), label, font=font, fill="black")
            canvas.paste(Image.open(path).convert("RGB").crop(box), (x + 4, y + 26))
        path = r / f"screen-sr-{state['category']}.png"
        canvas.save(path)
        examples.append(path.name)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    labels = [x["name"] for x in table]
    values = [x["hard_cer"] * 100 for x in table]
    axes[0].bar(labels, values, color=["#6b7280", "#2563eb", "#d97706"])
    axes[0].axhline(
        freeze["target_hard_cer"] * 100,
        color="green",
        ls="--",
        label="25% reduction target",
    )
    axes[0].set_ylabel("Hard-document macro CER (%)")
    axes[0].legend()
    cats = ["code", "chinese", "table", "english"]
    positions = np.arange(4)
    for i, t in enumerate(table):
        axes[1].bar(
            positions + (i - 1) * 0.25,
            [t["categories"][c]["macro_cer"] * 100 for c in cats],
            width=0.25,
            label=t["name"],
        )
    axes[1].set_xticks(positions, cats)
    axes[1].set_ylabel("Category macro CER (%)")
    axes[1].legend()
    fig.savefig(r / "screen-sr-curves.png", dpi=170)
    fig.savefig(r / "screen-sr-curves.svg")
    plt.close(fig)
    rows = "".join(
        f"<tr><td>{html.escape(t['name'])}</td><td>{t['hard_cer'] * 100:.3f}%</td><td>{t['relative_reduction'] * 100:.2f}%</td><td>{t['lines']['exact_units']}/1046</td><td>{t['cells']['exact_units']}/1806</td><td>{t['text_roi_psnr']:.2f}</td></tr>"
        for t in table
    )
    (r / "SCREEN-SR-REPORT.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Official screen-content SR trial</title><style>body{font:17px system-ui;max-width:1200px;margin:40px auto;padding:20px;color:#172033}td,th{padding:12px;border-bottom:1px solid #ccc;text-align:left}img{max-width:100%}</style><h1>Official ITSRN on real Mac low-bitrate inputs</h1><p>32 frames, 16 development documents, frozen 15 hard groups. Mac VideoToolbox 720p / 600 Kbps request / 4:2:0. Official pretrained ITSRN directly reconstructs 1080p in FP32.</p><table><tr><th>Method</th><th>Hard CER</th><th>CER reduction vs official EDSR</th><th>Exact lines</th><th>Exact table cells</th><th>Text ROI PSNR (dB)</th></tr>'
        + rows
        + '</table><img src="screen-sr-curves.png"><p>'
        + f"RTX4090 warm FP32 tensor-pipeline median: {timing['fp32_warm_p50_ms']:.1f} ms; p95: {timing['fp32_warm_p95_ms']:.1f} ms (31 frames). Warm AMP median: {timing['amp_warm_p50_ms']:.1f} ms (5 repeats). Exceeds the 200 ms refinement budget before network/display. This original implementation is not selected for production."
        + "</p><h2>Pixel-preserving fixed crops</h2><p>First three source lines of the first code, table and Chinese documents; fixed selection, no OCR-result selection. View at native image size.</p>"
        + "".join('<img src="' + x + '">' for x in examples)
        + '<p>Final test untouched. OCR errors, including recognition failures on readable text, are retained. No claim of human preference or true constrained-link end-to-end performance.</p><p><a href="SCREEN-SR-RESULTS.json">Full metrics, regressions and evidence hashes</a></p>'
    )
    for t in table:
        print(
            t["name"],
            "hard CER",
            t["hard_cer"],
            "cells",
            t["cells"]["macro_cer"],
            "ROI PSNR",
            t["text_roi_psnr"],
        )
    print(timing)


if __name__ == "__main__":
    main()
