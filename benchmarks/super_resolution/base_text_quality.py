"""Text-first diagnostic for base_sweep's center-timestamp reconstructions.

Scores OCR against source-extracted text or explicit clean-reference
transcriptions, never OCR-generated pseudo labels. Keeps case/digits/punctuation;
primary CER ignores whitespace.
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageOps

from benchmarks.super_resolution.metrics import metrics
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def reference_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        current = [i]
        for j, y in enumerate(b, 1):
            current.append(
                min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (x != y))
            )
        previous = current
    return previous[-1]


def distance(a, b):
    """Exact unit-cost Levenshtein, using bit vectors for long OCR blocks."""
    if len(a) > len(b):
        a, b = b, a
    m = len(a)
    if not m:
        return len(b)
    masks = {}
    for i, c in enumerate(a):
        masks[c] = masks.get(c, 0) | (1 << i)
    mask = (1 << m) - 1
    high = 1 << (m - 1)
    positive = mask
    negative = 0
    score = m
    for c in b:
        equal = masks.get(c, 0)
        vertical = equal | negative
        horizontal = (((equal & positive) + positive) ^ positive) | equal
        plus = negative | ~(horizontal | positive)
        minus = positive & horizontal
        score += int(bool(plus & high)) - int(bool(minus & high))
        plus = (plus << 1) | 1
        minus <<= 1
        positive = (minus | ~(vertical | plus)) & mask
        negative = (plus & vertical) & mask
    return score


def normalized(text):
    return "".join(c for c in text if not c.isspace())


def ocr(task):
    # Same RGB crop, fixed Lanczos x3 and 12px white border for every method.
    with Image.open(task["image"]) as image:
        x, y, w, h = task["xywh"]
        crop = image.convert("RGB").crop((x, y, x + w, y + h))
        if crop.size != (w, h):
            raise ValueError("Crop mismatch")
        roi_metrics = metrics(
            np.array(
                Image.open(task["reference"]).convert("RGB").crop((x, y, x + w, y + h))
            ),
            np.array(crop),
        )
        if not np.isfinite(roi_metrics["psnr"]):
            roi_metrics["psnr"] = None
        scaled = ImageOps.expand(
            crop.resize((w * 3, h * 3), Image.Resampling.LANCZOS),
            border=12,
            fill="white",
        )
        scaled.save(task["crop"])
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    pred = subprocess.check_output(
        [
            "tesseract",
            str(task["crop"]),
            "stdout",
            "-l",
            "eng",
            "--oem",
            "1",
            "--psm",
            "6",
            "--dpi",
            "300",
        ],
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    gt = task["text"]
    a = normalized(gt)
    b = normalized(pred)
    # Main score retains every non-whitespace character, including case and punctuation.
    error = distance(a, b)
    digits_a = "".join(c for c in a if c.isdigit())
    digits_b = "".join(c for c in b if c.isdigit())
    punct_a = "".join(c for c in a if not c.isalnum())
    punct_b = "".join(c for c in b if not c.isalnum())
    return dict(
        config=task["config"],
        method=task["method"],
        scene=task["scene"],
        region=task["id"],
        text_gt=gt,
        text_predicted=pred,
        characters=len(a),
        edit_distance=error,
        cer=error / len(a),
        exact_nonspace=a == b,
        words=len(gt.split()),
        word_errors=distance(gt.split(), pred.split()),
        digits=len(digits_a),
        digit_errors=distance(digits_a, digits_b),
        punctuation=len(punct_a),
        punctuation_errors=distance(punct_a, punct_b),
        crop_png=str(Path(task["crop"]).relative_to(task["root"])),
        crop_sha256=sha256(Path(task["crop"])),
        roi_quality=roi_metrics,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--ground-truth", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--time-s", type=int, default=2)
    p.add_argument("--image-index", type=Path)
    args = p.parse_args()
    root = args.run
    out = root / ("text" if args.time_s == 2 else f"text-{args.time_s}s")
    out.mkdir(exist_ok=False)
    truth = json.loads(args.ground_truth.read_text())
    m = json.loads((root / "manifest.json").read_text())
    r = json.loads((root / "results.json").read_text())
    assert r["complete"] and json.loads((root / "COMPLETED.json").read_text())[
        "results_sha256"
    ] == sha256(root / "results.json")
    assert truth["time_s"] == args.time_s
    image_index = (
        {} if args.image_index is None else json.loads(args.image_index.read_text())
    )
    if image_index:
        assert image_index["results_sha256"] == sha256(root / "results.json")
        image_index = {
            (x["config"], x["scene"], x["method"]): x["image"]
            for x in image_index["records"]
        }
        for row in r["records"]:
            if row["time_s"] == args.time_s and row["image"] is None:
                row["image"] = image_index[(row["config"], row["scene"], row["method"])]
    atomic_json(out / "ground-truth.json", truth)
    provenance = dict(
        started=time.time(),
        command=sys.argv,
        script_sha256=sha256(Path(__file__)),
        results_sha256=sha256(root / "results.json"),
        manifest_sha256=sha256(root / "manifest.json"),
        truth_sha256=sha256(args.ground_truth),
        time_s=args.time_s,
        image_index_sha256=None
        if args.image_index is None
        else sha256(args.image_index),
        tesseract=subprocess.check_output(
            ["tesseract", "--version"], stderr=subprocess.STDOUT, text=True
        ),
        tessdata_sha256=sha256(Path("/opt/homebrew/share/tessdata/eng.traineddata")),
        primary="Case-sensitive non-whitespace Unicode CER = Levenshtein edits / GT characters. Case, digits and punctuation retained. Indentation/whitespace correctness is not evaluated.",
        aggregation="Within each scene, pool character errors across all selected blocks; then average the selected scene CERs equally. Worst scene also reported.",
        scope=f"{truth.get('origin', 'Assistant-transcribed, visually checked clean reference text')}. {len(truth['regions'])} English blocks at {args.time_s}s, not human-annotated full videos. OCR is a proxy for readability and has its own errors; clean-reference floor is measured. No Chinese, cursor/input latency or temporal text stability assessment.",
        preprocessing="Native 1080p RGB crop -> Lanczos x3 -> 12px white border; identical for all methods. Tesseract eng, LSTM OEM1, PSM6, dpi300.",
        references={
            s["id"]: sha256(root / s["id"] / f"reference-{args.time_s}s.png")
            for s in m["scenes"]
        },
    )
    atomic_json(out / "environment.json", provenance)
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    tasks = []
    for region in truth["regions"]:
        ref = root / region["scene"] / f"reference-{args.time_s}s.png"
        for record in [
            dict(
                config="reference",
                method="reference",
                image={"path": str(ref.relative_to(root)), "sha256": sha256(ref)},
            )
        ] + [
            x
            for x in r["records"]
            if x["time_s"] == args.time_s and x["scene"] == region["scene"]
        ]:
            image = root / record["image"]["path"]
            assert sha256(image) == record["image"]["sha256"]
            folder = out / "crops" / record["config"] / record["method"]
            folder.mkdir(parents=True, exist_ok=True)
            tasks.append(
                dict(
                    region,
                    image=str(image),
                    reference=str(ref),
                    config=record["config"],
                    method=record["method"],
                    crop=str(folder / f"{region['scene']}-{region['id']}.png"),
                    root=str(out),
                )
            )
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, row in enumerate(executor.map(ocr, tasks), 1):
            rows.append(row)
            if i % 40 == 0:
                print(f"OCR {i}/{len(tasks)}", flush=True)
    assert len(rows) == len(tasks)
    atomic_json(out / "rows.json", rows)
    flat = [
        {k: v for k, v in row.items() if k != "roi_quality"}
        | {
            "roi_psnr": row["roi_quality"]["psnr"],
            "roi_ssim": row["roi_quality"]["ssim"],
        }
        for row in rows
    ]
    with (out / "rows.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["config"], row["method"])].append(row)
    summary = []
    for (config, method), values in groups.items():
        assert len(values) == len(truth["regions"])
        scenes = []
        for scene in [
            s
            for s in m["scenes"]
            if any(g["scene"] == s["id"] for g in truth["regions"])
        ]:
            sv = [x for x in values if x["scene"] == scene["id"]]
            scenes.append(
                dict(
                    scene=scene["id"],
                    cer=sum(x["edit_distance"] for x in sv)
                    / sum(x["characters"] for x in sv),
                    errors=sum(x["edit_distance"] for x in sv),
                    characters=sum(x["characters"] for x in sv),
                )
            )
        item = dict(
            config=config,
            method=method,
            scene_cer=scenes,
            cer=float(np.mean([x["cer"] for x in scenes])),
            worst_scene_cer=max(x["cer"] for x in scenes),
            exact_blocks=sum(x["exact_nonspace"] for x in values),
            roi_ssim=float(np.mean([x["roi_quality"]["ssim"] for x in values])),
            digits=sum(x["digits"] for x in values),
            digit_errors=sum(x["digit_errors"] for x in values),
            punctuation=sum(x["punctuation"] for x in values),
            punctuation_errors=sum(x["punctuation_errors"] for x in values),
        )
        if config != "reference":
            cfg = next(c for c in m["configs"] if c["id"] == config)
            runs = [x for x in m["runs"] if x["config"] == config]
            item.update(
                cfg,
                budget_pass=all(x["budget_pass"] for x in runs),
                max_actual_kbps=max(x["rate"]["retained_video_bps"] for x in runs)
                / 1000,
            )
        summary.append(item)
    reference = next(x for x in summary if x["method"] == "reference")
    for row in summary:
        if row["method"] == "reference":
            continue
        base = next(
            x
            for x in summary
            if x["config"] == row["config"] and x["method"] == "baseline"
        )
        row["cer_reduction_vs_baseline_pp"] = (base["cer"] - row["cer"]) * 100
        row["cer_above_clean_reference_pp"] = (row["cer"] - reference["cer"]) * 100
    winners = []
    for bps in (64000, 128000, 256000, 600000, 1000000):
        pool = [x for x in summary if x.get("budget_pass") and x["bps"] == bps]
        best = (
            min(pool, key=lambda x: (x["cer"], x["worst_scene_cer"], -x["roi_ssim"]))
            if pool
            else None
        )
        winners.append(dict(target_bps=bps, best=best))
    atomic_json(out / "summary.json", summary)
    atomic_json(
        out / "selection.json",
        dict(
            primary="minimum mean scene CER among all encoder-budget-eligible configurations; not a human readability threshold or live-network feasibility claim",
            reference=reference,
            winners=winners,
        ),
    )
    lines = [
        "# Text-first base-resolution PoC",
        "",
        "Text correctness is the primary selection criterion. Whole-image LPIPS/SSIM are secondary diagnostics.",
        "",
        provenance["scope"],
        "",
        provenance["primary"],
        provenance["aggregation"],
        "",
        f"Clean-reference OCR CER: {reference['cer'] * 100:.2f}% (this is OCR error, not video degradation).",
        "",
        "| Video target Kbps | Lowest text CER candidate | Actual max video Kbps | CER ↓ | Change vs same-base baseline |",
        "|---:|---|---:|---:|---:|",
    ]
    for winner in winners:
        x = winner["best"]
        if x is None:
            continue
        lines.append(
            f"| {winner['target_bps'] // 1000} | {x['height']}p {x['method']}, {x['chroma']}, {x['fps']}fps | {x['max_actual_kbps']:.1f} | {x['cer'] * 100:.2f}% | {x['cer_reduction_vs_baseline_pp']:+.2f} percentage points lower |"
        )
    lines += [
        "",
        "Passing the encoder gate only checks <=105% average video bitrate after 1s and zero VBV underflow warnings on every selected clip. "
        "RTP/IP/control/audio overhead and real-link congestion are excluded. Failed configurations stay in raw results but cannot win.",
        "",
        "Do not equate low OCR CER with perfect text or every user being able to read it. The English blocks and one timestamp are a small exploratory sample. "
        "Punctuation-only and digit-only edit counts are diagnostic projections; text has not been corrected by a language model. "
        "Case and punctuation are never normalized away. Fixed OCR preprocessing is applied identically to the reference, direct/bicubic and all enhanced methods.",
        "",
        "[Raw text and OCR predictions](rows.json) · [All method/configuration text scores](summary.json) · [Transcriptions and crop coordinates](ground-truth.json) · [Provenance](environment.json).",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    page = [
        '<!doctype html><html><meta charset="utf-8"><title>Text-first base sweep</title><style>body{font:15px system-ui;margin:28px}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccc}img{max-width:100%}pre{white-space:pre-wrap}</style><body><h1>Text first: characters, numbers and punctuation</h1>',
        f"<p>Clean-reference OCR CER {reference['cer'] * 100:.2f}%. Visually checked English blocks, one timestamp. Equal-scene character errors. No human readability threshold established.</p>",
        '<p><a href="REPORT.md">Protocol and winners</a> · <a href="summary.json">Every candidate</a> · <a href="rows.json">Every OCR output</a> · <a href="../index.html">Full quality gallery</a></p>',
        "<table><tr><th>Video Kbps</th><th>Candidate</th><th>CER</th><th>Reduction vs baseline</th></tr>",
    ]
    for winner in winners:
        x = winner["best"]
        if x:
            page.append(
                f"<tr><td>{winner['target_bps'] // 1000}</td><td>{x['height']}p {x['method']} / {x['chroma']} / {x['fps']}</td><td>{x['cer'] * 100:.2f}%</td><td>{x['cer_reduction_vs_baseline_pp']:+.2f} pp</td></tr>"
            )
    page.append("</table>")
    for winner in winners:
        x = winner["best"]
        if not x:
            continue
        page.append(
            f"<h2>{winner['target_bps'] // 1000} Kbps: {x['config']} / {x['method']}</h2>"
        )
        for row in [
            v for v in rows if v["config"] == x["config"] and v["method"] == x["method"]
        ]:
            clean = next(
                v
                for v in rows
                if v["config"] == "reference"
                and v["scene"] == row["scene"]
                and v["region"] == row["region"]
            )
            page.append(
                f'<h3>{row["scene"]} / {row["region"]} — CER {row["cer"] * 100:.1f}%</h3><p>Reference:</p><img src="{clean["crop_png"]}"><p>Candidate:</p><img src="{row["crop_png"]}"><details><summary>Exact ground truth and OCR output</summary><pre>GT: {html.escape(row["text_gt"])}\nOCR: {html.escape(row["text_predicted"])}</pre></details>'
            )
    page.append("</body></html>")
    (out / "index.html").write_text("\n".join(page))
    atomic_json(
        out / "COMPLETED.json",
        dict(
            rows=len(rows),
            finished=time.time(),
            summary_sha256=sha256(out / "summary.json"),
            rows_sha256=sha256(out / "rows.json"),
            script_sha256=sha256(Path(__file__)),
        ),
    )
    print(
        json.dumps(dict(reference_cer=reference["cer"], winners=winners), indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
