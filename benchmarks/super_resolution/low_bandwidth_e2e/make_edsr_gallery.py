"""Build a fixed, native-pixel EDSR validation comparison gallery."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil

from benchmarks.super_resolution.training_state import atomic_json


SOURCES = {
    "Decoded baseline": "validation-baseline/ocr-rows.json",
    "QuickSRNet official": "ocr-results/official-validation/ocr-rows.json",
    "EDSR official": "edsr-followup-v1/ocr/official-validation/ocr-rows.json",
    "EDSR 1500 epoch": "edsr-followup-v1/ocr/finetuned-validation/ocr-rows.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict) -> tuple:
    return (
        row["id"],
        row.get("base_height", row.get("base", [None, None])[1]),
        row["requested_bps"],
        row["frame_index"],
    )


def run(root: Path) -> None:
    protocol = json.loads((root / "EVAL_PROTOCOL.json").read_text())
    cases = protocol["cases"]
    fixed = protocol["fixed_display_ids"]["validation"]
    first_paper = next(
        x["id"]
        for x in cases
        if x["split"] == "validation" and x["source_kind"] == "paper"
    )
    first_web = next(
        x["id"]
        for x in cases
        if x["split"] == "validation" and x["source_kind"] == "web"
    )
    # Choose source IDs without looking at OCR outcomes; both rates and bases are fixed.
    selections = [
        (fixed[1], 540, 600000, 90),
        (fixed[4], 720, 1000000, 90),
        (first_paper, 540, 600000, 90),
        (first_web, 720, 1000000, 90),
    ]
    columns = {
        name: {key(x): x for x in json.loads((root / rel).read_text())}
        for name, rel in SOURCES.items()
    }
    gallery = root / "edsr-followup-v1/gallery"
    assets = gallery / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    sections = []
    manifest = {
        "schema": "isharescreen_edsr_fixed_validation_gallery_v1",
        "selection_rule": "Protocol fixed validation code IDs 1 and 4, first paper and web validation source; fixed 540p/600k or 720p/1000k and frame 90; no result-based selection.",
        "sources": SOURCES,
        "samples": [],
    }
    for case_id, height, rate, frame in selections:
        sample_key = (case_id, height, rate, frame)
        truth = columns["Decoded baseline"][sample_key]["ground_truth"]
        cells = []
        sample = {"key": sample_key, "ground_truth": truth, "methods": {}}
        for name, rows in columns.items():
            row = rows[sample_key]
            if row["ground_truth"] != truth:
                raise ValueError(f"Ground truth changed: {sample_key}, {name}")
            src = root / row["native_crop"]
            if sha256(src) != row["native_crop_sha256"]:
                raise ValueError(f"Crop hash mismatch: {src}")
            dst = (
                assets
                / f"{case_id}-{height}p-{rate // 1000}k-f{frame}-{name.lower().replace(' ', '-')}.png"
            )
            shutil.copyfile(src, dst)
            digest = sha256(dst)
            sample["methods"][name] = {
                "crop": str(dst.relative_to(gallery)),
                "sha256": digest,
                "prediction": row["prediction"],
                "cer": row["cer"],
            }
            cells.append(
                f'<div class="cell"><h3>{html.escape(name)}</h3>'
                f'<a href="{html.escape(str(dst.relative_to(gallery)))}">'
                f'<img src="{html.escape(str(dst.relative_to(gallery)))}" alt="{html.escape(name)} raw crop"></a>'
                f"<p>CER {row['cer']:.3f} · 原始裁剪可点开</p>"
                f"<pre>{html.escape(row['prediction'] or '[OCR 空输出]')}</pre></div>"
            )
        manifest["samples"].append(sample)
        sections.append(
            f"<section><h2>{html.escape(case_id)} · {height}p · {rate // 1000} Kbps · frame {frame}</h2>"
            f"<p><b>真值：</b><code>{html.escape(truth)}</code></p>"
            f'<div class="grid">{"".join(cells)}</div></section>'
        )
    page = (
        '<!doctype html><html lang="zh"><meta charset="utf-8"><title>EDSR 文字裁剪对比</title>'
        "<style>body{font:16px system-ui;max-width:1500px;margin:auto;padding:24px;background:#f6f7f8;color:#17212a}"
        "section{background:#fff;padding:18px;margin:24px 0;border:1px solid #ddd;border-radius:8px}"
        ".grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.cell{min-width:0;border-top:1px solid #ddd}"
        "img{max-width:100%;image-rendering:pixelated;border:1px solid #aaa;background:white}pre{white-space:pre-wrap;overflow-wrap:anywhere}"
        "code{overflow-wrap:anywhere}@media(max-width:800px){.grid{grid-template-columns:1fr}}</style>"
        "<h1>EDSR 低带宽文字对比</h1><p>四个事先固定的验证样本；图像是 1080p 展示坐标中的原始像素裁剪，未再锐化或插值。"
        "OCR 仅用于量化，直接看图中的数字、标点和路径。这个页面不代表独立测试结论。</p>"
        + "".join(sections)
        + "</html>"
    )
    (gallery / "index.html").write_text(page)
    atomic_json(gallery / "GALLERY.json", manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root.resolve())
