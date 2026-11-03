"""Render labeled, nearest-neighbor validation crop comparisons at exact pixel scale."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.low_bandwidth_e2e.analyze_percent import load, tokens


SAMPLES = (
    ("percent-validation-status-00", 540, 600000, 60),
    ("percent-validation-quota-01", 540, 1000000, 60),
    ("percent-validation-billing-02", 720, 1000000, 60),
)


def run(root: Path, output: Path) -> None:
    root = root.resolve()
    methods = {
        "Direct resize": root / "baseline-validation",
        "Official QuickSRNet": root / "ocr/official-quick-validation",
        "Historical EDSR": root / "ocr/old-validation",
        "Generic 800 steps": root / "ocr/formal-generic-validation",
        "Percent 800 steps": root / "ocr/formal-symbol-validation",
    }
    rows = {name: load(path, 144) for name, path in methods.items()}
    protocol = json.loads((root / "EVAL_PROTOCOL.json").read_text())
    cases = {
        case["id"]: case for case in protocol["cases"] if case["split"] == "validation"
    }
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    smaller = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    scale, label_width, pad = 2, 290, 15
    panels = []
    evidence = []
    for key in SAMPLES:
        case = cases[key[0]]
        x, y, w, h = case["line"]["xywh"]
        height = 36 + 6 * (h * scale + 31) + pad
        panel = Image.new("RGB", (label_width + w * scale + 2 * pad, height), "#eef1f4")
        draw = ImageDraw.Draw(panel)
        truth = tokens(case["line"]["text"])
        draw.text(
            (pad, 7),
            f"{key[0]}  {key[1]}p  {key[2] // 1000} Kbps  truth: {truth}",
            fill="#111827",
            font=font,
        )
        reference_path = root / case["hr"]
        with Image.open(reference_path) as full:
            reference = full.convert("RGB").crop((x, y, x + w, y + h))
        display = [("Clean reference", reference, truth)]
        for name, folder in methods.items():
            row = rows[name][key]
            crop_path = root / row["native_crop"]
            with Image.open(crop_path) as source:
                pixels = source.convert("RGB").copy()
            if pixels.size != (w, h):
                raise ValueError(f"Changed crop geometry: {crop_path}")
            display.append((name, pixels, tokens(row["prediction"])))
            evidence.append(
                dict(
                    case=key[0],
                    base_height=key[1],
                    requested_bps=key[2],
                    frame=key[3],
                    method=name,
                    crop=str(crop_path.relative_to(root)),
                    crop_sha256=hashlib.sha256(crop_path.read_bytes()).hexdigest(),
                    numeric_percent_tokens=tokens(row["prediction"]),
                    ground_truth_tokens=truth,
                )
            )
        for index, (name, pixels, found) in enumerate(display):
            top = 36 + index * (h * scale + 31)
            verdict = "OK" if found == truth else "ERR"
            draw.text(
                (pad, top + 6),
                f"{name}  {verdict}",
                fill="#047857" if verdict == "OK" else "#b91c1c",
                font=smaller,
            )
            panel.paste(
                pixels.resize((w * scale, h * scale), Image.Resampling.NEAREST),
                (label_width + pad, top),
            )
            draw.text(
                (label_width + pad, top + h * scale + 2),
                str(found),
                fill="#475569",
                font=smaller,
            )
        panels.append(panel)
    gallery = Image.new(
        "RGB",
        (
            max(panel.width for panel in panels),
            sum(panel.height for panel in panels) + pad * (len(panels) - 1),
        ),
        "#ffffff",
    )
    top = 0
    for panel in panels:
        gallery.paste(panel, (0, top))
        top += panel.height + pad
    output.parent.mkdir(parents=True, exist_ok=True)
    gallery.save(output)
    (output.parent / "GALLERY.json").write_text(
        json.dumps(
            dict(
                schema="isharescreen_percent_validation_gallery_v1",
                image=output.name,
                image_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                scale=scale,
                method="nearest-neighbor inspection enlargement of unchanged FHD native crops",
                selection="Validation examples chosen after inspection to show failures and a success; not representative statistics.",
                rows=evidence,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
