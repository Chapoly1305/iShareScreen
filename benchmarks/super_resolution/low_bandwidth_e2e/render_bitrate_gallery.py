"""Render source-bound, original-pixel text crops for the bitrate frontier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from isharescreen.enhancement.models import sha256


def read(path: Path):
    return json.loads(path.read_text())


def run(root: Path, split: str, case_id: str, output: Path) -> None:
    root = root.resolve()
    protocol = read(root / "EVAL_PROTOCOL.json")
    case = next(
        c for c in protocol["cases"] if c["id"] == case_id and c["split"] == split
    )
    base = read(root / ("baseline-" + split) / "ocr-rows.json")
    quick = read(root / "ocr-results" / ("official-" + split) / "ocr-rows.json")

    def keyed(rows):
        return {
            (r["id"], r["base_height"], r["requested_bps"], r["frame_index"]): r
            for r in rows
        }

    base, quick = keyed(base), keyed(quick)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 20)
        small = ImageFont.truetype("DejaVuSansMono.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
        small = font
    entries = []
    for bps in protocol["main_grid"]["requested_video_bps"]:
        for label, height, method in [
            ("720p direct", 720, base),
            ("720p QuickSRNet", 720, quick),
            ("1080p direct", 1080, base),
        ]:
            row = method[(case_id, height, bps, 60)]
            path = root / row["native_crop"]
            if sha256(path) != row["native_crop_sha256"]:
                raise ValueError("Gallery crop differs from scored original pixels")
            entries.append(
                (f"{bps / 1e6:g} Mbps  {label}", row, Image.open(path).convert("RGB"))
            )
    clean = Image.open(root / case["hr"]).convert("RGB")
    x, y, w, h = case["line"]["xywh"]
    clean = clean.crop((x, y, x + w, y + h))
    width = max(1250, 270 + max([clean.width] + [im.width for _, _, im in entries]) * 2)
    row_height = max(
        86, max([clean.height] + [im.height for _, _, im in entries]) * 2 + 30
    )
    canvas = Image.new("RGB", (width, 95 + row_height * (len(entries) + 1)), "#f5f6f8")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 10), f"{split.upper()}  {case_id}", fill="#182330", font=font)
    draw.text(
        (18, 42),
        "Crop pixels shown at 2x nearest-neighbor; same source line and frame 60.",
        fill="#465363",
        font=small,
    )

    def line(index: int, label: str, image: Image.Image, cer: float | None):
        top = 95 + index * row_height
        draw.rectangle((8, top, width - 8, top + row_height - 5), fill="#ffffff")
        draw.text((18, top + 11), label, fill="#182330", font=small)
        if cer is not None:
            draw.text((18, top + 38), f"OCR CER {cer:.3f}", fill="#465363", font=small)
        canvas.paste(
            image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST),
            (270, top + 9),
        )

    line(0, "Clean original", clean, None)
    for index, (label, row, image) in enumerate(entries, 1):
        line(index, label, image, row["cer"])
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    print("GALLERY", output, canvas.size)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.split, a.case, a.output)
