"""Build a sampled visual comparison from hash-verified receiver screenshots.

This is a slideshow, not a continuous screen recording or a latency test.
Original PNG crops are also saved for judging text without video compression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


def verified_image(row: dict) -> Image.Image:
    path = Path(row["path"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row["sha256"]:
        raise ValueError(f"Screenshot hash differs: {path}")
    with Image.open(path) as image:
        return image.convert("RGB")


def build(root: Path, output: Path, font_path: Path) -> None:
    # Fixed category representatives; no selection by observed improvement.
    states = (1, 9, 17, 25)
    ages = (0.5, 1.0, 2.0, 2.8)
    crop = (36, 78, 936, 678)
    modes = ("bypass", "refresh")
    indexes = {}
    for mode in modes:
        path = root / "stream-runs" / f"confirm-{mode}-01" / "TIMED-IMAGES.json"
        index = json.loads(path.read_text())
        if not index["complete"]:
            raise ValueError(f"Incomplete screenshot index: {path}")
        indexes[mode] = {
            (row["state_id"], row["source_age_s"]): row for row in index["images"]
        }
    # Validate all selected evidence before creating any output.
    pictures = {
        (mode, state, age): verified_image(indexes[mode][state, age]).crop(crop)
        for mode in modes
        for state in states
        for age in ages
    }
    font = ImageFont.truetype(str(font_path), 20)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for number, (state, age) in enumerate((s, a) for s in states for a in ages):
        canvas = Image.new("RGB", (1840, 730), "#eeeeee")
        draw = ImageDraw.Draw(canvas)
        group = indexes["refresh"][state, age]["group"]
        draw.text(
            (20, 8),
            f"{group} | saved sample at receiver age {age:.1f}s",
            font=font,
            fill="black",
        )
        draw.text(
            (20, 36),
            "SAMPLED STILL COMPARISON - 2s per slide; NOT continuous playback",
            font=font,
            fill="black",
        )
        for column, mode in enumerate(modes):
            row = indexes[mode][state, age]
            title = (
                "Decoded video (no SR)"
                if mode == "bypass"
                else "SR + source detail (25% source blend)"
            )
            x = 10 + column * 920
            draw.text((x, 68), title, font=font, fill="black")
            canvas.paste(pictures[mode, state, age], (x, 100))
            records.append({"slide": number, "mode": mode, **row})
        draw.text(
            (20, 705),
            "Native-pixel crops. Open the PNGs at 100% for text quality; video adds compression.",
            font=font,
            fill="black",
        )
        canvas.save(output / f"slide-{number:02d}.png")
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-framerate",
        "1/2",
        "-i",
        str(output / "slide-%02d.png"),
        "-c:v",
        "libx264",
        "-crf",
        "10",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        "-movflags",
        "+faststart",
        str(output / "sampled-comparison.mp4"),
    ]
    subprocess.run(command, check=True)
    manifest = {
        "scope": __doc__,
        "states": states,
        "ages_s": ages,
        "crop_xyxy": crop,
        "seconds_per_slide": 2,
        "command": command,
        "evidence": records,
        "selection": "First 14px state of each category; no quality-based selection",
        "limitations": [
            "Separate sequential runs aligned by receiver source-announcement age",
            "Cannot assess flicker, continuous transitions or source-to-display delay",
            "Bypass is decoded video, not the trained EDSR quality comparator",
        ],
    }
    (output / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    slides = "\n".join(
        f'<figure><img src="slide-{i:02d}.png"><figcaption>Slide {i + 1}</figcaption></figure>'
        for i in range(16)
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Sampled visual review</title>'
        "<style>body{font:18px sans-serif;background:#eee}img{max-width:none}</style>"
        "<h1>Sampled still comparison, not continuous screen recording</h1>"
        "<p>Two seconds per slide does not represent original timing. "
        "Scroll horizontally to inspect native pixels; browser zoom 100%. "
        "Use continuous capture to judge flicker or transitions.</p>"
        '<video controls width="1280" src="sampled-comparison.mp4"></video>' + slides
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    )
    args = parser.parse_args()
    build(args.root, args.output, args.font)


if __name__ == "__main__":
    main()
