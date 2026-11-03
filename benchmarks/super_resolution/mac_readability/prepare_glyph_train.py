"""Prepare independently captured train glyph pages for the same Mac codec recipe."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.super_resolution.codescan_campaign.data import frame_at
from benchmarks.super_resolution.mac_readability.quality import sha, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--base-inputs", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "images").mkdir()
    (args.output / "targets").mkdir()
    definition = json.loads((args.source / "FROZEN.json").read_text())
    assert definition["role"] == "train_only" and definition["complete"]
    base = json.loads(args.base_inputs.read_text())
    rows = []
    source_manifests = {}
    for batch in range(4):
        reference = args.source / f"batch-{batch:02d}/reference"
        source = json.loads((reference / "manifest.json").read_text())
        assert (
            source["complete"]
            and source["role"] == "train_only_never_validation_or_test"
        )
        source_manifests[str(reference.relative_to(args.source))] = sha(
            reference / "manifest.json"
        )
        for state in source["states"]:
            original = reference / state["reference"]["path"]
            assert sha(original) == state["reference"]["sha256"]
            canvas = np.asarray(Image.open(original).convert("RGB")).copy()
            assert canvas.shape == (1080, 1920, 3)
            background = canvas[0, 0].copy()
            canvas[:48] = background
            canvas[1024:] = background
            # Markers are useful for capture validation but not training content.
            high = frame_at(canvas, background, 45)
            identifier = state["id"]
            low_path = args.output / "images" / (identifier + ".png")
            high_path = args.output / "targets" / (identifier + ".png")
            Image.fromarray(canvas).save(low_path)
            Image.fromarray(high).save(high_path)
            boxes = []
            for label in state["labels"]:
                x, y, w, h = label["xywh"]
                boxes.append([x, y - 16, x + w, y + h - 16])
            bounds = [
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            ]
            rows.append(
                dict(
                    id=identifier,
                    language="ChineseGlyph",
                    repository="synthetic/" + state["group"],
                    group_id=state["group"],
                    split="train",
                    source_sha256=sha(original),
                    source_text_sha256=state["origin"]["content_sha256"],
                    input_path=str(low_path.relative_to(args.output)),
                    input_sha256=sha(low_path),
                    input_pixel_sha256=hashlib.sha256(canvas.tobytes()).hexdigest(),
                    hr=str(high_path.relative_to(args.output)),
                    hr_sha256=sha(high_path),
                    hr_pixel_sha256=hashlib.sha256(high.tobytes()).hexdigest(),
                    background_rgb=background.tolist(),
                    font_pixels=state["font_points"] * 2,
                    transform=dict(
                        content_box=bounds,
                        line_boxes=boxes,
                        sample_shift_y=-16,
                        source_size=[1920, 1080],
                        marker_removal="Replace top48 and bottom56 pixels with source background before whole-frame encoding",
                    ),
                )
            )
    assert len(rows) == 128 and len({r["group_id"] for r in rows}) == 64
    for key in ["input_pixel_sha256", "hr_pixel_sha256"]:
        assert len({r[key] for r in rows}) == 128
    save(
        args.output / "INPUTS.json",
        dict(
            schema="mac_glyph_train_inputs_v1",
            role="train_only",
            complete=True,
            recipe=base["recipe"],
            runner_sha256=sha(Path(__file__)),
            definitions_freeze_sha256=sha(args.source / "FROZEN.json"),
            native_source_manifests=source_manifests,
            validation_or_test_used=False,
            counts=dict(samples=128, repositories=64, languages=1),
            rows=rows,
        ),
    )
    print(
        "Prepared128 original Mac glyph frames; marker bands removed; targets aligned to1.5s/shift16"
    )


if __name__ == "__main__":
    main()
