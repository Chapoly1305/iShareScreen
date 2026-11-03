"""Stage a balanced, repository-distinct Mac codec pilot from frozen train only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.codescan_campaign.data import render, frame_at
from benchmarks.super_resolution.mac_readability.quality import sha, save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    root = args.project
    frozen = json.loads(
        (root / "datasets/sources/codescan/split-v1/FROZEN.json").read_text()
    )
    rel = "datasets/train/codescan/manifest.json"
    assert sha(root / rel) == frozen["files_sha256"][rel]
    train = json.loads((root / rel).read_text())["rows"]
    selected = []
    repositories = set()
    for language in sorted({r["language"] for r in train}):
        pool = sorted(
            (r for r in train if r["language"] == language),
            key=lambda r: hashlib.sha256(("20261102:" + r["id"]).encode()).hexdigest(),
        )
        chosen = []
        for row in pool:
            if row["repository"] in repositories:
                continue
            chosen.append(row)
            repositories.add(row["repository"])
            if len(chosen) == 16:
                break
        assert len(chosen) == 16
        selected.extend(chosen)
    assert (
        len(selected) == 384
        and len(repositories) == 384
        and all(r["split"] == "train" for r in selected)
    )
    images = args.output / "images"
    images.mkdir()
    targets = args.output / "targets"
    targets.mkdir()

    def stage(row):
        canvas, bg, transform = render(root / "datasets/train/codescan/samples", row)
        target = frame_at(canvas, bg, 45)
        path = images / (row["id"] + ".png")
        high = targets / (row["id"] + ".png")
        Image.fromarray(canvas).save(path)
        Image.fromarray(target).save(high)
        return dict(
            id=row["id"],
            language=row["language"],
            repository=row["repository"],
            group_id=row["group_id"],
            source_sha256=row["files"]["img.png"],
            source_text_sha256=row["files"]["source.txt"],
            source_normalized_text_sha256=row["normalized_source_sha256"],
            split="train",
            input_path=str(path.relative_to(args.output)),
            input_sha256=sha(path),
            input_pixel_sha256=hashlib.sha256(canvas.tobytes()).hexdigest(),
            hr=str(high.relative_to(args.output)),
            hr_sha256=sha(high),
            hr_pixel_sha256=hashlib.sha256(target.tobytes()).hexdigest(),
            background_rgb=bg.tolist(),
            transform=transform,
        )

    rows = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for row in executor.map(stage, selected):
            rows.append(row)
    for key in [
        "source_sha256",
        "input_sha256",
        "input_pixel_sha256",
        "hr_sha256",
        "hr_pixel_sha256",
    ]:
        assert len({r[key] for r in rows}) == len(rows), key
    result = dict(
        schema="mac_train_pilot_inputs_v1",
        role="train_only",
        complete=True,
        source_freeze_sha256=sha(
            root / "datasets/sources/codescan/split-v1/FROZEN.json"
        ),
        train_manifest_sha256=sha(root / rel),
        runner_sha256=sha(Path(__file__)),
        render_runner_sha256=sha(
            root / "benchmarks/super_resolution/codescan_campaign/data.py"
        ),
        selection="SHA256 order with seed string20261102; 16 distinct repositories per each of24 languages; one sample per repository",
        originals_preserved=True,
        test_samples_read=False,
        validation_samples_used=False,
        recipe=dict(
            input_pixels=[1280, 720],
            reference_pixels=[1920, 1080],
            fps=30,
            frames=60,
            selected_pts_s=1.5,
            selected_input_frame=45,
            bitrate=600000,
            chroma="yuv420p",
            codec="Mac HEVC VideoToolbox hardware, real-time",
            color="BT709 limited range",
            motion="First30 frames static; frames30..59 shift upward by round(30*(n-30)/29) reference pixels",
            qualification="Require exact selected PTS; validate hashes and geometry; do not substitute a different timestamp if target frame is dropped",
        ),
        counts=dict(samples=len(rows), repositories=len(repositories), languages=24),
        rows=rows,
    )
    save(args.output / "INPUTS.json", result)
    print("Staged", len(rows), "isolated train sources")


if __name__ == "__main__":
    main()
