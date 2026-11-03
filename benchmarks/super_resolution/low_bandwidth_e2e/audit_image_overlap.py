"""Audit exact pixel identity and perceptual similarity across frozen splits."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path):
    return json.loads(path.read_text())


def phash(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(
            image.convert("L").resize((32, 32), Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    dct = cv2.dct(gray)[:8, :8].flatten()
    bits = dct > np.median(dct[1:])
    return sum(int(bit) << i for i, bit in enumerate(bits))


def run(root: Path, output: Path) -> None:
    pilot_path = root / "pilot/source-index.json"
    eval_path = root / "EVAL_PROTOCOL.json"
    old_path = root.parent / "text-corpus-v1/dataset-views-v2.json"
    pilot, evaluation, old = read(pilot_path), read(eval_path), read(old_path)
    train = pilot["new_pdf"] + pilot["existing_train"]
    if (
        len(train) != 2000
        or len(evaluation["cases"]) != 108
        or len(old["views"]) != 10000
    ):
        raise ValueError("Frozen corpus sizes differ")
    old_by_pixel = defaultdict(list)
    old_by_file = defaultdict(list)
    for row in old["views"]:
        old_by_pixel[row["hr_rgb_sha256"]].append(row["id"])
        old_by_file[row["hr_sha256"]].append(row["id"])
    training_exact = []
    independent_exact = []
    for row in train:
        if row["hr_rgb_sha256"] in old_by_pixel:
            training_exact.append(
                dict(id=row["id"], matching_old=old_by_pixel[row["hr_rgb_sha256"]])
            )
    for case in evaluation["cases"]:
        matches = old_by_file.get(case["hr_sha256"], [])
        if matches:
            independent_exact.append(
                dict(id=case["id"], kind="file_bytes", matching_old=matches)
            )
        # Official PDF and generated HTML protocol cases bind both original bytes
        # and visible pixels. The latter catches harmless PNG metadata changes.
        if case.get("hr_rgb_sha256") in old_by_pixel:
            independent_exact.append(
                dict(
                    id=case["id"],
                    kind="rgb_pixels",
                    matching_old=old_by_pixel[case["hr_rgb_sha256"]],
                )
            )
    independent_groups = {case["source_group"] for case in evaluation["cases"]}
    crossing = sorted({row["source_group"] for row in train} & independent_groups)
    train_hashes = []
    for index, row in enumerate(train):
        path = (
            Path(row["hr"])
            if Path(row["hr"]).is_absolute()
            else root / "pilot" / row["hr"]
        )
        if sha256(path) != row["hr_sha256"]:
            raise ValueError(f"Training render changed: {row['id']}")
        train_hashes.append(phash(path))
        if (index + 1) % 200 == 0:
            print("TRAIN IMAGE HASH", index + 1, "/", len(train), flush=True)
    test_hashes = []
    for case in evaluation["cases"]:
        path = root / case["hr"]
        if sha256(path) != case["hr_sha256"]:
            raise ValueError(f"Independent render changed: {case['id']}")
        test_hashes.append(phash(path))
    near = []
    for case, value in zip(evaluation["cases"], test_hashes):
        for row, other in zip(train, train_hashes):
            distance = (value ^ other).bit_count()
            if distance <= 4:
                near.append(
                    dict(
                        eval_id=case["id"],
                        eval_split=case["split"],
                        eval_kind=case["source_kind"],
                        eval_source_group=case["source_group"],
                        train_id=row["id"],
                        train_kind=row["kind"],
                        train_source_group=row["source_group"],
                        phash_hamming=distance,
                    )
                )
    report = dict(
        schema="isharescreen_low_bandwidth_image_overlap_audit_v1",
        pilot_source_index_sha256=sha256(pilot_path),
        eval_protocol_sha256=sha256(eval_path),
        old_10000_manifest_sha256=sha256(old_path),
        train_views=len(train),
        independent_cases=len(evaluation["cases"]),
        historical_views=len(old["views"]),
        exact_train_in_old_count=len(training_exact),
        exact_train_in_old_examples=training_exact[:25],
        exact_independent_in_old=independent_exact,
        crossing_source_groups=crossing,
        phash_algorithm="8x8 median DCT of bilinear 32x32 gray render; Hamming distance <=4 only flags visual review, not a definitive duplicate",
        perceptual_near_train_vs_independent=near,
        near_pair_count=len(near),
        existing_text_shingle_audit="independent-pdf/overlap-audit.json and html-overlap-audit.json compare visible 5-word shingles against all historical 10,000 views",
        limitation="Perceptual audit covers 2,000 pilot train views and 108 independent cases. Historical 10,000 are compared by exact image identity and separate visible-text 5-gram audit; perceptual hashing of all 10,000 historical images is not included.",
        runner_sha256=sha256(Path(__file__)),
        finished=time.time(),
    )
    atomic_json(output, report)
    print(
        "IMAGE AUDIT COMPLETE",
        len(near),
        "perceptual flags",
        len(independent_exact),
        "exact independent/old overlaps",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
