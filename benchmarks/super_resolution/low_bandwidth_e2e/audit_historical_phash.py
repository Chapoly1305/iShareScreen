"""Perceptual screen against every frozen historical 10k render."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from benchmarks.super_resolution.low_bandwidth_e2e.audit_image_overlap import phash
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path):
    return json.loads(path.read_text())


def run(root: Path, output: Path) -> None:
    old_path = root.parent / "text-corpus-v1/dataset-views-v2.json"
    eval_path = root / "EVAL_PROTOCOL.json"
    old, protocol = read(old_path), read(eval_path)
    if len(old["views"]) != 10000 or len(protocol["cases"]) != 108:
        raise ValueError("Historical/evaluation corpus counts differ")
    hashes = []
    for index, row in enumerate(old["views"]):
        path = Path(row["hr"])
        if not path.is_absolute():
            path = old_path.parent / path
        if not path.is_file():
            raise FileNotFoundError(path)
        hashes.append(phash(path))
        if (index + 1) % 1000 == 0:
            atomic_json(
                output.with_suffix(".status.json"),
                dict(
                    phase="historical_hash",
                    completed=index + 1,
                    total=10000,
                    updated=time.time(),
                ),
            )
            print("HISTORICAL IMAGE HASH", index + 1, "/ 10000", flush=True)
    flags = []
    nearest = []
    for case in protocol["cases"]:
        source = root / case["hr"]
        if sha256(source) != case["hr_sha256"]:
            raise ValueError(f"Independent evaluation render changed: {case['id']}")
        value = phash(source)
        distance = sorted(
            ((value ^ digest).bit_count(), index) for index, digest in enumerate(hashes)
        )
        top = []
        for delta, index in distance[:5]:
            row = old["views"][index]
            item = dict(
                eval_id=case["id"],
                eval_split=case["split"],
                eval_source_group=case["source_group"],
                old_id=row["id"],
                old_split=row["split"],
                old_kind=row["kind"],
                old_source_group=row["source_group"],
                phash_hamming=delta,
            )
            top.append(item)
        for delta, index in distance:
            if delta > 4:
                break
            row = old["views"][index]
            flags.append(
                dict(
                    eval_id=case["id"],
                    eval_split=case["split"],
                    eval_source_group=case["source_group"],
                    old_id=row["id"],
                    old_split=row["split"],
                    old_kind=row["kind"],
                    old_source_group=row["source_group"],
                    phash_hamming=delta,
                )
            )
        nearest.append(dict(eval_id=case["id"], best=top))
    report = dict(
        schema="isharescreen_low_bandwidth_historical_perceptual_audit_v1",
        historical_manifest_sha256=sha256(old_path),
        eval_protocol_sha256=sha256(eval_path),
        historical_images_hashed=len(hashes),
        independent_images_hashed=108,
        threshold_hamming_le=4,
        flags=flags,
        flags_by_split=dict(Counter(x["eval_split"] for x in flags)),
        nearest_five_per_eval=nearest,
        scope="Perceptual DCT screen across all historical 10,000 renders and frozen 108 independent cases. A close visual layout is a review flag; exact pixels and source text are audited separately.",
        runner_sha256=sha256(Path(__file__)),
        finished=time.time(),
    )
    atomic_json(output, report)
    atomic_json(
        output.with_suffix(".status.json"),
        dict(phase="completed", completed=10000, total=10000, updated=time.time()),
    )
    print("HISTORICAL PHASH AUDIT COMPLETE", len(flags), "flags", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
