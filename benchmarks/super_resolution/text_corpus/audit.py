"""Audit every frozen image/label and flag content leakage before training."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path):
    return json.loads(path.read_text())


def inspect(root, row):
    if (
        sha256(root / row["hr"]) != row["hr_sha256"]
        or sha256(root / row["labels"]) != row["labels_sha256"]
    ):
        raise ValueError("Changed image or geometric labels: " + row["id"])
    with Image.open(root / row["hr"]) as image:
        rgb = np.array(image.convert("RGB"))
    if (
        rgb.shape != (1080, 1920, 3)
        or hashlib.sha256(rgb.tobytes()).hexdigest() != row["hr_rgb_sha256"]
    ):
        raise ValueError("RGB pixels differ: " + row["id"])
    labels = read(root / row["labels"])
    if any(
        labels[k] != row[k] for k in ["id", "source_group", "split", "hr_rgb_sha256"]
    ):
        raise ValueError("Label identity mismatch")
    for word in labels["words"]:
        x, y, w, h = word["xywh"]
        if not (
            x >= 0
            and y >= 0
            and w > 0
            and h > 0
            and x + w <= 1920.01
            and y + h <= 1080.01
        ):
            raise ValueError("Invalid text box")
    text = "".join("".join(w["text"].split()) for w in labels["words"])
    return dict(
        id=row["id"],
        source_group=row["source_group"],
        split=row["split"],
        kind=row["kind"],
        text_sha256=hashlib.sha256(text.encode()).hexdigest(),
        characters=len(text),
        words=len(labels["words"]),
        coverage=labels["metadata"]["text_box_coverage"],
    )


def run(args):
    root = args.root.resolve()
    index = root / args.index
    views = read(index)
    sources = read(root / "sources.json")
    if not views["complete"] or views["count"] != 10000:
        raise ValueError("Need the completed view set")
    if views["sources_sha256"] != sha256(root / "sources.json"):
        raise ValueError("Source identity changed")
    for source in sources["sources"]:
        if sha256(root / source["path"]) != source["sha256"]:
            raise ValueError("Changed frozen source: " + source["id"])
    groups = {}
    seen = {}
    texts = defaultdict(list)
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, result in enumerate(
            pool.map(lambda r: inspect(root, r), views["views"]), 1
        ):
            group = views["group_aliases"].get(
                result["source_group"], result["source_group"]
            )
            if group in groups and groups[group] != result["split"]:
                raise ValueError("Source-family split leak")
            groups[group] = result["split"]
            texts[result["text_sha256"]].append(result)
            rows.append(result)
            if n % 500 == 0:
                print("AUDIT", n, "/ 10000", flush=True)
    duplicates = [values for values in texts.values() if len(values) > 1]
    leaks = [values for values in duplicates if len({v["split"] for v in values}) > 1]
    atomic_json(
        root / args.output,
        dict(
            schema="isharescreen_text_quality_audit_v1",
            views_sha256=sha256(index),
            source_files_verified=len(sources["sources"]),
            images_and_labels_verified=len(rows),
            source_groups=len(groups),
            duplicate_visible_text_groups=len(duplicates),
            cross_split_visible_text_duplicates=leaks,
            minimum_characters=min(v["characters"] for v in rows),
            minimum_text_box_coverage=min(v["coverage"] for v in rows),
            counts=dict(Counter(v["kind"] for v in rows)),
            split_counts=dict(Counter(v["split"] for v in rows)),
            passed=not leaks,
            script_sha256=sha256(Path(__file__)),
            finished=time.time(),
            scope="Full byte/pixel/geometry/group audit and exact visible-text duplication check. Does not prove absence of all near-duplicate passages or pixel-perfect text-layer semantics. Pilot visual review covers selected layouts only.",
        ),
    )
    print(
        "AUDIT COMPLETE: cross-split exact-visible-text groups", len(leaks), flush=True
    )
    if leaks:
        raise ValueError(
            "Cross-split visible-text duplicates require resolution before training"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--index", default="dataset-views.json")
    p.add_argument("--output", default="quality-audit.json")
    run(p.parse_args())


if __name__ == "__main__":
    main()
