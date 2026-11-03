"""Check visible-text passage overlap with the existing source-bound corpus."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def shingles(words: list[dict]) -> set[bytes]:
    tokens = re.findall(
        r"[\w]+",
        unicodedata.normalize(
            "NFKC", " ".join(word["text"] for word in words)
        ).casefold(),
    )
    return {
        hashlib.blake2b(" ".join(tokens[i : i + 5]).encode(), digest_size=8).digest()
        for i in range(max(0, len(tokens) - 4))
    }


def run(protocol_path: Path, old_root: Path, output: Path) -> None:
    protocol = json.loads(protocol_path.read_text())
    old = json.loads((old_root / "dataset-views-v3.json").read_text())
    entries = []
    postings: dict[bytes, list[int]] = defaultdict(list)
    for row in old["views"]:
        labels = json.loads((old_root / row["labels"]).read_text())
        grams = shingles(labels["words"])
        index = len(entries)
        entries.append((row["id"], row["split"], row["source_group"], grams))
        for gram in grams:
            postings[gram].append(index)
    independent = []
    comparisons = []
    high = []
    root = protocol_path.parent
    for row in protocol["cases"]:
        labels = json.loads((root / row["labels"]).read_text())
        grams = shingles(labels["words"])
        independent.append((row, grams))
        hits: Counter[int] = Counter()
        for gram in grams:
            hits.update(postings.get(gram, ()))
        ranked = []
        for index, common in hits.most_common(20):
            old_id, old_split, old_group, old_grams = entries[index]
            containment = common / min(len(grams), len(old_grams))
            jaccard = common / (len(grams) + len(old_grams) - common)
            ranked.append(
                dict(
                    old_view=old_id,
                    old_split=old_split,
                    old_group=old_group,
                    shared_5grams=common,
                    containment=containment,
                    jaccard=jaccard,
                )
            )
        ranked.sort(
            key=lambda item: (item["containment"], item["shared_5grams"]), reverse=True
        )
        match = ranked[0] if ranked else None
        result = dict(
            case=row["id"],
            split=row["split"],
            source_group=row["source_group"],
            fivegrams=len(grams),
            nearest_old=match,
        )
        comparisons.append(result)
        if match and match["shared_5grams"] >= 30 and match["containment"] >= 0.8:
            high.append(result)
    cross_split = []
    for i, (left, grams_left) in enumerate(independent):
        for right, grams_right in independent[i + 1 :]:
            if left["split"] == right["split"]:
                continue
            shared = len(grams_left & grams_right)
            containment = shared / min(len(grams_left), len(grams_right))
            if shared >= 30 and containment >= 0.8:
                cross_split.append(
                    dict(
                        left=left["id"],
                        right=right["id"],
                        shared_5grams=shared,
                        containment=containment,
                    )
                )
    atomic_json(
        output,
        dict(
            schema="isharescreen_visible_text_overlap_audit_v1",
            protocol_sha256=sha256(protocol_path),
            old_views_sha256=sha256(old_root / "dataset-views-v3.json"),
            old_view_count=len(entries),
            independent_count=len(independent),
            metric="case-folded NFKC word 5-gram containment; flag >=30 shared grams and >=0.8 containment",
            comparisons=comparisons,
            high_overlap_with_old=high,
            cross_split_near_duplicates=cross_split,
            passed=not high and not cross_split,
            scope="Visible text near-duplicate screen; not a proof against all paraphrases, shared short passages, or image-only duplication.",
            runner_sha256=sha256(Path(__file__)),
        ),
    )
    print(
        "AUDIT",
        len(entries),
        "prior views",
        len(independent),
        "new cases",
        len(high),
        "old overlaps",
        len(cross_split),
        "cross-split overlaps",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.old_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
