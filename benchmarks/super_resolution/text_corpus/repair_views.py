"""Quarantine exact cross-split text duplicates and admit distinct replacements.

Initial generation and codec artifacts remain immutable. The new release has a
new index and an explicit two-stage lineage; no failed audit is overwritten.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import time

from benchmarks.super_resolution.text_corpus.render import read
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def visible_hash(labels):
    return hashlib.sha256(
        "".join("".join(w["text"].split()) for w in labels["words"]).encode()
    ).hexdigest()


def run(args):
    root = args.root.resolve()
    initial = read(root / "dataset-views.json")
    audit = read(root / "quality-audit.json")
    if audit["views_sha256"] != sha256(root / "dataset-views.json"):
        raise ValueError("Audit identity mismatch")
    if (root / "dataset-views-v2.json").exists():
        raise ValueError("Repaired view release already exists")
    rejected = {
        r["id"] for group in audit["cross_split_visible_text_duplicates"] for r in group
    }
    rows = [r for r in initial["views"] if r["id"] not in rejected]
    seen = {r["hr_rgb_sha256"] for r in rows}
    used = {r["id"] for r in initial["views"]}
    texts = defaultdict(set)
    for r in rows:
        texts[visible_hash(read(root / r["labels"]))].add(r["split"])
    needed = Counter(
        (r["kind"], r["split"]) for r in initial["views"] if r["id"] in rejected
    )
    added = []
    for path in sorted((root / "views").glob("extra-*/view.json")):
        labels = read(path)
        key = (labels["kind"], labels["split"])
        if needed[key] <= 0 or labels["id"] in used or labels["hr_rgb_sha256"] in seen:
            continue
        text = visible_hash(labels)
        if text in texts:
            continue
        if sha256(root / labels["hr"]) != labels["hr_sha256"]:
            raise ValueError("Replacement pixels changed")
        row = {
            k: labels[k]
            for k in [
                "id",
                "kind",
                "split",
                "source_group",
                "hr",
                "hr_sha256",
                "hr_rgb_sha256",
                "characters",
            ]
        }
        row.update(labels=str(path.relative_to(root)), labels_sha256=sha256(path))
        rows.append(row)
        added.append(row)
        needed[key] -= 1
        seen.add(row["hr_rgb_sha256"])
        texts[text].add(row["split"])
    if any(needed.values()) or len(rows) != 10000:
        raise ValueError("Insufficient distinct replacement layouts")
    result = dict(
        initial,
        views=rows,
        initial_views_sha256=sha256(root / "dataset-views.json"),
        initial_audit_sha256=sha256(root / "quality-audit.json"),
        quarantined_ids=sorted(rejected),
        replacement_ids=[r["id"] for r in added],
        counts=dict(Counter(r["kind"] for r in rows)),
        split_counts=dict(Counter(r["split"] for r in rows)),
        repair="Remove every member of exact visible-text groups shared across splits; use distinct same-kind/same-split layouts from the already rendered supplementary pool.",
        repair_script_sha256=sha256(Path(__file__)),
        revised=time.time(),
    )
    atomic_json(root / "dataset-views-v2.json", result)
    atomic_json(
        root / "repair-views.json",
        dict(
            schema="isharescreen_text_views_v1",
            views=added,
            count=len(added),
            complete=True,
            parent_sha256=sha256(root / "dataset-views-v2.json"),
        ),
    )
    print(
        "QUARANTINED",
        sorted(rejected),
        "REPLACED",
        result["replacement_ids"],
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
