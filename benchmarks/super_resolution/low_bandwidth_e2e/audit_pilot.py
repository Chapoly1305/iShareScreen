"""Verify new train source groups and prepare cross-corpus overlap screening."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def run(root: Path, old_root: Path) -> None:
    work = root / "pilot"
    source_path = work / "source-index.json"
    source = read(source_path)
    receipt = read(work / "SOURCE-INDEX-COMPLETED.json")
    plan = read(work / "source-plan.json")
    eval_protocol = read(root / "EVAL_PROTOCOL.json")
    if sha256(source_path) != receipt["index_sha256"] or source[
        "plan_sha256"
    ] != sha256(work / "source-plan.json"):
        raise ValueError("Pilot source identity differs")
    if source["count"] != 2000 or not source["complete"]:
        raise ValueError("Pilot source index incomplete")
    new = source["new_pdf"]
    old = source["existing_train"]
    if len(new) != 1200 or len(old) != 800:
        raise ValueError("Unexpected source composition")
    selected = {row["source_group"] for row in eval_protocol["cases"]}
    train_groups = {row["source_group"] for row in new + old}
    if selected & train_groups:
        raise ValueError("Original source group overlaps independent validation/test")
    old_views = {
        row["id"]: row for row in read(old_root / "dataset-views-v3.json")["views"]
    }
    for row in old:
        original = old_views[row["id"]]
        if (
            original["split"] != "train"
            or original["source_group"] != row["source_group"]
        ):
            raise ValueError("Old evaluation/test source entered pilot")
    candidates = {row["page_hash"] for row in plan["doclaynet_candidates"]}
    if any(
        Path(row["source_pdf"]).parent.name not in candidates
        or row["official_split"] != "train"
        for row in new
    ):
        raise ValueError("New PDF is outside official train candidate list")
    for row in new:
        if sha256(work / row["source_pdf"]) != row["source_pdf_sha256"]:
            raise ValueError("Original PDF changed")
        if (
            sha256(work / row["hr"]) != row["hr_sha256"]
            or sha256(work / row["labels"]) != row["labels_sha256"]
        ):
            raise ValueError("Rendered new source changed")
    atomic_json(
        work / "newpdf-visible-protocol.json",
        dict(
            schema="isharescreen_train_pdf_overlap_input_v1",
            source_index_sha256=sha256(source_path),
            cases=[
                dict(
                    id=row["id"],
                    labels=row["labels"],
                    split="train",
                    source_group=row["source_group"],
                )
                for row in new
            ],
        ),
    )
    atomic_json(
        work / "source-audit.json",
        dict(
            schema="isharescreen_pilot_source_audit_v1",
            source_index_sha256=sha256(source_path),
            eval_protocol_sha256=sha256(root / "EVAL_PROTOCOL.json"),
            train_views=2000,
            new_pdf_views=len(new),
            old_train_views=len(old),
            kind_counts=dict(Counter(row["kind"] for row in new + old)),
            train_source_groups=len(train_groups),
            independent_group_intersection=0,
            original_pdf_source_and_render_hashes_verified=True,
            old_views_all_original_train=True,
            official_new_pdf_train_candidate_binding=True,
            visible_text_cross_corpus_audit="pending in newpdf-visible-overlap.json",
            runner_sha256=sha256(Path(__file__)),
            finished=time.time(),
        ),
    )
    print(
        "PILOT SOURCE AUDIT",
        len(new),
        len(old),
        len(train_groups),
        "groups",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.old_root.resolve())


if __name__ == "__main__":
    main()
