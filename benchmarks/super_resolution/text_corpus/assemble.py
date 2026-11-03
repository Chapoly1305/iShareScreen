"""Release repaired training pairs while preserving both codec-stage receipts."""

import argparse
from collections import Counter
from pathlib import Path
import time

from benchmarks.super_resolution.text_corpus.render import read
from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement.models import sha256


def run(args):
    root = args.root.resolve()
    views = read(root / "dataset-views-v3.json")
    audit = read(root / "quality-audit-v3.json")
    original = read(root / "hevc-v1/pairs.json")
    repairs = read(root / "hevc-repairs-v1/pairs.json")
    jobs = {}
    for stage, data in [("hevc-v1", original), ("hevc-repairs-v1", repairs)]:
        if read(root / stage / "COMPLETED.json")["pairs_sha256"] != sha256(
            root / stage / "pairs.json"
        ) or data["plan_sha256"] != sha256(root / stage / "plan.json"):
            raise ValueError("Codec completion/plan identity differs")
        jobs.update(
            {(stage, j["id"]): j for j in read(root / stage / "plan.json")["jobs"]}
        )
    if (
        not all(d["complete"] for d in [views, original, repairs])
        or not audit["passed"]
    ):
        raise ValueError("Incomplete or unaudited parent stage")
    if original["views_sha256"] != views["initial_views_sha256"]:
        raise ValueError("Original encoding source differs")
    if (
        repairs["views_sha256"] != sha256(root / "repair-views.json")
        or read(root / "repair-views.json")["parent_sha256"]
        != views["geometry_parent_sha256"]
    ):
        raise ValueError("Replacement encoding source differs")
    if audit["views_sha256"] != sha256(root / "dataset-views-v3.json"):
        raise ValueError("Final audit source differs")
    excluded = set(views["quarantined_ids"])
    pairs = [p for p in original["pairs"] if p["id"] not in excluded] + repairs["pairs"]
    indexed = {r["id"]: r for r in views["views"]}
    if len(pairs) != 10000 or {p["id"] for p in pairs} != set(indexed):
        raise ValueError("Missing or duplicated final view IDs")
    for p in pairs:
        expected = indexed[p["id"]]
        amended = {r["id"] for r in views["annotation_updates"]}
        if p["id"] in amended:
            for k in ["labels", "labels_sha256", "characters"]:
                p[k] = expected[k]
        if any(p[k] != expected[k] for k in expected):
            raise ValueError("Pair/view identity mismatch")
        if sha256(root / p["lr"]) != p["lr_sha256"]:
            raise ValueError("LR image changed")
    episodes = {p["encoded"]: p for p in pairs}
    episode_records = {}
    for encoded in episodes:
        path = root / encoded
        receipt_path = path.parent / "COMPLETED.json"
        receipt = read(receipt_path)
        job = jobs[(Path(encoded).parts[0], path.parent.name)]
        if sha256(path) != receipt["video_sha256"] or receipt[
            "job_binding"
        ] != fingerprint(job):
            raise ValueError("Encoded episode identity differs")
        episode_records[encoded] = dict(
            video_sha256=receipt["video_sha256"], receipt_sha256=sha256(receipt_path)
        )
    out = root / "hevc-v2"
    out.mkdir(exist_ok=False)
    result = dict(
        original,
        pairs=pairs,
        count=len(pairs),
        complete=True,
        views_sha256=sha256(root / "dataset-views-v3.json"),
        parent_stages={
            name: sha256(root / name / "pairs.json")
            for name in ["hevc-v1", "hevc-repairs-v1"]
        },
        excluded_view_ids=sorted(excluded),
        annotation_updates=views["annotation_updates"],
        quality_audit_sha256=sha256(root / "quality-audit-v3.json"),
        episode_records=episode_records,
        rate_counts=dict(Counter(p["requested_bps"] for p in pairs)),
        split_counts=dict(Counter(p["split"] for p in pairs)),
        keyframe_samples=sum(p["keyframe"] for p in pairs),
        episodes=len(episodes),
        vbv_underflow_episodes=sum(
            p["episode_vbv_underflows"] > 0 for p in episodes.values()
        ),
        finished=time.time(),
    )
    atomic_json(out / "pairs.json", result)
    atomic_json(
        out / "COMPLETED.json",
        dict(pairs_sha256=sha256(out / "pairs.json"), complete=True),
    )
    print(
        "RELEASED", len(pairs), "aligned HEVC pairs", result["split_counts"], flush=True
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
