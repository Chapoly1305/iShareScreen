"""Encode the 2,000-view source pilot through the existing real HEVC pair path."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

from benchmarks.super_resolution.text_corpus.degrade import encode
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def order(value: object) -> str:
    return hashlib.sha256(
        ("isharescreen-pilot-encoder-v1:" + str(value)).encode()
    ).hexdigest()


def make_plan(work: Path, source_index: Path) -> dict:
    data = read(source_index)
    if not data["complete"] or data["count"] != 2000:
        raise ValueError("Source pilot is not complete")
    views = data["new_pdf"] + data["existing_train"]
    buckets = {600000: [], 1000000: []}
    for view in views:
        bps = 600000 if int(order(view["id"])[:8], 16) % 2 else 1000000
        buckets[bps].append(view)
    jobs = []
    for bps, group in buckets.items():
        for start in range(0, len(group), 12):
            part = group[start : start + 12]
            jobs.append(
                dict(
                    id=f"pilot-{bps // 1000}-{start // 12:04d}",
                    split="train",
                    source_group="mixed_original_train_sources",
                    bps=bps,
                    views=part,
                )
            )
    jobs.sort(key=lambda row: order(row["id"]))
    return dict(
        schema="isharescreen_low_bandwidth_hevc_training_pilot_v1",
        stage="hevc-pilot-v1",
        source_index_sha256=sha256(source_index),
        script_sha256=sha256(Path(__file__)),
        existing_encoder_sha256=sha256(Path(encode.__code__.co_filename)),
        jobs=jobs,
        fps=30,
        frames_per_view=30,
        geometry=[960, 540],
        reference_geometry=[1920, 1080],
        resampler="OpenCV Lanczos4 RGB downsample of source-rendered FHD",
        codec="libx265 fast/zerolatency, yuv420p, BT.709 full range, 60-frame GOP, no B frames/lookahead, one-second VBV and HRD/strict-CBR request",
        sample="One actual decoded frame per original view at local frame 0/5/15/29; scroll episode flag and keyframe stored.",
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[
            0
        ],
        created=time.time(),
    )


def run(work: Path, workers: int, limit_jobs: int | None) -> None:
    work.mkdir(parents=True, exist_ok=True)
    with (work / "ENCODE.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_index = work / "source-index.json"
        source_receipt = read(work / "SOURCE-INDEX-COMPLETED.json")
        if sha256(source_index) != source_receipt["index_sha256"]:
            raise ValueError("Source index identity differs")
        stage = work / "hevc-pilot-v1"
        stage.mkdir(exist_ok=True)
        plan_path = stage / "plan.json"
        if not plan_path.exists():
            atomic_json(plan_path, make_plan(work, source_index))
        plan = read(plan_path)
        if (
            plan["source_index_sha256"] != sha256(source_index)
            or plan["script_sha256"] != sha256(Path(__file__))
            or plan["existing_encoder_sha256"]
            != sha256(Path(encode.__code__.co_filename))
        ):
            raise ValueError("Pilot encoding code/source plan changed")
        jobs = plan["jobs"][:limit_jobs] if limit_jobs is not None else plan["jobs"]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = []
            for index, row in enumerate(
                pool.map(lambda job: encode(work, plan, job), jobs), 1
            ):
                rows.append(row)
                if index % 5 == 0 or index == len(jobs):
                    atomic_json(
                        stage / "status.json",
                        dict(
                            phase="encoding",
                            jobs_done=index,
                            jobs_total=len(plan["jobs"]),
                            encoded_views=sum(len(r["pairs"]) for r in rows),
                            updated=time.time(),
                        ),
                    )
                    print("PILOT ENCODE", index, "/", len(plan["jobs"]), flush=True)
        pairs = []
        for episode in rows:
            for pair in episode["pairs"]:
                pairs.append(
                    dict(
                        pair,
                        actual_episode_video_bps=episode["actual_average_video_bps"],
                        episode_vbv_underflows=episode["vbv_underflows"],
                    )
                )
        result = dict(
            schema="isharescreen_low_bandwidth_pilot_pairs_v1",
            plan_sha256=sha256(plan_path),
            source_index_sha256=sha256(source_index),
            complete=len(jobs) == len(plan["jobs"]) and len(pairs) == 2000,
            episodes=len(rows),
            count=len(pairs),
            rate_counts=dict(Counter(pair["requested_bps"] for pair in pairs)),
            underflow_episodes=sum(row["vbv_underflows"] > 0 for row in rows),
            pairs=pairs,
            runner_sha256=sha256(Path(__file__)),
            finished=time.time(),
        )
        atomic_json(
            stage / ("pairs-smoke.json" if limit_jobs is not None else "pairs.json"),
            result,
        )
        if result["complete"]:
            atomic_json(
                stage / "PAIRS-COMPLETED.json",
                dict(
                    pairs_sha256=sha256(stage / "pairs.json"),
                    count=len(pairs),
                    finished=time.time(),
                ),
            )
        print(
            "PILOT PAIRS", result["count"], "complete", result["complete"], flush=True
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-jobs", type=int)
    args = parser.parse_args()
    run(args.work.resolve(), args.workers, args.limit_jobs)


if __name__ == "__main__":
    main()
