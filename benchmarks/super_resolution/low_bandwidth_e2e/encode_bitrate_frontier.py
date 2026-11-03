"""Encode the frozen bitrate frontier, with per-cell verified resume."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import time

from benchmarks.super_resolution.low_bandwidth_e2e.encode_evaluation import encode_one
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def run(root: Path, workers: int, limit_cases: int | None) -> None:
    root = root.resolve()
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["schema"] != "isharescreen_bitrate_frontier_eval_v1":
        raise ValueError("Expected frozen bitrate frontier")
    out = root / "evaluation-inputs"
    out.mkdir(exist_ok=True)
    with (out / "TASK.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cases = (
            protocol["cases"][:limit_cases]
            if limit_cases is not None
            else protocol["cases"]
        )
        jobs = [
            (case, width, height, bps)
            for case in cases
            for width, height in protocol["main_grid"]["base_geometry"]
            for bps in protocol["main_grid"]["requested_video_bps"]
        ]
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, row in enumerate(
                pool.map(lambda job: encode_one(root, protocol, *job), jobs), 1
            ):
                rows.append(row)
                if index % 8 == 0 or index == len(jobs):
                    atomic_json(
                        out / "status.json",
                        dict(
                            completed=index,
                            expected=len(jobs),
                            eligible=sum(r["encoder_eligible"] for r in rows),
                            updated=time.time(),
                        ),
                    )
                    print("FRONTIER ENCODE", index, "/", len(jobs), flush=True)
        if len(rows) != len(jobs):
            raise ValueError("Missing HEVC cells")
        complete = limit_cases is None and len(cases) == len(protocol["cases"])
        manifest = dict(
            schema="isharescreen_bitrate_frontier_inputs_v1",
            protocol_sha256=sha256(protocol_path),
            complete=complete,
            counts=dict(Counter(row["split"] for row in rows)),
            eligible=sum(row["encoder_eligible"] for row in rows),
            rows=rows,
            runner_sha256=sha256(Path(__file__)),
            finished=time.time(),
        )
        path = out / (
            "smoke-manifest.json" if limit_cases is not None else "manifest.json"
        )
        atomic_json(path, manifest)
        if complete:
            atomic_json(
                out / "INPUTS-COMPLETED.json",
                dict(
                    manifest_sha256=sha256(path), count=len(rows), finished=time.time()
                ),
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-cases", type=int)
    args = parser.parse_args()
    run(args.root, args.workers, args.limit_cases)
