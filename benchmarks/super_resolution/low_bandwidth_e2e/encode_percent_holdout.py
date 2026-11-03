"""Encode frozen percent holdout with the original HEVC policy."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import time

from benchmarks.super_resolution.low_bandwidth_e2e.encode_evaluation import encode_one
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def run(root: Path, workers: int) -> None:
    root = root.resolve()
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["schema"] != "isharescreen_percent_code_holdout_v1":
        raise ValueError("Expected frozen percent holdout")
    output = root / "evaluation-inputs"
    output.mkdir(exist_ok=True)
    with (output / "TASK.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        jobs = [
            (case, width, height, bps)
            for case in protocol["cases"]
            for width, height in protocol["main_grid"]["base_geometry"]
            for bps in protocol["main_grid"]["requested_video_bps"]
        ]
        rows = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for i, row in enumerate(
                pool.map(lambda job: encode_one(root, protocol, *job), jobs), 1
            ):
                rows.append(row)
                if i % 12 == 0:
                    atomic_json(
                        output / "status.json",
                        dict(
                            completed=i,
                            expected=len(jobs),
                            eligible=sum(r["encoder_eligible"] for r in rows),
                            updated=time.time(),
                        ),
                    )
                    print("PERCENT ENCODE", i, "/", len(jobs), flush=True)
        if len(rows) != len(jobs) or not all(x["encoder_eligible"] for x in rows):
            raise ValueError("Incomplete or rate-ineligible percent holdout encoding")
        atomic_json(
            output / "manifest.json",
            dict(
                schema="isharescreen_percent_encoded_holdout_v1",
                protocol_sha256=sha256(protocol_path),
                runner_sha256=sha256(Path(__file__)),
                rows=rows,
                count=len(rows),
                complete=True,
                finished=time.time(),
            ),
        )
        atomic_json(
            output / "INPUTS-COMPLETED.json",
            dict(
                protocol_sha256=sha256(protocol_path),
                manifest_sha256=sha256(output / "manifest.json"),
                count=len(rows),
                finished=time.time(),
            ),
        )
        print("PERCENT ENCODE COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.root, args.workers)
