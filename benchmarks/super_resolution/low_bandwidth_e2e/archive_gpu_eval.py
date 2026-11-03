"""Verify a recovered GPU evaluation and charge its measured GPU occupancy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "remote-results" / args.job
    receipt = read(out / "COMPLETED.json")
    result_path = out / "results.json"
    if sha256(result_path) != receipt["results_sha256"]:
        raise ValueError("Recovered result changed from completion receipt")
    result = read(result_path)
    if result["binding"]["protocol_sha256"] != sha256(root / "EVAL_PROTOCOL.json"):
        raise ValueError("Result refers to another frozen protocol")
    if receipt["rows"] != len(result["rows"]):
        raise ValueError("Result row count differs from receipt")
    if receipt["checkpoint_sha256"] != result["identity"]["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from receipt")
    for row in result["rows"]:
        if sha256(out / row["crop"]) != row["crop_sha256"]:
            raise ValueError(f"Recovered crop changed: {row['crop']}")
        if (
            row["display"]
            and sha256(out / row["display"]["path"]) != row["display"]["sha256"]
        ):
            raise ValueError(f"Recovered display changed: {row['display']['path']}")
    registry_path = root / "EXPERIMENTS.json"
    registry = read(registry_path)
    matches = [x for x in registry["experiments"] if x["id"] == args.job]
    if len(matches) != 1:
        raise ValueError("Experiment registry must contain exactly one job")
    entry = matches[0]
    if entry["state"] == "completed_and_verified":
        raise ValueError("Job has already been archived and charged")
    if entry["host"] != args.host:
        raise ValueError("Registered host differs")
    elapsed = result["elapsed_gpu_job_s"]
    charged = elapsed + 3.0
    ledger_path = root / "RESOURCE_LEDGER.json"
    ledger = read(ledger_path)
    if any(x["id"] == args.job for x in ledger["entries"]):
        raise ValueError("GPU budget would be charged twice")
    if ledger["charged_gpu_hours"] + charged / 3600 > ledger["budget_gpu_hours"]:
        raise ValueError("GPU budget exceeded")
    gpu_entry = dict(
        id=args.job,
        host=args.host,
        physical_gpu=[0],
        gpu_uuid=args.gpu_uuid,
        gpu_name=result["gpu"],
        operation=f"{result['binding']['split']} GPU inference, {result['binding']['method']}, {len(result['rows'])} paired rows",
        start_unix=result["completed"] - elapsed,
        end_unix=result["completed"] + 3.0,
        charged_wall_seconds=charged,
        gpu_count=1,
        charged_gpu_hours=charged / 3600,
        measured_inference_job_seconds=elapsed,
        evidence=str((out / "COMPLETED.json").relative_to(root)),
        runner_sha256=result["binding"]["runner_sha256"],
        checkpoint_sha256=receipt["checkpoint_sha256"],
        note="Measured GPU worker time plus 3 seconds conservative startup/teardown allowance.",
    )
    entry.update(
        state="completed_and_verified",
        pid=args.pid,
        log=args.log,
        completion_receipt=gpu_entry["evidence"],
        rows=len(result["rows"]),
        gpu_seconds_charged=charged,
        checkpoint_sha256=receipt["checkpoint_sha256"],
        next="Run pinned Linux OCR on H100; preserve paired source rows and native crops.",
    )
    status_path = root / "STATUS.json"
    status = read(status_path)
    for job in status["jobs"]:
        if job["id"] == args.job:
            job.update(entry)
    ledger["entries"].append(gpu_entry)
    ledger["charged_gpu_hours"] = sum(x["charged_gpu_hours"] for x in ledger["entries"])
    status["gpu_hours_spent_new_campaign"] = ledger["charged_gpu_hours"]
    status["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(ledger_path, ledger)
    atomic_json(registry_path, registry)
    atomic_json(status_path, status)
    print(
        "VERIFIED",
        args.job,
        len(result["rows"]),
        "rows",
        round(charged / 3600, 6),
        "GPU hours",
    )


if __name__ == "__main__":
    main()
