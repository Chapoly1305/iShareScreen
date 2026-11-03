"""Write a minimal, hash-bound file list and checkpoint map for a GPU worker."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def run(root: Path, output: Path, remote_data: Path, cases_limit: int | None) -> None:
    output.mkdir(parents=True, exist_ok=False)
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = read(protocol_path)
    cases = [case for case in protocol["cases"] if case["split"] == "validation"]
    if cases_limit is not None:
        cases = cases[:cases_limit]
    files = {"EVAL_PROTOCOL.json"}
    for case in cases:
        files.add(case["hr"])
        for width, height in protocol["main_grid"]["base_geometry"]:
            for bps in protocol["main_grid"]["requested_video_bps"]:
                folder = (
                    Path("evaluation-inputs")
                    / case["id"]
                    / f"{width}x{height}-{bps // 1000}k"
                )
                receipt_path = folder / "COMPLETED.json"
                receipt = read(root / receipt_path)
                if receipt["binding"]["protocol_sha256"] != sha256(protocol_path):
                    raise ValueError("Evaluation input belongs to another protocol")
                files.add(str(receipt_path))
                files.update(
                    str(folder / frame["path"]) for frame in receipt["sampled_frames"]
                )
    (output / "files-from.txt").write_text("\n".join(sorted(files)) + "\n")
    baseline = read(root / "BASELINES.json")
    models = {}
    checkpoints = []
    for row in baseline["models"]:
        identifier = row["id"]
        if identifier == "official_quicksrnet_x2":
            models[identifier] = dict(sha256=row["checkpoint_sha256"])
            continue
        remote = remote_data / "weights" / (identifier + ".pt")
        receipt = read(Path(row["completion_receipt"]))
        models[identifier] = dict(
            path=str(remote),
            sha256=row["checkpoint_sha256"],
            seed=row["seed"],
            protocol_binding=receipt["protocol_binding"],
        )
        checkpoints.append(
            dict(
                id=identifier,
                local=row["checkpoint"],
                remote=str(remote),
                sha256=row["checkpoint_sha256"],
            )
        )
    atomic_json(
        output / "weights.json",
        dict(
            schema="isharescreen_remote_validation_weights_v1",
            protocol_sha256=sha256(protocol_path),
            models=models,
            model_registry_sha256=sha256(root / "BASELINES.json"),
        ),
    )
    atomic_json(
        output / "stage-manifest.json",
        dict(
            schema="isharescreen_low_bandwidth_remote_stage_v1",
            protocol_sha256=sha256(protocol_path),
            split="validation",
            cases=[case["id"] for case in cases],
            file_count=len(files),
            files_sha256={name: sha256(root / name) for name in sorted(files)},
            weights_sha256=sha256(output / "weights.json"),
            checkpoints=checkpoints,
            remote_data=str(remote_data),
            runner_sha256=sha256(Path(__file__)),
            finished=time.time(),
        ),
    )
    print("STAGE LIST", len(cases), "cases", len(files), "files", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-data", type=Path, required=True)
    parser.add_argument("--limit-cases", type=int)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve(), args.remote_data, args.limit_cases)


if __name__ == "__main__":
    main()
