"""List and verify only the frozen percent holdout files a GPU worker needs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def run(root: Path, split: str, output: Path) -> None:
    root = root.resolve()
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["schema"] != "isharescreen_percent_code_holdout_v1":
        raise ValueError("Expected frozen percent holdout")
    files = {"EVAL_PROTOCOL.json"}
    cases = [c for c in protocol["cases"] if c["split"] == split]
    for case in cases:
        files.add(case["hr"])
        for width, height in protocol["main_grid"]["base_geometry"]:
            for bps in protocol["main_grid"]["requested_video_bps"]:
                folder = (
                    Path("evaluation-inputs")
                    / case["id"]
                    / f"{width}x{height}-{bps // 1000}k"
                )
                receipt_path = root / folder / "COMPLETED.json"
                receipt = json.loads(receipt_path.read_text())
                if (
                    receipt["binding"]["protocol_sha256"] != sha256(protocol_path)
                    or receipt["binding"]["case"] != case["id"]
                    or not receipt["encoder_eligible"]
                    or len(receipt["sampled_frames"]) != 2
                ):
                    raise ValueError(f"Invalid encoded cell: {folder}")
                files.add(str(folder / "COMPLETED.json"))
                for frame in receipt["sampled_frames"]:
                    files.add(str(folder / frame["path"]))
    output.mkdir(parents=True, exist_ok=False)
    checks = {name: sha256(root / name) for name in sorted(files)}
    (output / "files-from.txt").write_text("\n".join(sorted(files)) + "\n")
    atomic_json(
        output / "STAGE.json",
        dict(
            schema="isharescreen_percent_holdout_stage_v1",
            protocol_sha256=sha256(protocol_path),
            split=split,
            cases=[c["id"] for c in cases],
            files_sha256=checks,
            count=len(files),
            runner_sha256=sha256(Path(__file__)),
            finished=time.time(),
        ),
    )
    print("PERCENT STAGE", split, len(cases), len(files), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.split, args.output)
