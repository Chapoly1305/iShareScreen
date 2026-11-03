"""Stage only frozen independent-test inputs and validation-selected models."""

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
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--existing-weights", type=Path, required=True)
    parser.add_argument(
        "--methods", nargs="+", help="Selected methods assigned to this physical host"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    selection_path = args.selection.resolve()
    selection = read(selection_path)
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = read(protocol_path)
    if selection["eval_protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Selection refers to another evaluation protocol")
    roster = args.methods or selection["test_methods"]
    if len(roster) != len(set(roster)) or not roster:
        raise ValueError("Test roster must be nonempty and unique")
    if not set(roster).issubset(selection["test_methods"]):
        raise ValueError("Host assignment includes an unselected test method")
    source_weights = read(args.existing_weights.resolve())
    if source_weights["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Model weights refer to another evaluation protocol")
    models = {}
    for name in roster:
        if name not in source_weights["models"]:
            raise ValueError(f"Selected method is unavailable on this host: {name}")
        models[name] = source_weights["models"][name]
        if selection["test_checkpoint_sha256"][name] != models[name]["sha256"]:
            raise ValueError(f"Selected checkpoint identity changed: {name}")
    cases = [case for case in protocol["cases"] if case["split"] == "test"]
    if len(cases) != 72:
        raise ValueError("Independent test case count differs from frozen protocol")
    files = {"EVAL_PROTOCOL.json"}
    for case in cases:
        hr = root / case["hr"]
        if sha256(hr) != case["hr_sha256"]:
            raise ValueError(f"High-resolution truth render changed: {case['id']}")
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
                    raise ValueError("Encoded test input refers to another protocol")
                files.add(str(receipt_path))
                for frame in receipt["sampled_frames"]:
                    path = folder / frame["path"]
                    if sha256(root / path) != frame["sha256"]:
                        raise ValueError("Decoded test frame differs from receipt")
                    files.add(str(path))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "files-from.txt").write_text("\n".join(sorted(files)) + "\n")
    atomic_json(
        args.output / "weights-test.json",
        dict(
            schema="isharescreen_remote_test_weights_v1",
            protocol_sha256=sha256(protocol_path),
            selection_sha256=sha256(selection_path),
            models=models,
        ),
    )
    atomic_json(
        args.output / "stage-manifest.json",
        dict(
            schema="isharescreen_low_bandwidth_test_stage_v1",
            protocol_sha256=sha256(protocol_path),
            selection_sha256=sha256(selection_path),
            test_cases=[c["id"] for c in cases],
            methods=roster,
            files_sha256={name: sha256(root / name) for name in sorted(files)},
            weights_sha256=sha256(args.output / "weights-test.json"),
            finished=time.time(),
            runner_sha256=sha256(Path(__file__)),
        ),
    )
    print(
        "TEST STAGE", len(cases), "cases", len(files), "files", len(roster), "methods"
    )


if __name__ == "__main__":
    main()
