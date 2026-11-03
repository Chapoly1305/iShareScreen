"""Verify every transferred independent-test byte and selected checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.stage.read_text())
    selection = json.loads(args.selection.read_text())
    if digest(args.selection) != manifest["selection_sha256"]:
        raise ValueError("Frozen validation selection changed in transfer")
    if manifest["protocol_sha256"] != digest(args.data / "EVAL_PROTOCOL.json"):
        raise ValueError("Independent evaluation protocol changed in transfer")
    for relative, expected in manifest["files_sha256"].items():
        if digest(args.data / relative) != expected:
            raise ValueError(f"Test input changed in transfer: {relative}")
    weights = args.data / "weights-test.json"
    if digest(weights) != manifest["weights_sha256"]:
        raise ValueError("Selected model binding changed in transfer")
    models = json.loads(weights.read_text())["models"]
    if set(models) != set(manifest["methods"]):
        raise ValueError("Host-selected model roster differs")
    for name, bound in models.items():
        if (
            name not in selection["test_methods"]
            or selection["test_checkpoint_sha256"][name] != bound["sha256"]
        ):
            raise ValueError(f"Checkpoint differs from validation selection: {name}")
        if (
            name != "official_quicksrnet_x2"
            and digest(Path(bound["path"])) != bound["sha256"]
        ):
            raise ValueError(f"Selected checkpoint bytes changed: {name}")
    print(
        "VERIFIED REMOTE TEST",
        len(manifest["test_cases"]),
        "cases",
        len(manifest["files_sha256"]),
        "files",
        len(models),
        "model checkpoints",
    )


if __name__ == "__main__":
    main()
