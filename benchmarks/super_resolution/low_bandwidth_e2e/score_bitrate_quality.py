"""Compute paired full FHD PSNR/SSIM for the direct display on worker CPU."""

from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.metrics import metrics
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def run(root: Path, split: str, output: Path) -> None:
    root = root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = read(protocol_path)
    binding = dict(
        protocol_sha256=sha256(protocol_path),
        split=split,
        runner_sha256=sha256(Path(__file__)),
    )
    with (output / "TASK.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress_path = output / "progress.json"
        if progress_path.exists():
            progress = read(progress_path)
            if progress["binding"] != binding:
                raise ValueError("Quality score resume binding differs")
            rows = progress["rows"]
        else:
            rows = []
        done = {
            (r["case"], r["base_height"], r["requested_bps"], r["frame_index"])
            for r in rows
        }
        if len(done) != len(rows):
            raise ValueError("Duplicate scored quality rows")
        cases = [c for c in protocol["cases"] if c["split"] == split]
        expected = (
            len(cases)
            * len(protocol["main_grid"]["base_geometry"])
            * len(protocol["main_grid"]["requested_video_bps"])
            * 2
        )
        for case in cases:
            source = root / case["hr"]
            if sha256(source) != case["hr_sha256"]:
                raise ValueError("Source reference changed")
            reference = np.asarray(Image.open(source).convert("RGB"))
            for width, height in protocol["main_grid"]["base_geometry"]:
                for bps in protocol["main_grid"]["requested_video_bps"]:
                    cell = (
                        root
                        / "evaluation-inputs"
                        / case["id"]
                        / f"{width}x{height}-{bps // 1000}k"
                    )
                    receipt = read(cell / "COMPLETED.json")
                    if (
                        receipt["binding"]["protocol_sha256"]
                        != binding["protocol_sha256"]
                    ):
                        raise ValueError("Encoded cell belongs to another protocol")
                    for frame in receipt["sampled_frames"]:
                        key = case["id"], height, bps, frame["index"]
                        if key in done:
                            continue
                        path = cell / frame["path"]
                        if sha256(path) != frame["sha256"]:
                            raise ValueError("Decoded pixels changed")
                        decoded = np.asarray(Image.open(path).convert("RGB"))
                        displayed = (
                            decoded
                            if height == 1080
                            else cv2.resize(
                                decoded, (1920, 1080), interpolation=cv2.INTER_CUBIC
                            )
                        )
                        rows.append(
                            dict(
                                case=case["id"],
                                source_group=case["source_group"],
                                source_kind=case["source_kind"],
                                base_height=height,
                                requested_bps=bps,
                                frame_index=frame["index"],
                                **metrics(reference, displayed),
                            )
                        )
                        done.add(key)
                    atomic_json(
                        progress_path,
                        dict(
                            binding=binding,
                            expected=expected,
                            rows=rows,
                            updated=time.time(),
                        ),
                    )
                    print("DIRECT QUALITY", split, len(rows), "/", expected, flush=True)
        if len(rows) != expected:
            raise ValueError("Incomplete direct quality grid")
        atomic_json(
            output / "results.json",
            dict(binding=binding, rows=rows, completed=time.time()),
        )
        atomic_json(
            output / "COMPLETED.json",
            dict(
                results_sha256=sha256(output / "results.json"),
                rows=len(rows),
                finished=time.time(),
            ),
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    run(a.root, a.split, a.output)
