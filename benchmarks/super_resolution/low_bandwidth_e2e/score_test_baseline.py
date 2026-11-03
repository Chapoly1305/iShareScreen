"""Score unenhanced frozen independent-test frames after validation selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.low_bandwidth_e2e.score_validation_baseline import (
    score,
)
from benchmarks.super_resolution.metrics import metrics
from benchmarks.super_resolution.text_corpus.ocr_protocol import prepare_crop
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def score_one(
    root: Path,
    output: Path,
    case: dict,
    reference: np.ndarray,
    width: int,
    height: int,
    bps: int,
    frame: dict,
    receipt: dict,
) -> dict:
    cell = root / "evaluation-inputs" / case["id"] / f"{width}x{height}-{bps // 1000}k"
    image_path = cell / frame["path"]
    if sha256(image_path) != frame["sha256"]:
        raise ValueError("Decoded test frame changed")
    base = np.array(Image.open(image_path).convert("RGB"))
    display = (
        base
        if height == 1080
        else cv2.resize(base, (1920, 1080), interpolation=cv2.INTER_CUBIC)
    )
    x, y, w, h = case["line"]["xywh"]
    stem = f"{case['id']}-{height}p-{bps // 1000}k-f{frame['index']}"
    native = output / f"{stem}-native.png"
    Image.fromarray(display[y : y + h, x : x + w]).save(native)
    crop = output / f"{stem}-ocr.png"
    prepare_crop(Image.fromarray(display), case["line"]["xywh"], crop)
    quality = metrics(reference, display)
    return dict(
        id=case["id"],
        source_kind=case["source_kind"],
        source_group=case["source_group"],
        content_category=case["content_category"],
        line_box_height=h,
        base=[width, height],
        requested_bps=bps,
        measured_video_bps=receipt["measured_video_bps_after_first_second"],
        encoder_eligible=receipt["encoder_eligible"],
        vbv_underflows=receipt["vbv_underflows"],
        frame_index=frame["index"],
        keyframe=frame["keyframe"],
        method="baseline",
        native_crop=str(native.relative_to(root)),
        native_crop_sha256=sha256(native),
        ocr_crop=str(crop.relative_to(root)),
        ocr_crop_sha256=sha256(crop),
        quality=quality,
        **score(crop, case["line"]["text"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--tessdata", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = read(protocol_path)
    selection_path = args.selection.resolve()
    selection = read(selection_path)
    if selection["eval_protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Validation selection is not bound to the frozen test")
    if (
        sha256(args.tessdata.resolve() / "eng.traineddata")
        != protocol["ocr"]["traineddata_sha256"]
    ):
        raise ValueError("OCR language model differs")
    if (
        subprocess.check_output(
            ["tesseract", "--version"], stderr=subprocess.STDOUT, text=True
        )
        != protocol["ocr"]["engine"]
    ):
        raise ValueError("OCR engine differs")
    output = root / "test-baseline"
    output.mkdir(parents=True, exist_ok=True)
    with (output / "TASK.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        binding = dict(
            protocol_sha256=sha256(protocol_path),
            selection_sha256=sha256(selection_path),
            runner_sha256=sha256(Path(__file__)),
        )
        progress = output / "progress.json"
        if progress.exists():
            previous = read(progress)
            if previous["binding"] != binding:
                raise ValueError(
                    "Previous baseline run has another selection or runner"
                )
            rows = previous["rows"]
            for row in rows:
                if sha256(root / row["ocr_crop"]) != row["ocr_crop_sha256"]:
                    raise ValueError("Existing OCR crop changed")
        else:
            rows = []
        done = {
            (r["id"], r["base"][1], r["requested_bps"], r["frame_index"]) for r in rows
        }
        if len(done) != len(rows):
            raise ValueError("Duplicate baseline test row")
        cases = [case for case in protocol["cases"] if case["split"] == "test"]
        if len(cases) != 72:
            raise ValueError("Frozen test case count differs")
        with ThreadPoolExecutor(max_workers=8) as pool:
            for case in cases:
                reference_path = root / case["hr"]
                if sha256(reference_path) != case["hr_sha256"]:
                    raise ValueError("Truth render changed")
                reference = np.array(Image.open(reference_path).convert("RGB"))
                pending = {}
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
                            raise ValueError(
                                "HEVC test frame belongs to another protocol"
                            )
                        for frame in receipt["sampled_frames"]:
                            key = (case["id"], height, bps, frame["index"])
                            if key in done:
                                continue
                            pending[
                                pool.submit(
                                    score_one,
                                    root,
                                    output,
                                    case,
                                    reference,
                                    width,
                                    height,
                                    bps,
                                    frame,
                                    receipt,
                                )
                            ] = key
                for future in as_completed(pending):
                    row = future.result()
                    rows.append(row)
                    done.add(pending[future])
                atomic_json(
                    progress,
                    dict(binding=binding, rows=rows, expected=864, updated=time.time()),
                )
                print("TEST BASELINE", len(rows), "/ 864", flush=True)
        if len(rows) != 864 or len(done) != 864:
            raise ValueError("Incomplete independent baseline")
        atomic_json(output / "ocr-rows.json", rows)
        atomic_json(
            output / "COMPLETED.json",
            dict(
                binding=binding,
                rows=len(rows),
                rows_sha256=sha256(output / "ocr-rows.json"),
                finished=time.time(),
            ),
        )
        print("TEST BASELINE COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    main()
