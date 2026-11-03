"""Score direct FHD display of the frozen bitrate frontier on H100."""

from __future__ import annotations

import argparse
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


def run(
    root: Path,
    split: str,
    tessdata: Path,
    allow_partial: bool = False,
    skip_quality: bool = False,
) -> None:
    root = root.resolve()
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    if sha256(tessdata / "eng.traineddata") != protocol["ocr"]["traineddata_sha256"]:
        raise ValueError("OCR language model differs")
    if (
        subprocess.check_output(
            ["tesseract", "--version"], stderr=subprocess.STDOUT, text=True
        )
        != protocol["ocr"]["engine"]
    ):
        raise ValueError("OCR engine differs")
    output = root / ("baseline-" + split)
    output.mkdir(exist_ok=True)
    with (output / "TASK.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress = output / "progress.json"
        if progress.exists():
            prior = json.loads(progress.read_text())
            if prior["protocol_sha256"] != sha256(protocol_path):
                raise ValueError("Previous baseline scoring used a different protocol")
            rows = prior["rows"]
            for row in rows:
                if (
                    sha256(root / row["native_crop"]) != row["native_crop_sha256"]
                    or sha256(root / row["ocr_crop"]) != row["ocr_crop_sha256"]
                ):
                    raise ValueError("Previous scored crop changed")
        else:
            rows = []
        done = {
            (r["id"], r["base_height"], r["requested_bps"], r["frame_index"])
            for r in rows
        }
        if len(done) != len(rows):
            raise ValueError("Duplicate baseline rows")
        for case in [c for c in protocol["cases"] if c["split"] == split]:
            reference_path = root / case["hr"]
            if sha256(reference_path) != case["hr_sha256"]:
                raise ValueError("Reference changed")
            reference = np.asarray(Image.open(reference_path).convert("RGB"))
            for width, height in protocol["main_grid"]["base_geometry"]:
                for bps in protocol["main_grid"]["requested_video_bps"]:
                    folder = (
                        root
                        / "evaluation-inputs"
                        / case["id"]
                        / f"{width}x{height}-{bps // 1000}k"
                    )
                    if allow_partial and not (folder / "COMPLETED.json").exists():
                        continue
                    receipt = json.loads((folder / "COMPLETED.json").read_text())
                    if receipt["binding"]["protocol_sha256"] != sha256(protocol_path):
                        raise ValueError("Encoded input refers to another protocol")
                    for frame in receipt["sampled_frames"]:
                        key = (case["id"], height, bps, frame["index"])
                        if key in done:
                            continue
                        source = folder / frame["path"]
                        if sha256(source) != frame["sha256"]:
                            raise ValueError("Decoded input changed")
                        decoded = np.asarray(Image.open(source).convert("RGB"))
                        displayed = (
                            decoded
                            if height == 1080
                            else cv2.resize(
                                decoded, (1920, 1080), interpolation=cv2.INTER_CUBIC
                            )
                        )
                        x, y, w, h = case["line"]["xywh"]
                        stem = f"{case['id']}-{height}p-{bps // 1000}k-f{frame['index']:03d}"
                        native = output / "native-crops" / (stem + ".png")
                        native.parent.mkdir(exist_ok=True)
                        Image.fromarray(displayed[y : y + h, x : x + w]).save(native)
                        ocr = output / "ocr-crops" / (stem + ".png")
                        ocr.parent.mkdir(exist_ok=True)
                        prepare_crop(
                            Image.fromarray(displayed), case["line"]["xywh"], ocr
                        )
                        quality = (
                            {"psnr": None, "ssim": None}
                            if skip_quality
                            else metrics(reference, displayed)
                        )
                        rows.append(
                            dict(
                                id=case["id"],
                                split=split,
                                source_kind=case["source_kind"],
                                source_group=case["source_group"],
                                content_category=case["content_category"],
                                font_size=case["font_size"],
                                base_height=height,
                                requested_bps=bps,
                                measured_video_bps=receipt[
                                    "measured_video_bps_after_first_second"
                                ],
                                encoder_eligible=receipt["encoder_eligible"],
                                frame_index=frame["index"],
                                keyframe=frame["keyframe"],
                                method="baseline",
                                native_crop=str(native.relative_to(root)),
                                native_crop_sha256=sha256(native),
                                ocr_crop=str(ocr.relative_to(root)),
                                ocr_crop_sha256=sha256(ocr),
                                psnr=quality["psnr"],
                                ssim=quality["ssim"],
                                **score(ocr, case["line"]["text"]),
                            )
                        )
                        done.add(key)
                    atomic_json(
                        progress,
                        dict(
                            protocol_sha256=sha256(protocol_path),
                            rows=rows,
                            updated=time.time(),
                        ),
                    )
            print("FRONTIER BASELINE", split, case["id"], len(rows), flush=True)
        expected = (
            len([c for c in protocol["cases"] if c["split"] == split])
            * len(protocol["main_grid"]["base_geometry"])
            * len(protocol["main_grid"]["requested_video_bps"])
            * 2
        )
        if allow_partial and len(rows) < expected:
            print(
                "FRONTIER BASELINE PARTIAL", split, len(rows), "/", expected, flush=True
            )
            return
        if len(rows) != expected:
            raise ValueError("Incomplete baseline")
        atomic_json(output / "ocr-rows.json", rows)
        atomic_json(
            output / "COMPLETED.json",
            dict(
                protocol_sha256=sha256(protocol_path),
                runner_sha256=sha256(Path(__file__)),
                rows_sha256=sha256(output / "ocr-rows.json"),
                rows=len(rows),
                finished=time.time(),
            ),
        )
        print("FRONTIER BASELINE COMPLETE", split, len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--tessdata", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument(
        "--skip-quality",
        action="store_true",
        help="OCR first; compute unchanged full FHD PSNR/SSIM separately on worker CPU",
    )
    args = parser.parse_args()
    run(
        args.root,
        args.split,
        args.tessdata.resolve(),
        args.allow_partial,
        args.skip_quality,
    )
