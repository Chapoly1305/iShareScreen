"""Score transferred GPU crops with the frozen H100 Linux OCR backend."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from PIL import Image

from benchmarks.super_resolution.low_bandwidth_e2e.score_validation_baseline import (
    score,
)
from benchmarks.super_resolution.text_corpus.ocr_protocol import prepare_crop
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def run(root: Path, gpu_result: Path, output: Path, tessdata: Path) -> None:
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = read(protocol_path)
    if sha256(tessdata / "eng.traineddata") != protocol["ocr"]["traineddata_sha256"]:
        raise ValueError("OCR language model differs from frozen protocol")
    if (
        subprocess.check_output(
            ["tesseract", "--version"], stderr=subprocess.STDOUT, text=True
        )
        != protocol["ocr"]["engine"]
    ):
        raise ValueError("OCR engine differs from frozen protocol")
    receipt = read(gpu_result / "COMPLETED.json")
    result_path = gpu_result / "results.json"
    if sha256(result_path) != receipt["results_sha256"]:
        raise ValueError("GPU result receipt differs")
    result = read(result_path)
    binding = result["binding"]
    if binding["protocol_sha256"] != sha256(protocol_path) or result.get(
        "subset_smoke"
    ):
        raise ValueError("GPU result does not cover the frozen full protocol")
    if receipt["rows"] != len(result["rows"]):
        raise ValueError("GPU row count differs")
    case_by_id = {
        c["id"]: c for c in protocol["cases"] if c["split"] == binding["split"]
    }
    expected = (
        len(case_by_id)
        * len(protocol["main_grid"]["base_geometry"])
        * len(protocol["main_grid"]["requested_video_bps"])
        * 2
    )
    if len(result["rows"]) != expected:
        raise ValueError(
            f"Expected {expected} paired GPU rows, got {len(result['rows'])}"
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "TASK.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        progress_path = output / "progress.json"
        score_binding = dict(
            protocol_sha256=sha256(protocol_path),
            gpu_result_sha256=sha256(result_path),
            method=binding["method"],
            runner_sha256=sha256(Path(__file__)),
        )
        if progress_path.exists():
            prior = read(progress_path)
            if prior["binding"] != score_binding:
                raise ValueError(
                    "Prior OCR scoring used a different result or protocol"
                )
            rows = prior["rows"]
            for row in rows:
                if sha256(output / row["ocr_crop"]) != row["ocr_crop_sha256"]:
                    raise ValueError("Existing scored OCR crop changed")
        else:
            rows = []
        done = {
            (x["id"], x["base_height"], x["requested_bps"], x["frame_index"])
            for x in rows
        }
        if len(done) != len(rows):
            raise ValueError("Duplicate previous OCR rows")
        for gpu in result["rows"]:
            key = (
                gpu["case"],
                gpu["base_height"],
                gpu["requested_bps"],
                gpu["frame_index"],
            )
            if key in done:
                continue
            case = case_by_id[gpu["case"]]
            source = gpu_result / gpu["crop"]
            if sha256(source) != gpu["crop_sha256"]:
                raise ValueError(f"GPU crop changed: {source}")
            with Image.open(source) as image:
                width, height = image.size
                if [width, height] != case["line"]["xywh"][2:]:
                    raise ValueError(
                        "GPU crop has different geometry from source text line"
                    )
                target = (
                    output
                    / "ocr-crops"
                    / gpu["case"]
                    / f"{gpu['base_height']}p-{gpu['requested_bps'] // 1000}k"
                )
                target.mkdir(parents=True, exist_ok=True)
                crop = target / f"frame-{gpu['frame_index']:03d}.png"
                prepare_crop(image, [0, 0, width, height], crop)
            rows.append(
                dict(
                    id=gpu["case"],
                    source_kind=case["source_kind"],
                    source_group=case["source_group"],
                    content_category=case["content_category"],
                    line_box_height=case["line"]["xywh"][3],
                    base_height=gpu["base_height"],
                    requested_bps=gpu["requested_bps"],
                    measured_video_bps=gpu["measured_video_bps"],
                    encoder_eligible=gpu["encoder_eligible"],
                    frame_index=gpu["frame_index"],
                    keyframe=gpu["keyframe"],
                    method=binding["method"],
                    native_crop=str(source.relative_to(root)),
                    native_crop_sha256=gpu["crop_sha256"],
                    ocr_crop=str(crop.relative_to(root)),
                    ocr_crop_sha256=sha256(crop),
                    psnr=gpu["psnr"],
                    ssim=gpu["ssim"],
                    forward_ms=gpu["forward_ms"],
                    full_processing_ms=gpu["full_processing_ms"],
                    **score(crop, case["line"]["text"]),
                )
            )
            done.add(key)
            if len(rows) % 12 == 0:
                atomic_json(
                    progress_path,
                    dict(
                        binding=score_binding,
                        rows=rows,
                        expected=expected,
                        updated=time.time(),
                    ),
                )
                print(
                    "GPU OCR", binding["method"], len(rows), "/", expected, flush=True
                )
        if len(rows) != expected:
            raise ValueError("Incomplete GPU OCR results")
        atomic_json(output / "ocr-rows.json", rows)
        atomic_json(
            output / "COMPLETED.json",
            dict(
                binding=score_binding,
                rows_sha256=sha256(output / "ocr-rows.json"),
                rows=len(rows),
                finished=time.time(),
            ),
        )
        print("GPU OCR COMPLETE", binding["method"], len(rows), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tessdata", type=Path, required=True)
    args = parser.parse_args()
    run(
        args.root.resolve(),
        args.gpu_result.resolve(),
        args.output.resolve(),
        args.tessdata.resolve(),
    )


if __name__ == "__main__":
    main()
