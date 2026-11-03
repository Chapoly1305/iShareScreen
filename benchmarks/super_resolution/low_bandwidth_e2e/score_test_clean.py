"""Record the fixed OCR backend's clean-render ceiling on independent test."""

from __future__ import annotations

import argparse
import json
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
    if read(selection_path)["eval_protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Selection differs from independent test protocol")
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
    output = root / "test-clean-reference"
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in (c for c in protocol["cases"] if c["split"] == "test"):
        image_path = root / case["hr"]
        if sha256(image_path) != case["hr_sha256"]:
            raise ValueError("High-resolution source render changed")
        target = output / (case["id"] + "-ocr.png")
        with Image.open(image_path) as image:
            prepare_crop(image.convert("RGB"), case["line"]["xywh"], target)
        rows.append(
            dict(
                id=case["id"],
                source_group=case["source_group"],
                source_kind=case["source_kind"],
                content_category=case["content_category"],
                crop=str(target.relative_to(root)),
                crop_sha256=sha256(target),
                **score(target, case["line"]["text"]),
            )
        )
    if len(rows) != 72:
        raise ValueError("Clean reference OCR does not cover the frozen test")
    atomic_json(output / "ocr-rows.json", rows)
    atomic_json(
        output / "COMPLETED.json",
        dict(
            rows=72,
            protocol_sha256=sha256(protocol_path),
            selection_sha256=sha256(selection_path),
            runner_sha256=sha256(Path(__file__)),
            rows_sha256=sha256(output / "ocr-rows.json"),
            finished=time.time(),
        ),
    )
    print(
        "CLEAN TEST OCR",
        len(rows),
        "rows",
        "pooled CER",
        sum(x["character_edits"] for x in rows) / sum(x["characters"] for x in rows),
        "clean exact",
        sum(x["exact_line"] for x in rows),
    )


if __name__ == "__main__":
    main()
