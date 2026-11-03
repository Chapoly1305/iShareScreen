"""Score frozen clean percent holdout lines without filtering OCR failures."""

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


def run(root: Path, tessdata: Path) -> None:
    root = root.resolve()
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    if sha256(tessdata / "eng.traineddata") != protocol["ocr"]["traineddata_sha256"]:
        raise ValueError("Tessdata differs from frozen protocol")
    if (
        subprocess.check_output(
            ["tesseract", "--version"], stderr=subprocess.STDOUT, text=True
        )
        != protocol["ocr"]["engine"]
    ):
        raise ValueError("Tesseract build differs from frozen protocol")
    out = root / "clean-reference"
    if out.exists():
        raise FileExistsError(
            "Clean reference already exists; inspect receipt before rerun"
        )
    out.mkdir()
    rows = []
    for case in protocol["cases"]:
        source = root / case["hr"]
        if sha256(source) != case["hr_sha256"]:
            raise ValueError(f"Reference changed: {case['id']}")
        crop = out / "crops" / (case["id"] + ".png")
        crop.parent.mkdir(exist_ok=True)
        prepare_crop(Image.open(source).convert("RGB"), case["line"]["xywh"], crop)
        rows.append(
            dict(
                id=case["id"],
                split=case["split"],
                source_group=case["source_group"],
                font_size=case["font_size"],
                crop=str(crop.relative_to(root)),
                crop_sha256=sha256(crop),
                **score(crop, case["line"]["text"]),
            )
        )
    atomic_json(out / "ocr-rows.json", rows)
    atomic_json(
        out / "COMPLETED.json",
        dict(
            protocol_sha256=sha256(protocol_path),
            runner_sha256=sha256(Path(__file__)),
            rows_sha256=sha256(out / "ocr-rows.json"),
            rows=len(rows),
            finished=time.time(),
        ),
    )
    print("PERCENT CLEAN COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tessdata", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.tessdata.resolve())
