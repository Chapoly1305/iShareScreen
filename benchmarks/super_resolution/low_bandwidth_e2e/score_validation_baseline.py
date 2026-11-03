"""Score all six unenhanced validation cells with the frozen Linux OCR backend."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import subprocess
import time
import unicodedata

import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.base_text_quality import distance
from benchmarks.super_resolution.metrics import metrics
from benchmarks.super_resolution.text_corpus.ocr_protocol import prepare_crop
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def score(path: Path, truth: str) -> dict:
    prediction = subprocess.check_output(
        [
            "tesseract",
            str(path),
            "stdout",
            "-l",
            "eng",
            "--oem",
            "1",
            "--psm",
            "7",
            "--dpi",
            "300",
        ],
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ, OMP_THREAD_LIMIT="1"),
    )
    gt = unicodedata.normalize("NFKC", truth)
    pred = unicodedata.normalize("NFKC", prediction)
    exact_gt = "".join(c for c in gt if not c.isspace())
    exact_pred = "".join(c for c in pred if not c.isspace())
    gt_words = gt.split()
    pred_words = pred.split()
    gt_digits = "".join(c for c in exact_gt if c.isdigit())
    pred_digits = "".join(c for c in exact_pred if c.isdigit())
    gt_punct = "".join(c for c in exact_gt if not c.isalnum())
    pred_punct = "".join(c for c in exact_pred if not c.isalnum())
    return dict(
        prediction=prediction,
        ground_truth=truth,
        characters=len(exact_gt),
        character_edits=distance(exact_gt, exact_pred),
        cer=distance(exact_gt, exact_pred) / len(exact_gt),
        words=len(gt_words),
        word_edits=distance(gt_words, pred_words),
        wer=distance(gt_words, pred_words) / len(gt_words),
        exact_line=exact_gt == exact_pred,
        digits=len(gt_digits),
        digit_edits=distance(gt_digits, pred_digits),
        punctuation=len(gt_punct),
        punctuation_edits=distance(gt_punct, pred_punct),
    )


def run(root: Path, tessdata: Path) -> None:
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = read(protocol_path)
    if sha256(tessdata / "eng.traineddata") != protocol["ocr"]["traineddata_sha256"]:
        raise ValueError("OCR model differs from frozen protocol")
    if (
        subprocess.check_output(
            ["tesseract", "--version"], stderr=subprocess.STDOUT, text=True
        )
        != protocol["ocr"]["engine"]
    ):
        raise ValueError("Tesseract build differs from frozen protocol")
    out = root / "validation-baseline"
    if out.exists():
        raise FileExistsError("Validation baseline output exists; inspect before rerun")
    out.mkdir()
    rows = []
    clean = []
    for case in [c for c in protocol["cases"] if c["split"] == "validation"]:
        source = root / case["hr"]
        if sha256(source) != case["hr_sha256"]:
            raise ValueError("Source render changed")
        reference = np.array(Image.open(source).convert("RGB"))
        clean_path = out / (case["id"] + "-clean-ocr.png")
        prepare_crop(Image.fromarray(reference), case["line"]["xywh"], clean_path)
        clean.append(
            dict(
                id=case["id"],
                source_group=case["source_group"],
                crop=str(clean_path.relative_to(root)),
                crop_sha256=sha256(clean_path),
                **score(clean_path, case["line"]["text"]),
            )
        )
        for width, height in protocol["main_grid"]["base_geometry"]:
            for bps in protocol["main_grid"]["requested_video_bps"]:
                cell = (
                    root
                    / "evaluation-inputs"
                    / case["id"]
                    / f"{width}x{height}-{bps // 1000}k"
                )
                receipt = read(cell / "COMPLETED.json")
                if receipt["binding"]["protocol_sha256"] != sha256(protocol_path):
                    raise ValueError("Evaluation encode belongs to another protocol")
                for frame in receipt["sampled_frames"]:
                    path = cell / frame["path"]
                    if sha256(path) != frame["sha256"]:
                        raise ValueError("Decoded frame differs")
                    base = np.array(Image.open(path).convert("RGB"))
                    output = (
                        base
                        if height == 1080
                        else cv2.resize(
                            base, (1920, 1080), interpolation=cv2.INTER_CUBIC
                        )
                    )
                    native_crop = out / (
                        f"{case['id']}-{height}p-{bps // 1000}k-f{frame['index']}-native.png"
                    )
                    x, y, w, h = case["line"]["xywh"]
                    Image.fromarray(output[y : y + h, x : x + w]).save(native_crop)
                    crop = out / (
                        f"{case['id']}-{height}p-{bps // 1000}k-f{frame['index']}-ocr.png"
                    )
                    prepare_crop(Image.fromarray(output), case["line"]["xywh"], crop)
                    quality = metrics(reference, output)
                    rows.append(
                        dict(
                            id=case["id"],
                            source_kind=case["source_kind"],
                            source_group=case["source_group"],
                            content_category=case["content_category"],
                            line_box_height=case["line"]["xywh"][3],
                            base=[width, height],
                            requested_bps=bps,
                            measured_video_bps=receipt[
                                "measured_video_bps_after_first_second"
                            ],
                            encoder_eligible=receipt["encoder_eligible"],
                            vbv_underflows=receipt["vbv_underflows"],
                            frame_index=frame["index"],
                            keyframe=frame["keyframe"],
                            method="baseline",
                            native_crop=str(native_crop.relative_to(root)),
                            native_crop_sha256=sha256(native_crop),
                            ocr_crop=str(crop.relative_to(root)),
                            ocr_crop_sha256=sha256(crop),
                            quality=quality,
                            **score(crop, case["line"]["text"]),
                        )
                    )
        print("VALIDATION BASELINE", case["id"], flush=True)
    atomic_json(out / "clean-reference-rows.json", clean)
    atomic_json(out / "ocr-rows.json", rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["base"][1], row["requested_bps"], row["frame_index"])].append(row)
    summaries = []
    for (height, bps, frame), values in sorted(grouped.items()):
        chars = sum(row["characters"] for row in values)
        words = sum(row["words"] for row in values)
        summaries.append(
            dict(
                base_height=height,
                requested_bps=bps,
                frame=frame,
                cases=len(values),
                source_groups=len({r["source_group"] for r in values}),
                pooled_cer=sum(r["character_edits"] for r in values) / chars,
                pooled_wer=sum(r["word_edits"] for r in values) / words,
                exact_lines=sum(r["exact_line"] for r in values),
                digit_edits=sum(r["digit_edits"] for r in values),
                punctuation_edits=sum(r["punctuation_edits"] for r in values),
                mean_psnr=float(np.mean([r["quality"]["psnr"] for r in values])),
                mean_ssim=float(np.mean([r["quality"]["ssim"] for r in values])),
                encoder_eligible=sum(r["encoder_eligible"] for r in values),
            )
        )
    atomic_json(out / "summary.json", summaries)
    atomic_json(
        out / "COMPLETED.json",
        dict(
            protocol_sha256=sha256(protocol_path),
            rows_sha256=sha256(out / "ocr-rows.json"),
            summary_sha256=sha256(out / "summary.json"),
            clean_rows_sha256=sha256(out / "clean-reference-rows.json"),
            runner_sha256=sha256(Path(__file__)),
            finished=time.time(),
            scope="Validation-only baseline; independent test remains unopened for model selection.",
        ),
    )
    print("VALIDATION BASELINE COMPLETE", len(rows), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tessdata", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.tessdata.resolve())


if __name__ == "__main__":
    main()
