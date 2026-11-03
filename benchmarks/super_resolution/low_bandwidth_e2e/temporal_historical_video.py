"""Exploratory full-frame temporal diagnostics for historical direct video paths.

This post-hoc diagnostic is separate from the frozen text-selection protocol.
It pairs all 120 decoded frames of each 30 fps encode with every other frame
of the 60 fps source recording. It makes no OCR or live-network claim.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import zip_longest
from pathlib import Path

import av
import cv2
import numpy as np

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def _score(task: tuple[dict, str, tuple[int, int, int, int]]) -> dict:
    cell, reference, box = task
    if sha256(Path(cell["video"])) != cell["video_sha256"]:
        raise ValueError("video changed")
    cv2.setNumThreads(1)
    psnr = []
    roi_temporal_error = []
    previous_ref = None
    previous_output = None
    x, y, w, h = box
    with av.open(reference) as ref_container, av.open(cell["video"]) as enc_container:
        refs = (
            frame for i, frame in enumerate(ref_container.decode(video=0)) if i % 2 == 0
        )
        encs = enc_container.decode(video=0)
        for i, pair in enumerate(zip_longest(refs, encs)):
            if any(frame is None for frame in pair):
                raise ValueError(f"frame count mismatch: {cell['config']}")
            ref_frame, enc_frame = pair
            if abs(float(ref_frame.pts * ref_frame.time_base) - i / 30) > 0.002:
                raise ValueError("reference PTS mismatch")
            if abs(float(enc_frame.pts * enc_frame.time_base) - i / 30) > 0.002:
                raise ValueError("encoded PTS mismatch")
            ref = ref_frame.to_ndarray(format="rgb24")
            output = enc_frame.to_ndarray(format="rgb24")
            if cell["height"] == 720:
                output = cv2.resize(output, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            ref_roi = ref[y : y + h, x : x + w].astype(np.int16)
            out_roi = output[y : y + h, x : x + w].astype(np.int16)
            error = ref.astype(np.float32) - output.astype(np.float32)
            mse = float(np.mean(error * error))
            psnr.append(10 * np.log10(255**2 / mse) if mse else float("inf"))
            if previous_ref is not None:
                roi_temporal_error.append(
                    float(
                        np.mean(
                            np.abs(
                                (out_roi - previous_output) - (ref_roi - previous_ref)
                            )
                        )
                    )
                )
            previous_ref, previous_output = ref_roi, out_roi
    if len(psnr) != 120 or len(roi_temporal_error) != 119:
        raise ValueError("incomplete full-video diagnostic")
    return dict(
        scene=cell["scene"],
        config=cell["config"],
        height=cell["height"],
        target_kbps=cell["target_kbps"],
        source_video_sha256=cell["video_sha256"],
        frames=len(psnr),
        mean_psnr=float(np.mean(psnr)),
        min_psnr=float(np.min(psnr)),
        p5_psnr=float(np.percentile(psnr, 5)),
        roi_temporal_error_mean=float(np.mean(roi_temporal_error)),
        roi_temporal_error_p95=float(np.percentile(roi_temporal_error, 95)),
        psnr_by_frame=psnr,
        roi_temporal_error_by_transition=roi_temporal_error,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    plan = json.loads((root / "PLAN.json").read_text())
    manifest_path = root / "manifest.json"
    encodes = json.loads((root / "ENCODES-COMPLETED.json").read_text())
    if encodes["manifest_sha256"] != sha256(manifest_path):
        raise ValueError("encode manifest changed")
    manifest = json.loads(manifest_path.read_text())
    truth = json.loads(
        Path(
            "benchmarks/super_resolution/artifacts/base-sweep-v1/text-ground-truth.json"
        ).read_text()
    )
    if (
        sha256(
            Path(
                "benchmarks/super_resolution/artifacts/base-sweep-v1/text-ground-truth.json"
            )
        )
        != plan["ground_truth_sha256"]
    ):
        raise ValueError("region source changed")
    tasks = []
    for cell in manifest["rows"]:
        source = next(s for s in plan["sources"] if s["id"] == cell["scene"])
        region = next(r for r in truth["regions"] if r["scene"] == cell["scene"])
        tasks.append((cell, source["reference"], tuple(region["xywh"])))
    with ProcessPoolExecutor(max_workers=4) as executor:
        rows = list(executor.map(_score, tasks))
    if len(rows) != 18:
        raise ValueError("incomplete diagnostic")
    result = dict(
        schema="historical_video_temporal_diagnostic_v1",
        plan_sha256=sha256(root / "PLAN.json"),
        manifest_sha256=sha256(manifest_path),
        rows=rows,
        caveat="Post-hoc pixel diagnostic only. Temporal ROI difference error is not perceived flicker, text correctness or live-network performance.",
    )
    path = root / "TEMPORAL_DIAGNOSTIC.json"
    atomic_json(path, result)
    atomic_json(
        root / "TEMPORAL-COMPLETED.json",
        dict(
            result_sha256=sha256(path),
            cells=len(rows),
            total_frames=sum(r["frames"] for r in rows),
        ),
    )
    for kbps in (1000, 1250, 1500):
        for height in (720, 1080):
            selected = [
                r for r in rows if r["target_kbps"] == kbps and r["height"] == height
            ]
            print(
                f"{kbps}k {height}p full-video PSNR {np.mean([r['mean_psnr'] for r in selected]):.2f}; "
                f"ROI temporal error {np.mean([r['roi_temporal_error_mean'] for r in selected]):.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
