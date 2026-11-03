"""Bounded luminance-detail controls after official EDSR; no learned/OCR input.

Six fixed image-only settings are frozen before any candidate is scored.
This is an independent development experiment, never a final-test selection.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.mac_readability.quality import save, sha


def enhance(rgb, sigma, amount):
    x = rgb.astype(np.float32)
    y = x @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    detail = amount * (
        y - cv2.GaussianBlur(y, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    )
    low = cv2.erode(y, np.ones((3, 3), np.uint8))
    high = cv2.dilate(y, np.ones((3, 3), np.uint8))
    delta = np.clip(np.clip(y + np.clip(detail, -8, 8), low, high) - y, -8, 8)
    return np.clip(np.rint(x + delta[:, :, None]), 0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    cv2.setNumThreads(1)
    root = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    out = root / "detail-controls-v1"
    out.mkdir(exist_ok=False)
    index = read(root / "dense-models-v1/IMAGES-LOCAL.json")
    rows = [r for r in index["images"] if r["method"] == "edsr-official"]
    assert len(rows) == 32
    settings = [
        dict(id=f"detail-s{sigma:.1f}-a{amount:.2f}", sigma=sigma, amount=amount)
        for sigma in [0.6, 1.0]
        for amount in [0.25, 0.5, 1.0]
    ]
    protocol = dict(
        schema="bounded_detail_controls_v1",
        scope=__doc__,
        started=time.time(),
        settings=settings,
        script_sha256=sha(Path(__file__)),
        input_sha256=sha(root / "dense-models-v1/IMAGES-LOCAL.json"),
        comparator_sha256=sha(root / "COMPARATOR-FROZEN.json"),
        max_channel_delta=8,
        boundary="Reflect101 Gaussian; luminance output bounded by original3x3 min/max; RGB values clipped to8-bit",
        selection="Frozen hard15 document macroCER; retain full16,zero-error,category,digit,punctuation,cell and visual regressions; latency must be profiled on actual4090 before integration",
        no_ocr_or_labels_in_transform=True,
        test_used=False,
    )
    save(out / "PROTOCOL.json", protocol)
    for setting in settings:
        folder = out / setting["id"]
        folder.mkdir()
        images = []
        timings = []
        for row in rows:
            assert sha(row["path"]) == row["sha256"]
            rgb = np.asarray(Image.open(row["path"]).convert("RGB"))
            start = time.perf_counter()
            enhanced = enhance(rgb, setting["sigma"], setting["amount"])
            timings.append((time.perf_counter() - start) * 1000)
            assert np.abs(enhanced.astype("i2") - rgb.astype("i2")).max() <= 8
            name = row["id"] + "-" + setting["id"]
            path = folder / (name + ".png")
            Image.fromarray(enhanced).save(path)
            images.append(
                dict(
                    row,
                    id=name,
                    method=setting["id"],
                    path=str(path.resolve()),
                    sha256=sha(path),
                )
            )
        save(
            folder / "IMAGES.json",
            dict(
                source_sha256=index["source_sha256"],
                images=images,
                diagnostic_h100_cpu_ms=timings,
                scope="CPU transform only, not receiver or end-to-end performance",
            ),
        )
    # All images and settings exist before the first quality score is inspected.
    for setting in settings:
        folder = out / setting["id"]
        save(
            out / "STATUS.json",
            dict(state="scoring", setting=setting["id"], updated=time.time()),
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.super_resolution.mac_readability.score_lines",
                "--source",
                str(
                    a.project
                    / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/reference"
                ),
                "--index",
                str(folder / "IMAGES.json"),
                "--output",
                str(folder / "ocr"),
                "--tessdata",
                str(a.project / "tools-runtime/tessdata"),
                "--frozen-ocr",
                str(root / "dense-clean-v1/PROTOCOL.json"),
                "--workers",
                str(a.workers),
            ],
            check=True,
        )
    save(
        out / "STATUS.json",
        dict(state="scoring_complete", updated=time.time(), selection_pending=True),
    )


def read(path):
    return json.loads(path.read_text())


if __name__ == "__main__":
    main()
