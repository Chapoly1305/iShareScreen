"""Extract automatic worst adjacent-frame crops from the fixed MP4 gallery."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    root = Path(
        "benchmarks/super_resolution/artifacts/text-low-bandwidth-e2e-v1"
    ).resolve()
    media = read(root / "comparison-media/results.json")
    if (
        sha256(root / "comparison-media/results.json")
        != read(root / "comparison-media/COMPLETED.json")["results_sha256"]
    ):
        raise ValueError("Comparison media changed from completion receipt")
    protocol = read(root / "EVAL_PROTOCOL.json")
    cases = {case["id"]: case for case in protocol["cases"] if case["split"] == "test"}
    output = root / "comparison-media/temporal-inspection"
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for item in media["rows"]:
        source = root / "comparison-media" / item["video"]
        if sha256(source) != item["video_sha256"]:
            raise ValueError("Presentation video changed")
        x, y, w, h = cases[item["case"]]["line"]["xywh"]
        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise ValueError("Cannot read presentation video")
        previous = None
        best = None
        index = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape != (1080, 3840, 3):
                raise ValueError("Comparison presentation dimensions changed")
            base = rgb[y : y + h, x : x + w].copy()
            model = rgb[y : y + h, x + 1920 : x + 1920 + w].copy()
            if previous is not None:
                base_diff = float(
                    np.abs(base.astype(np.int16) - previous[0].astype(np.int16)).mean()
                )
                model_diff = float(
                    np.abs(model.astype(np.int16) - previous[1].astype(np.int16)).mean()
                )
                amplification = model_diff - base_diff
                if best is None or amplification > best["amplification"]:
                    best = dict(
                        frame=index,
                        base_diff=base_diff,
                        model_diff=model_diff,
                        amplification=amplification,
                        before_base=previous[0],
                        after_base=base,
                        before_model=previous[1],
                        after_model=model,
                    )
            previous = (base, model)
            index += 1
        capture.release()
        if index != item["frames"] or best is None:
            raise ValueError("Temporal inspection lacks frozen video frames")
        top = np.concatenate((best["before_base"], best["before_model"]), axis=1)
        bottom = np.concatenate((best["after_base"], best["after_model"]), axis=1)
        target = output / (source.stem + "-worst-adjacent.png")
        Image.fromarray(np.concatenate((top, bottom), axis=0)).save(target)
        records.append(
            dict(
                video=item["video"],
                case=item["case"],
                source_kind=item["source_kind"],
                requested_bps=item["requested_bps"],
                base_height=item["base_height"],
                worst_second_frame=best["frame"],
                base_crop_mean_abs_rgb=best["base_diff"],
                model_crop_mean_abs_rgb=best["model_diff"],
                amplification=best["amplification"],
                crop=str(target.relative_to(root)),
                crop_sha256=sha256(target),
                presentation_video_sha256=item["video_sha256"],
                layout="Top row previous frame, bottom row current; left no enhancement, right selected model. Original crop pixels from H.264 presentation encode.",
            )
        )
    atomic_json(
        output / "INSPECTION.json",
        dict(
            schema="isharescreen_low_bandwidth_temporal_inspection_v1",
            media_results_sha256=sha256(root / "comparison-media/results.json"),
            records=records,
            scope="Automatic worst model-minus-baseline adjacent-frame difference among all 89 pairs in each preselected video; presentation encoding makes this qualitative, while raw RGB temporal statistics are in comparison-media/results.json.",
        ),
    )
    print("TEMPORAL INSPECTION", len(records), "videos")


if __name__ == "__main__":
    main()
