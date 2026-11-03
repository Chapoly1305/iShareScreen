"""Frozen steady-state OCR on all dense development groups, including OCR failures.

Primary sample: 2.8 seconds after each source/font change. Groups are source
documents, not font variants or lines. Historical final test is never read.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess
import time

import av
import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.mac_readability.quality import (
    markers,
    ocr,
    save,
    sha,
    summarize,
)
from benchmarks.super_resolution.metrics import metrics


def regions(state):
    for label in state["labels"]:
        yield label
    if state["category"] != "table":
        return
    for index, label in enumerate(state["labels"]):
        x, y, w, h = label["xywh"]
        advance = (label["rect_points"][2] - 4) * 2 / len(label["text"])
        for j, match in enumerate(
            re.finditer(r"\S(?:.*?\S)?(?= {2,}|$)", label["text"])
        ):
            yield dict(
                label,
                unit="cell",
                label_id=f"cell-{index:02d}-{j:02d}",
                text=match.group(),
                xywh=[
                    round(x + 4 + match.start() * advance - 2),
                    y,
                    round(len(match.group()) * advance + 4),
                    h,
                ],
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--controls", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--score-index", type=Path)
    p.add_argument("--tessdata", type=Path, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--images-only", action="store_true")
    p.add_argument("--clean-only", action="store_true")
    p.add_argument(
        "--allow-output-frame-drops",
        action="store_true",
        help="Retain real encoder frame drops; require the exact frozen PTS and both source markers for every scored frame",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    source = json.loads((args.source / "manifest.json").read_text())
    assert source["complete"] and len(source["states"]) == 32
    for state in source["states"]:
        assert (
            sha(args.source / state["reference"]["path"])
            == state["reference"]["sha256"]
        )
    result = dict(
        schema="dense_development_ocr_v1",
        scope=__doc__,
        script_sha256=sha(Path(__file__)),
        source_sha256=sha(args.source / "manifest.json"),
        sampling_age_s=2.8,
        complete=False,
        started=time.time(),
        tesseract=subprocess.check_output(
            ["tesseract", "--version"], text=True, stderr=subprocess.STDOUT
        ),
        tessdata={
            name: sha(args.tessdata / (name + ".traineddata"))
            for name in ["eng", "chi_sim"]
        },
        preprocessing="Identical to calibration: source boxes; source-defined dark inversion; Lanczos x3; white border12; OEM1 PSM7 dpi300",
        limitations=[
            "Development groups only; not final test",
            "OCR is not human preference",
            "Code indentation is not part of nonspace exactness",
        ],
    )
    result["frame_admission"] = (
        "All groups retained with real output frame drops reported; each scored frame must exist at the exact frozen PTS and match both state markers. No smooth-30fps claim."
        if args.allow_output_frame_drops
        else "Require 180 output frames per clip before scoring"
    )
    save(args.output / "PROTOCOL.json", result)
    images = []
    if args.score_index:
        index = json.loads(args.score_index.read_text())
        assert index["source_sha256"] == result["source_sha256"]
        for row in index["images"]:
            assert sha(row["path"]) == row["sha256"]
        images = index["images"]
        result["index_sha256"] = sha(args.score_index)
    else:
        for state in source["states"]:
            path = args.source / state["reference"]["path"]
            images.append(
                dict(
                    id=state["id"] + "-clean",
                    path=str(path.resolve()),
                    sha256=sha(path),
                    group=state["group"],
                    category=state["category"],
                    state_id=state["state_id"],
                    config="clean",
                    method="reference",
                    source_age_s=None,
                )
            )
        if not args.clean_only:
            controls = json.loads((args.controls / "MANIFEST.json").read_text())
            assert controls["source_manifest_sha256"] == result["source_sha256"]
            result["controls_sha256"] = sha(args.controls / "MANIFEST.json")
            result["invalid_codec_runs"] = []
            outdir = args.output / "images"
            outdir.mkdir()
            for run in controls["runs"]:
                path = args.controls / run["path"]
                assert sha(path) == run["sha256"]
                if not run["validated"]:
                    result["invalid_codec_runs"].append(run["id"])
                    if not args.allow_output_frame_drops:
                        continue
                targets = [(2.8, run["state_ids"][0]), (5.8, run["state_ids"][1])]
                with av.open(str(path)) as container:
                    for frame in container.decode(video=0):
                        pts = float(frame.pts * frame.time_base)
                        if not targets or pts + 1e-6 < targets[0][0]:
                            continue
                        target, state_id = targets.pop(0)
                        assert pts - target < 0.05
                        if args.allow_output_frame_drops:
                            assert abs(pts - target) < 1e-5, (run["id"], target, pts)
                        state = next(
                            s for s in source["states"] if s["state_id"] == state_id
                        )
                        rgb = frame.to_ndarray(format="rgb24")
                        assert markers(rgb) == [state_id] * 2, (
                            run["id"],
                            pts,
                            markers(rgb),
                        )
                        name = run["id"] + f"-s{state_id:02d}"
                        low = outdir / (name + "-decoded.png")
                        Image.fromarray(rgb).save(low)
                        display = cv2.resize(
                            rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC
                        )
                        high = outdir / (name + "-display.png")
                        Image.fromarray(display).save(high)
                        images.append(
                            dict(
                                id=name,
                                path=str(high.resolve()),
                                sha256=sha(high),
                                decoded_path=str(low.resolve()),
                                decoded_sha256=sha(low),
                                height=run["height"],
                                group=state["group"],
                                category=state["category"],
                                state_id=state_id,
                                config=f"{run['height']}p-600k",
                                method="bicubic" if run["height"] == 720 else "native",
                                source_age_s=2.8,
                                pts_s=pts,
                                codec_frame_count_valid=run["validated"],
                                actual_output_frames=int(
                                    run["probe"]["streams"][0]["nb_read_frames"]
                                ),
                            )
                        )
                assert not targets
    save(
        args.output / "IMAGES.json",
        dict(source_sha256=result["source_sha256"], images=images),
    )
    if args.images_only:
        result.update(
            images=len(images),
            images_prepared=True,
            ocr_scored=False,
            finished=time.time(),
        )
        save(args.output / "PROTOCOL.json", result)
        return
    crops = args.output / "crops"
    crops.mkdir()
    tasks = []
    for item in images:
        state = next(s for s in source["states"] if s["state_id"] == item["state_id"])
        for label in regions(state):
            tasks.append(
                dict(
                    image_id=item["id"],
                    image=item["path"],
                    method=item["method"],
                    group=state["group"],
                    state_id=state["state_id"],
                    config=item["config"],
                    source_age_s=item["source_age_s"],
                    label=label,
                    dark=state["dark"],
                    tessdata=str(args.tessdata.resolve()),
                    crop=str(
                        (
                            crops / (item["id"] + "-" + label["label_id"] + ".png")
                        ).resolve()
                    ),
                )
            )
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for count, row in enumerate(executor.map(ocr, tasks), 1):
            state = next(
                s for s in source["states"] if s["state_id"] == row["state_id"]
            )
            row["category"] = state["category"]
            rows.append(row)
            if count % 100 == 0:
                save(
                    args.output / "PROGRESS.json", dict(scored=count, total=len(tasks))
                )
                print(count, "/", len(tasks), flush=True)
    save(args.output / "ROWS.json", rows)
    save(args.output / "SUMMARY.json", summarize(rows))
    save(
        args.output / "CATEGORY-SUMMARY.json",
        {
            category: summarize([r for r in rows if r["category"] == category])
            for category in sorted({r["category"] for r in rows})
        },
    )
    result.update(
        complete=True, images=len(images), ocr_regions=len(rows), finished=time.time()
    )
    save(args.output / "PROTOCOL.json", result)


if __name__ == "__main__":
    main()
