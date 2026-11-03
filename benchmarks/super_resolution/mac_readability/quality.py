"""Source-known Mac text diagnostics with frozen OCR and PTS-based sampling.

These eight calibration states are not independent final-test documents.
The hardware-codec controls are not Screen Sharing or a bandwidth-limited WAN.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

import av
import cv2
import numpy as np
from PIL import Image, ImageOps

from benchmarks.super_resolution.base_text_quality import distance, normalized
from benchmarks.super_resolution.metrics import metrics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def markers(rgb):
    # Read centers of the two redundant eight-bit state markers, after resizing
    # only for this alignment check. Quality always uses the original pixels.
    pixels = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_NEAREST)
    return [
        sum(
            (float(pixels[y, (440 + b * 10 + 4) * 2].mean()) > 128) << b
            for b in range(8)
        )
        for y in (24, 1048)
    ]


def labels(state):
    for i, label in enumerate(state["labels"]):
        yield dict(label, unit="line", label_id=f"line-{i:02d}")
    if state["group"] != "table":
        return
    # Menlo is monospaced; source spaces define the cell boundaries, independently
    # of OCR output. Headers containing a single space remain one cell.
    for i, label in enumerate(state["labels"][:-4]):
        x, y, w, h = label["xywh"]
        advance = (label["rect_points"][2] - 4) * 2 / len(label["text"])
        for j, match in enumerate(
            re.finditer(r"\S(?:.*?\S)?(?= {2,}|$)", label["text"])
        ):
            yield dict(
                label,
                unit="cell",
                label_id=f"cell-{i:02d}-{j:02d}",
                text=match.group(),
                xywh=[
                    round(x + 4 + match.start() * advance - 2),
                    y,
                    round(len(match.group()) * advance + 4),
                    h,
                ],
            )


def ocr(task):
    label = task["label"]
    x, y, w, h = label["xywh"]
    with Image.open(task["image"]) as image:
        crop = image.convert("RGB").crop((x, y, x + w, y + h))
    if task["dark"]:
        crop = ImageOps.invert(crop)
    crop = ImageOps.expand(
        crop.resize((3 * w, 3 * h), Image.Resampling.LANCZOS), 12, "white"
    )
    path = Path(task["crop"])
    crop.save(path)
    language = "chi_sim+eng" if any(ord(c) > 127 for c in label["text"]) else "eng"
    predicted = subprocess.check_output(
        [
            "tesseract",
            str(path),
            "stdout",
            "--tessdata-dir",
            task["tessdata"],
            "-l",
            language,
            "--oem",
            "1",
            "--psm",
            "7",
            "--dpi",
            "300",
        ],
        text=True,
        stderr=subprocess.PIPE,
        env=dict(os.environ, OMP_THREAD_LIMIT="1"),
    )
    a, b = normalized(label["text"]), normalized(predicted)
    digits = lambda s: "".join(c for c in s if c.isdigit())
    punct = lambda s: "".join(c for c in s if not c.isalnum())
    return {
        k: task[k]
        for k in ["image_id", "method", "group", "state_id", "config", "source_age_s"]
    } | dict(
        label_id=label["label_id"],
        unit=label["unit"],
        font_pixels=label["font_points"] * 2,
        language=language,
        text_gt=label["text"],
        text_predicted=predicted,
        characters=len(a),
        edits=distance(a, b),
        exact_nonspace=a == b,
        digits=len(digits(a)),
        digit_edits=distance(digits(a), digits(b)),
        punctuation=len(punct(a)),
        punctuation_edits=distance(punct(a), punct(b)),
        crop=task["crop"],
        crop_sha256=sha(path),
    )


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["config"], row["method"], row["unit"], row["source_age_s"]].append(
            row
        )
    result = []
    for (config, method, unit, age), values in groups.items():
        sources = []
        for group in sorted({r["group"] for r in values}):
            subset = [r for r in values if r["group"] == group]
            edits = sum(r["edits"] for r in subset)
            n = sum(r["characters"] for r in subset)
            sources.append(dict(group=group, edits=edits, characters=n, cer=edits / n))
        result.append(
            dict(
                config=config,
                method=method,
                unit=unit,
                source_age_s=age,
                macro_cer=float(np.mean([g["cer"] for g in sources])),
                groups=sources,
                pooled_cer=sum(r["edits"] for r in values)
                / sum(r["characters"] for r in values),
                exact_units=sum(r["exact_nonspace"] for r in values),
                units=len(values),
                digit_edits=sum(r["digit_edits"] for r in values),
                digits=sum(r["digits"] for r in values),
                punctuation_edits=sum(r["punctuation_edits"] for r in values),
                punctuation=sum(r["punctuation"] for r in values),
            )
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--original-source", type=Path, required=True)
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tessdata", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--images-only", action="store_true")
    p.add_argument(
        "--score-index",
        type=Path,
        help="Score a separately generated, hash-bound reconstruction index",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    source = json.loads((args.source / "manifest.json").read_text())
    old = json.loads((args.original_source / "manifest.json").read_text())
    controls = json.loads((args.controls / "MANIFEST.json").read_text())
    assert controls["source_manifest_sha256"] == sha(
        args.original_source / "manifest.json"
    )
    corrections = []
    for state, prior in zip(source["states"], old["states"]):
        assert (
            state["id"] == prior["id"] and state["labels_state_id"] == state["state_id"]
        )
        assert (
            sha(args.source / state["reference"]["path"])
            == state["reference"]["sha256"]
        )
        assert (
            sha(args.original_source / prior["reference"]["path"])
            == prior["reference"]["sha256"]
        )
        assert np.array_equal(
            np.array(Image.open(args.source / state["reference"]["path"])),
            np.array(Image.open(args.original_source / prior["reference"]["path"])),
        )
        corrections.append(
            dict(
                id=state["id"],
                pixels_identical=True,
                labels_state_id=state["labels_state_id"],
            )
        )
    frozen = dict(
        schema="mac_readability_ocr_v1",
        scope=__doc__,
        script_sha256=sha(Path(__file__)),
        started=time.time(),
        source_sha256=sha(args.source / "manifest.json"),
        original_source_sha256=sha(args.original_source / "manifest.json"),
        label_correction_audit=corrections,
        controls_manifest_sha256=sha(args.controls / "MANIFEST.json"),
        tesseract=subprocess.check_output(
            ["tesseract", "--version"], text=True, stderr=subprocess.STDOUT
        ),
        tessdata={
            name: sha(args.tessdata / (name + ".traineddata"))
            for name in ["eng", "chi_sim"]
        },
        preprocessing="Fixed source-defined crop, invert only source-defined dark scenes, Lanczos x3, white border 12px; OEM1 PSM7 dpi300. Case-sensitive, no whitespace in primary CER.",
        sampling="PTS at or immediately after state-relative 0.2, 1.0 and 2.8 seconds; duplicated state markers must agree.",
        aggregation="Pool characters within each content group, then equal-weight the four groups. Cells separate; digit/punctuation subsequence edits separate.",
        limitations=[
            "Eight calibration states, not independent final-test documents",
            "Code indentation and case-only errors not separately assessed yet",
            "OCR is not human preference",
            "Retain invalid frame-count control as diagnostic only",
        ],
        complete=False,
    )
    save(args.output / "PROTOCOL.json", frozen)
    images = []
    if args.score_index:
        index = json.loads(args.score_index.read_text())
        assert index["source_sha256"] == frozen["source_sha256"]
        images = index["images"]
        for row in images:
            assert sha(row["path"]) == row["sha256"]
        frozen["input_index_sha256"] = sha(args.score_index)
        save(args.output / "PROTOCOL.json", frozen)
    else:
        image_root = args.output / "images"
        image_root.mkdir()
        for state in source["states"]:
            path = args.source / state["reference"]["path"]
            images.append(
                dict(
                    id=state["id"] + "-clean",
                    path=str(path.resolve()),
                    sha256=sha(path),
                    state_id=state["state_id"],
                    group=state["group"],
                    config="clean",
                    method="reference",
                    source_age_s=None,
                )
            )
        for run in controls["runs"]:
            path = args.controls / run["path"]
            assert sha(path) == run["sha256"]
            targets = [(v * 3 + age, v, age) for v in [0, 1] for age in [0.2, 1.0, 2.8]]
            selected = []
            with av.open(str(path)) as container:
                for frame in container.decode(video=0):
                    pts = float(frame.pts * frame.time_base)
                    while targets and pts + 1e-6 >= targets[0][0]:
                        target, v, age = targets.pop(0)
                        if pts - target > 0.05:
                            raise RuntimeError(f"Sample PTS gap: {path} {target} {pts}")
                        rgb = frame.to_ndarray(format="rgb24")
                        state = next(
                            s
                            for s in source["states"]
                            if s["id"] == f"{run['group']}-{v}"
                        )
                        found = markers(rgb)
                        if found != [state["state_id"]] * 2:
                            raise RuntimeError(f"Misaligned frame {path} {pts} {found}")
                        name = f"{path.stem}-s{v}-age{age:.1f}"
                        low = image_root / (name + "-decoded.png")
                        Image.fromarray(rgb).save(low)
                        output = image_root / (name + "-display.png")
                        display = cv2.resize(
                            rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC
                        )
                        Image.fromarray(display).save(output)
                        quality = metrics(
                            np.asarray(
                                Image.open(args.source / state["reference"]["path"])
                            ),
                            display,
                        )
                        quality = {
                            k: float(v) if np.isfinite(v) else None
                            for k, v in quality.items()
                        }
                        images.append(
                            dict(
                                id=name,
                                path=str(output.resolve()),
                                sha256=sha(output),
                                decoded_path=str(low.resolve()),
                                decoded_sha256=sha(low),
                                height=run["height"],
                                state_id=state["state_id"],
                                group=state["group"],
                                source_age_s=age,
                                pts_s=pts,
                                config=f"{run['height']}p-{run['requested_kbps']}k",
                                method="bicubic" if run["height"] == 720 else "native",
                                codec_frame_count_valid=run["validated"],
                                quality=quality,
                            )
                        )
            if targets:
                raise RuntimeError(f"Missing samples: {path}")
            print(path.name, "sampled 6 aligned frames", flush=True)
    save(
        args.output / "IMAGES.json",
        dict(source_sha256=frozen["source_sha256"], images=images),
    )
    if args.images_only:
        return
    crops = args.output / "crops"
    crops.mkdir()
    tasks = []
    for image in images:
        state = next(s for s in source["states"] if s["state_id"] == image["state_id"])
        for label in labels(state):
            tasks.append(
                dict(
                    image_id=image["id"],
                    image=image["path"],
                    method=image["method"],
                    group=image["group"],
                    state_id=image["state_id"],
                    config=image["config"],
                    source_age_s=image["source_age_s"],
                    label=label,
                    dark=state["dark"],
                    tessdata=str(args.tessdata.resolve()),
                    crop=str(
                        (
                            crops / (image["id"] + "-" + label["label_id"] + ".png")
                        ).resolve()
                    ),
                )
            )
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, row in enumerate(executor.map(ocr, tasks), 1):
            rows.append(row)
            if i % 100 == 0:
                save(args.output / "PROGRESS.json", dict(scored=i, total=len(tasks)))
                print("OCR", i, "/", len(tasks), flush=True)
    save(args.output / "ROWS.json", rows)
    save(args.output / "SUMMARY.json", summarize(rows))
    frozen.update(
        complete=True, finished=time.time(), rows=len(rows), images=len(images)
    )
    save(args.output / "PROTOCOL.json", frozen)


if __name__ == "__main__":
    main()
