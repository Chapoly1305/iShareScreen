"""Freeze source-derived English line regions before model evaluation.

Regions are selected by source geometry and deterministic hashes, never by a
model's prediction or OCR success. The clean-reference OCR floor is retained,
including failures. OCR output is not used as ground truth.
"""

import argparse
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

from PIL import Image, ImageOps

from benchmarks.super_resolution.base_text_quality import distance, normalized
from benchmarks.super_resolution.text_corpus.render import (
    read,
    round_robin,
    stable_order,
)
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256
from benchmarks.super_resolution.text_corpus.runtime_paths import tessdata


def regions(labels):
    lines = defaultdict(list)
    for w in labels["words"]:
        if "block" in w:
            key = (w["block"], w["line"])
        else:
            key = round(w["xywh"][1], 1)
        lines[key].append(w)
    result = []
    for words in lines.values():
        words = sorted(words, key=lambda w: w["xywh"][0])
        text = " ".join(w["text"] for w in words)
        compact = normalized(text)
        if (
            not 30 <= len(compact) <= 160
            or sum(ord(c) < 128 for c in compact) / len(compact) < 0.98
        ):
            continue
        x = min(w["xywh"][0] for w in words)
        y = min(w["xywh"][1] for w in words)
        right = max(w["xywh"][0] + w["xywh"][2] for w in words)
        bottom = max(w["xywh"][1] + w["xywh"][3] for w in words)
        if bottom - y > 55 or bottom - y < 9 or right - x < 180:
            continue
        left = max(0, math.floor(x) - 4)
        top = max(0, math.floor(y) - 3)
        r = min(1920, math.ceil(right) + 4)
        b = min(1080, math.ceil(bottom) + 3)
        result.append(
            dict(
                text=text,
                xywh=[left, top, r - left, b - top],
                characters=len(compact),
                digits=sum(c.isdigit() for c in compact),
                punctuation=sum(not c.isalnum() for c in compact),
                font_box_height=bottom - y,
            )
        )
    return sorted(result, key=stable_order)


def score_crop(path, text, psm=7):
    import os

    predicted = subprocess.check_output(
        [
            "tesseract",
            str(path),
            "stdout",
            "-l",
            "eng",
            "--oem",
            "1",
            "--psm",
            str(psm),
            "--dpi",
            "300",
        ],
        stderr=subprocess.PIPE,
        text=True,
        env=dict(os.environ, OMP_THREAD_LIMIT="1"),
    )
    a, b = normalized(text), normalized(predicted)
    da = "".join(c for c in a if c.isdigit())
    db = "".join(c for c in b if c.isdigit())
    pa = "".join(c for c in a if not c.isalnum())
    pb = "".join(c for c in b if not c.isalnum())
    return dict(
        predicted=predicted,
        edit_distance=distance(a, b),
        characters=len(a),
        cer=distance(a, b) / len(a),
        exact=a == b,
        digits=len(da),
        digit_edits=distance(da, db),
        punctuation=len(pa),
        punctuation_edits=distance(pa, pb),
    )


def prepare_crop(image, xywh, path):
    x, y, w, h = xywh
    crop = image.convert("RGB").crop((x, y, x + w, y + h))
    ImageOps.expand(
        crop.resize((w * 3, h * 3), Image.Resampling.LANCZOS), border=12, fill="white"
    ).save(path)


def run(args):
    root = args.root.resolve()
    data = read(root / "dataset-views-v2.json")
    out = root / "evaluation-protocol"
    out.mkdir(exist_ok=False)
    targets = {
        "validation": dict(paper=32, code=16, web=8, table=4, terminal=4),
        "test": dict(paper=48, code=24, web=8, table=8, terminal=8),
    }
    selected = []
    skipped = []
    aliases = data["group_aliases"]
    for split, counts in targets.items():
        for kind, count in counts.items():
            grouped = defaultdict(list)
            for view in sorted(data["views"], key=lambda v: stable_order(v["id"])):
                if view["split"] == split and view["kind"] == kind:
                    grouped[
                        aliases.get(view["source_group"], view["source_group"])
                    ].append(view)
            available = round_robin(
                [grouped[k] for k in sorted(grouped, key=stable_order)]
            )
            accepted = 0
            for view in available:
                labels = read(root / view["labels"])
                choices = regions(labels)
                if not choices:
                    skipped.append(
                        dict(
                            id=view["id"],
                            reason="No eligible source-known English single line",
                        )
                    )
                    continue
                region = choices[0]
                case = dict(
                    id=view["id"] + "-line",
                    view=view["id"],
                    split=split,
                    kind=kind,
                    source_group=aliases.get(
                        view["source_group"], view["source_group"]
                    ),
                    hr=view["hr"],
                    hr_sha256=view["hr_sha256"],
                    labels=view["labels"],
                    labels_sha256=view["labels_sha256"],
                    **region,
                )
                selected.append(case)
                accepted += 1
                if accepted == count:
                    break
            if accepted != count:
                raise ValueError(
                    f"Insufficient eligible {split}/{kind} lines: {accepted}/{count}"
                )

    def reference(case):
        image = root / case["hr"]
        if sha256(image) != case["hr_sha256"]:
            raise ValueError("Clean reference changed")
        path = out / (case["id"] + "-clean.png")
        prepare_crop(Image.open(image), case["xywh"], path)
        return dict(
            case,
            clean_crop=str(path.relative_to(root)),
            clean_crop_sha256=sha256(path),
            clean_ocr=score_crop(path, case["text"]),
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        cases = list(pool.map(reference, selected))
    protocol = dict(
        schema="isharescreen_source_text_ocr_protocol_v1",
        views_sha256=sha256(root / "dataset-views-v2.json"),
        script_sha256=sha256(Path(__file__)),
        cases=cases,
        skipped=skipped,
        selection="One deterministic source-known English line per selected view; source-group round-robin and content-kind quotas. Clean OCR success never filters a case.",
        scope="64 validation and 96 test line regions, not full-screen/full-video OCR or human readability. Specification families are in training; unseen-specification generalization is not established.",
        preprocessing="FHD line crop with fixed small margin, Lanczos x3, 12px white border; Tesseract eng/OEM1/PSM7/dpi300. Non-whitespace CER retains case, digits and punctuation.",
        tesseract=subprocess.check_output(
            ["tesseract", "--version"], stderr=subprocess.STDOUT, text=True
        ),
        tessdata_sha256=sha256(tessdata() / "eng.traineddata"),
        counts=dict(Counter(c["split"] for c in cases)),
        finished=time.time(),
    )
    atomic_json(out / "protocol.json", protocol)
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    print(
        "FROZEN OCR PROTOCOL",
        protocol["counts"],
        "clean mean CER",
        sum(c["clean_ocr"]["cer"] for c in cases) / len(cases),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=2)
    run(p.parse_args())


if __name__ == "__main__":
    main()
