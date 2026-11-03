"""Correct one rotated-PDF annotation without changing its rendered pixels."""

import argparse
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from benchmarks.super_resolution.text_corpus.render import read
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def run(args):
    import pymupdf

    root = args.root.resolve()
    parent = root / "dataset-views-v2.json"
    data = read(parent)
    sources = read(root / "render-plan.json")["source_map"]
    changed = []
    for row in data["views"]:
        if row["kind"] not in ("paper", "specification"):
            continue
        labels = read(root / row["labels"])
        with pymupdf.open(root / sources[labels["source"]]["path"]) as doc:
            page = doc[labels["page"] - 1]
            if page.rotation == 0:
                continue
            meta = labels["metadata"]
            scale = meta["scale"]
            clip = pymupdf.Rect(meta["clip_points"])
            pix = page.get_pixmap(
                matrix=pymupdf.Matrix(scale, scale),
                clip=clip,
                colorspace=pymupdf.csRGB,
                alpha=False,
            )
            image = np.array(Image.open(root / row["hr"]).convert("RGB"))
            x0 = (1920 - pix.width) // 2
            y0 = 48
            rendered = np.frombuffer(pix.samples, np.uint8).reshape(
                pix.height, pix.width, 3
            )
            if not np.array_equal(
                image[y0 : y0 + pix.height, x0 : x0 + pix.width], rendered
            ):
                raise ValueError(
                    "Rotated source no longer reproduces the exact existing pixels"
                )
            words = []
            mask = np.zeros((1080, 1920), np.uint8)
            for a, b, c, d, text, block, line, index in page.get_text(
                "words", sort=True
            ):
                rect = pymupdf.Rect(a, b, c, d) * page.rotation_matrix
                if (
                    rect.x0 < clip.x0 + 0.25
                    or rect.x1 > clip.x1 - 0.25
                    or rect.y0 < clip.y0 + 0.25
                    or rect.y1 > clip.y1 - 0.25
                ):
                    continue
                x = x0 + rect.x0 * scale - pix.x
                y = y0 + rect.y0 * scale - pix.y
                w = rect.width * scale
                h = rect.height * scale
                if (
                    x < 0
                    or y < 0
                    or x + w > 1920
                    or y + h > 1080
                    or not text.strip()
                    or "\ufffd" in text
                ):
                    continue
                words.append(
                    dict(
                        text=text,
                        xywh=[round(v, 3) for v in [x, y, w, h]],
                        block=block,
                        line=line,
                        index=index,
                    )
                )
                mask[int(y) : math.ceil(y + h), int(x) : math.ceil(x + w)] = 1
            labels["words"] = words
            labels["characters"] = sum(len(w["text"]) for w in words)
            labels["metadata"].update(
                text_box_coverage=float(mask.mean()),
                rotation_applied_to_word_boxes=page.rotation,
                annotation_parent_sha256=row["labels_sha256"],
                rotation_pixel_replay_verified=True,
            )
            path = (root / row["labels"]).with_name("view-rotation-fixed.json")
            atomic_json(path, labels)
            changed.append(
                dict(
                    id=row["id"],
                    old_labels=row["labels"],
                    old_sha256=row["labels_sha256"],
                    new_labels=str(path.relative_to(root)),
                    new_sha256=sha256(path),
                    rotation=page.rotation,
                )
            )
            row.update(
                labels=str(path.relative_to(root)),
                labels_sha256=sha256(path),
                characters=labels["characters"],
            )
    result = dict(
        data,
        geometry_parent_sha256=sha256(parent),
        annotation_updates=changed,
        geometry_repair_script_sha256=sha256(Path(__file__)),
        geometry_revised=time.time(),
    )
    atomic_json(root / "dataset-views-v3.json", result)
    protocol = read(root / "evaluation-protocol/protocol.json")
    indexed = {r["id"]: r for r in result["views"]}
    for case in protocol["cases"]:
        row = indexed[case["view"]]
        if (
            row["labels_sha256"] != case["labels_sha256"]
            or row["hr_sha256"] != case["hr_sha256"]
        ):
            raise ValueError("Frozen evaluation case changed")
    atomic_json(
        root / "evaluation-protocol/source-compatibility.json",
        dict(
            protocol_sha256=sha256(root / "evaluation-protocol/protocol.json"),
            final_views_sha256=sha256(root / "dataset-views-v3.json"),
            cases_verified=len(protocol["cases"]),
            unchanged_fields=[
                "reference pixels",
                "source text",
                "crop geometry",
                "split",
                "source group",
            ],
            finished=time.time(),
        ),
    )
    print("ROTATION ANNOTATIONS CORRECTED", changed, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
