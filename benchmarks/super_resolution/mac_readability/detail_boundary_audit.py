"""Attribute detail replay regressions without selecting patches from OCR labels."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.super_resolution.mac_readability.quality import save, sha


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    a = p.parse_args()
    root = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    src = (
        a.project
        / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/reference"
    )
    states = {s["state_id"]: s for s in read(src / "manifest.json")["states"]}
    base = {
        (x["state_id"], x["label_id"]): x
        for x in read(root / "training-v2-selected-full-quality/ROWS.json")
        if x["unit"] == "line"
    }
    native = {
        (x["state_id"], x["label_id"]): x
        for x in read(root / "precodec-quality-v1/ROWS.json")
        if x["config"] == "1080p-420-no-codec"
    }
    result = []
    tiles = read(root / "detail-tiles-v1/RESULTS.json")
    candidates = read(root / "detail-quality-v1/ROWS.json")
    image_index = read(root / "detail-replay-v1/IMAGES.json")
    native_images = read(root / "precodec-images-v1/IMAGES.json")
    for method in sorted({x["method"] for x in candidates}):
        budget = int(method.split("-")[-1].removesuffix("bytes"))
        rows = []
        for image in image_index["images"]:
            if image["method"] != method:
                continue
            sid = image["state_id"]
            mask = np.zeros((1080, 1920), bool)
            total = 76
            for t in next(x for x in tiles["images"] if x["state_id"] == sid)["tiles"]:
                if total + t["wire_bytes"] > budget:
                    break
                mask[t["y"] : t["y"] + t["height"], t["x"] : t["x"] + t["width"]] = True
                total += t["wire_bytes"]
            path = Path(image["path"])
            assert sha(path) == image["sha256"]
            pixels = np.asarray(Image.open(path).convert("RGB"))
            ni = next(
                x
                for x in native_images["images"]
                if x["state_id"] == sid and x["config"] == "1080p-420-no-codec"
            )
            npth = Path(ni["path"])
            assert sha(npth) == ni["sha256"]
            npixels = np.asarray(Image.open(npth).convert("RGB"))
            for row in [
                r for r in candidates if r["state_id"] == sid and r["method"] == method
            ]:
                key = (sid, row["label_id"])
                label = next(
                    l for l in states[sid]["labels"] if l["label_id"] == row["label_id"]
                )
                x, y, w, h = label["xywh"]
                m = mask[y : y + h, x : x + w]
                fraction = float(m.mean())
                crop = pixels[y : y + h, x : x + w]
                ncrop = npixels[y : y + h, x : x + w]
                same = np.array_equal(crop, ncrop)
                if same:
                    assert row["text_predicted"] == native[key]["text_predicted"], (
                        "Identical crop OCR changed"
                    )
                rows.append(
                    dict(
                        state_id=sid,
                        label_id=row["label_id"],
                        category=row["category"],
                        group=row["group"],
                        patch_fraction=fraction,
                        exposure="full"
                        if fraction == 1
                        else "none"
                        if fraction == 0
                        else "mixed",
                        native_crop_identical=same,
                        base_edits=base[key]["edits"],
                        native_edits=native[key]["edits"],
                        patch_edits=row["edits"],
                        newly_wrong=base[key]["exact_nonspace"]
                        and not row["exact_nonspace"],
                        worse_than_both=row["edits"]
                        > max(base[key]["edits"], native[key]["edits"]),
                        source_text=row["text_gt"],
                        base_text=base[key]["text_predicted"],
                        native_text=native[key]["text_predicted"],
                        patch_text=row["text_predicted"],
                    )
                )
        new = [x for x in rows if x["newly_wrong"]]
        result.append(
            dict(
                method=method,
                lines=len(rows),
                newly_wrong=len(new),
                new_wrong_exposure=dict(Counter(x["exposure"] for x in new)),
                new_wrong_native_identical=sum(x["native_crop_identical"] for x in new),
                new_wrong_native_already_wrong=sum(x["native_edits"] > 0 for x in new),
                mixed_worse_than_both=sum(
                    x["exposure"] == "mixed" and x["worse_than_both"] for x in rows
                ),
                table_new_wrong=len([x for x in new if x["category"] == "table"]),
                table_new_wrong_native_identical=sum(
                    x["category"] == "table" and x["native_crop_identical"] for x in new
                ),
                rows=rows,
            )
        )
    save(
        root / "DETAIL-BOUNDARY-AUDIT.json",
        dict(
            complete=True,
            scope=__doc__,
            script_sha256=sha(Path(__file__)),
            results=result,
            conclusion="Identical-source crop failures cannot be attributed to tile seams. Mixed worse-than-both is consistent with a seam/rendering interaction, not definitive causation.",
            test_used=False,
        ),
    )
    for x in result:
        print({k: v for k, v in x.items() if k != "rows"})


if __name__ == "__main__":
    main()
