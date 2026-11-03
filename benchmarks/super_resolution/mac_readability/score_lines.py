"""Score hash-bound images with the already frozen dense-development line OCR.

No new OCR settings, checkpoint selection rules, or final-test access.
Completed outputs resume only when input, scorer and OCR dependencies match.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import time

import cv2

from benchmarks.super_resolution.mac_readability import quality


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tessdata", type=Path, required=True)
    p.add_argument("--frozen-ocr", type=Path, required=True)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    cv2.setNumThreads(1)
    source = json.loads((a.source / "manifest.json").read_text())
    index = json.loads(a.index.read_text())
    assert source["complete"] and index["source_sha256"] == quality.sha(
        a.source / "manifest.json"
    )
    frozen = json.loads(a.frozen_ocr.read_text())
    tessdata = {
        name: quality.sha(a.tessdata / (name + ".traineddata"))
        for name in ["eng", "chi_sim"]
    }
    version = subprocess.check_output(
        ["tesseract", "--version"], text=True, stderr=subprocess.STDOUT
    )
    assert tessdata == frozen["tessdata"] and version == frozen["tesseract"]
    binding = dict(
        index_sha256=quality.sha(a.index),
        source_sha256=index["source_sha256"],
        script_sha256=quality.sha(Path(__file__)),
        ocr_sha256=quality.sha(Path(quality.__file__)),
        frozen_ocr_sha256=quality.sha(a.frozen_ocr),
        tessdata=tessdata,
        tesseract=version,
    )
    a.output.mkdir(parents=True, exist_ok=True)
    protocol = a.output / "PROTOCOL.json"
    if protocol.exists():
        prior = json.loads(protocol.read_text())
        assert prior["binding"] == binding
        if prior["complete"]:
            return
    record = dict(binding=binding, scope=__doc__, complete=False, started=time.time())
    quality.save(protocol, record)
    crops = a.output / "crops"
    crops.mkdir(exist_ok=True)
    tasks = []
    seen = set()
    for item in index["images"]:
        assert item["id"] not in seen
        seen.add(item["id"])
        image = Path(item["path"])
        image = image if image.is_absolute() else a.index.parent / image
        assert quality.sha(image) == item["sha256"]
        state = next(s for s in source["states"] if s["state_id"] == item["state_id"])
        assert (
            item["group"] == state["group"]
            and quality.sha(a.source / state["reference"]["path"])
            == state["reference"]["sha256"]
        )
        for label in state["labels"]:
            assert label["unit"] == "line"
            tasks.append(
                dict(
                    image_id=item["id"],
                    image=str(image.resolve()),
                    method=item["method"],
                    group=item["group"],
                    state_id=item["state_id"],
                    config=item["config"],
                    source_age_s=item.get("source_age_s"),
                    label=label,
                    dark=state["dark"],
                    tessdata=str(a.tessdata.resolve()),
                    crop=str(
                        (
                            crops / (item["id"] + "-" + label["label_id"] + ".png")
                        ).resolve()
                    ),
                )
            )
    rows = []
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, row in enumerate(pool.map(quality.ocr, tasks), 1):
            row["category"] = next(
                s["category"]
                for s in source["states"]
                if s["state_id"] == row["state_id"]
            )
            rows.append(row)
            if i % 100 == 0:
                quality.save(
                    a.output / "PROGRESS.json", dict(scored=i, total=len(tasks))
                )
                print(i, "/", len(tasks), flush=True)
    quality.save(a.output / "ROWS.json", rows)
    quality.save(a.output / "SUMMARY.json", quality.summarize(rows))
    quality.save(
        a.output / "CATEGORY-SUMMARY.json",
        {
            c: quality.summarize([r for r in rows if r["category"] == c])
            for c in sorted({r["category"] for r in rows})
        },
    )
    quality.save(
        a.output / "FONT-SUMMARY.json",
        {
            str(f): quality.summarize([r for r in rows if r["font_pixels"] == f])
            for f in sorted({r["font_pixels"] for r in rows})
        },
    )
    record.update(
        complete=True,
        finished=time.time(),
        images=len(index["images"]),
        ocr_regions=len(rows),
    )
    quality.save(protocol, record)


if __name__ == "__main__":
    main()
