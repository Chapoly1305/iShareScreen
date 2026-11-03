"""Fixed global source/SR interpolation; no OCR-dependent routing or new text.

Uses the same frozen steady SR output and exact native420 source carried by the
measured supplement protocol. Source weights are fixed before OCR scoring.
This offline quality experiment does not establish live composition latency.
"""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.mac_readability.quality import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    r = a.root
    out = r / "source-mix-v1"
    out.mkdir(exist_ok=False)
    cv2.setNumThreads(1)
    base = json.loads((r / "TRAINING-V2-SELECTED-IMAGES.json").read_text())
    native = json.loads((r / "precodec-images-v1/IMAGES.json").read_text())
    source = {
        x["state_id"]: x
        for x in native["images"]
        if x["config"] == "1080p-420-no-codec"
    }
    assert base["source_sha256"] == native["source_sha256"] and len(source) == 32
    record = dict(
        complete=False,
        scope=__doc__,
        source_sha256=base["source_sha256"],
        source_weights=[0.25, 0.5, 0.75],
        rule="cv2.addWeighted(native420,alpha,frozenbestSR,1-alpha,0), same alpha for everypixel andsource; no category/OCR routing",
        script_sha256=sha(Path(__file__)),
        opencv=cv2.__version__,
        base_index_sha256=sha(r / "TRAINING-V2-SELECTED-IMAGES.json"),
        source_index_sha256=sha(r / "precodec-images-v1/IMAGES.json"),
        supplement_cost_evidence_sha256=sha(r / "COMPONENT-COSTS.json"),
        images=[],
        test_used=False,
    )
    save(out / "RESULTS.json", record)
    for row in base["images"]:
        p = Path(row["path"])
        assert sha(p) == row["sha256"]
        sr = np.asarray(Image.open(p).convert("RGB"))
        source_row = source[row["state_id"]]
        p = Path(source_row["path"])
        assert sha(p) == source_row["sha256"]
        original = np.asarray(Image.open(p).convert("RGB"))
        assert sr.shape == original.shape == (1080, 1920, 3)
        for alpha in record["source_weights"]:
            method = f"source-mix-{round(alpha * 100):02d}"
            name = f"s{row['state_id']:02d}-{method}.png"
            path = out / name
            start = time.perf_counter()
            pixels = cv2.addWeighted(original, alpha, sr, 1 - alpha, 0)
            elapsed = (time.perf_counter() - start) * 1000
            Image.fromarray(pixels).save(path)
            record["images"].append(
                dict(
                    id=name[:-4],
                    method=method,
                    config="720p-600k-plus-full-source",
                    source_age_s=2.8,
                    state_id=row["state_id"],
                    group=row["group"],
                    path=str(path.resolve()),
                    sha256=sha(path),
                    source_weight=alpha,
                    source_image_sha256=source_row["sha256"],
                    base_image_sha256=row["sha256"],
                    h100_cpu_mix_ms=elapsed,
                )
            )
        print(row["state_id"], flush=True)
    record["complete"] = True
    save(out / "RESULTS.json", record)
    save(
        out / "IMAGES.json",
        dict(source_sha256=record["source_sha256"], images=record["images"]),
    )


if __name__ == "__main__":
    main()
