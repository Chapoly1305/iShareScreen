"""Decode actualMac strips; validate source pixels and retain byte-budget scope."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from benchmarks.super_resolution.mac_readability.detail_replay import (
    rgb420,
    shared_link,
)
from benchmarks.super_resolution.mac_readability.prepare_precodec import planar_nv12
from benchmarks.super_resolution.mac_readability.strip_transport import (
    StripReceiver,
    wire_bytes,
)
from benchmarks.super_resolution.mac_readability.quality import save, sha


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    a = p.parse_args()
    r = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    source = r / "detail-strips-v1"
    out = r / "strip-replay-v1"
    out.mkdir(exist_ok=False)
    sender = read(source / "RESULTS.json")
    assert sender["complete"] and len(sender["images"]) == 32
    for name, value in sender["scripts"].items():
        assert sha(Path(__file__).with_name(name)) == value
    pre = read(r / "precodec-v1/MANIFEST.json")
    assert sha(r / "precodec-v1/MANIFEST.json") == sender["controls_sha256"]
    base = read(r / "TRAINING-V2-SELECTED-IMAGES.json")
    assert base["source_sha256"] == sender["source_sha256"]
    definition = read(
        a.project
        / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/reference/manifest.json"
    )
    result = dict(
        complete=False,
        scope=__doc__,
        sender_sha256=sha(source / "RESULTS.json"),
        source_sha256=sender["source_sha256"],
        script_sha256=sha(Path(__file__)),
        base_index_sha256=sha(r / "TRAINING-V2-SELECTED-IMAGES.json"),
        caveats=[
            "Quality is frozensteadybase+bytebudget, not a wallclockdeadline image",
            "Linkreplay is a model, not physicalnetwork",
            "NoOCR/category-guided selection; finaltest untouched",
        ],
        images=[],
        measurements=[],
        test_used=False,
    )
    save(out / "RESULTS.json", result)
    for row in sender["images"]:
        orig = next(
            x
            for x in pre["images"]
            if x["height"] == 1080 and x["state_id"] == row["state_id"]
        )
        path = r / "precodec-v1" / orig["path"]
        assert sha(path) == row["raw_sha256"]
        native = rgb420(planar_nv12(path.read_bytes(), 1920, 1080))
        b = next(x for x in base["images"] if x["state_id"] == row["state_id"])
        path = Path(b["path"])
        assert sha(path) == b["sha256"]
        pixels = np.asarray(Image.open(path).convert("RGB"))
        receiver = StripReceiver(1920, 1080)
        receiver.begin(row["state_id"], bytes.fromhex(row["identity"]))
        total = wire_bytes(32)
        i = 0
        state = next(
            s for s in definition["states"] if s["state_id"] == row["state_id"]
        )
        for budget in sender["parameters"]["quality_budgets_wire_bytes"] + [
            row["wire_bytes"]
        ]:
            while (
                i < len(row["tiles"])
                and total + row["tiles"][i]["wire_bytes"] <= budget
            ):
                t = row["tiles"][i]
                path = source / t["path"]
                assert sha(path) == t["sha256"]
                assert receiver.accept(path.read_bytes())
                total += t["wire_bytes"]
                i += 1
            decoded = rgb420(receiver.planar())
            assert np.array_equal(decoded[receiver.mask], native[receiver.mask])
            output = pixels.copy()
            output[receiver.mask] = decoded[receiver.mask]
            if budget == row["wire_bytes"]:
                assert receiver.mask.all() and np.array_equal(output, native)
                continue
            method = f"source-strips-{budget}bytes"
            name = f"s{row['state_id']:02d}-{method}.png"
            dest = out / name
            Image.fromarray(output).save(dest)
            result["images"].append(
                dict(
                    id=name[:-4],
                    state_id=row["state_id"],
                    group=row["group"],
                    category=row["category"],
                    config="720p-600k-plus-strip-budget",
                    source_age_s=None,
                    method=method,
                    path=str(dest.resolve()),
                    sha256=sha(dest),
                )
            )
            exposures = []
            for label in state["labels"]:
                x, y, w, h = label["xywh"]
                fraction = float(receiver.mask[y : y + h, x : x + w].mean())
                exposures.append(dict(label_id=label["label_id"], fraction=fraction))
            result["measurements"].append(
                dict(
                    state_id=row["state_id"],
                    budget=budget,
                    wire_bytes=total,
                    strips=i,
                    refined_pixels=int(receiver.mask.sum()),
                    line_exposure=exposures,
                )
            )
        print(row["state_id"], flush=True)
    codec = read(
        a.project
        / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/codec-v1/MANIFEST.json"
    )
    result["link_replays"] = [
        shared_link(
            run,
            [
                next(x for x in sender["images"] if x["state_id"] == sid)
                for sid in run["state_ids"]
            ],
        )
        for run in codec["runs"]
        if run["height"] == 720
    ]
    result["complete"] = True
    save(out / "RESULTS.json", result)
    save(
        out / "IMAGES.json",
        dict(source_sha256=result["source_sha256"], images=result["images"]),
    )


if __name__ == "__main__":
    main()
