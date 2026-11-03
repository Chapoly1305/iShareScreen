"""Pack audited RGB pairs and text boxes for fresh random crops on a GPU host."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path):
    return json.loads(path.read_text())


def run(args):
    root = args.root.resolve()
    source = root / args.pairs
    data = read(source)
    if not data["complete"] or data["count"] != 10000:
        raise ValueError("Need the complete 10,000-pair corpus")
    views = read(root / args.views)
    if data["views_sha256"] != sha256(root / args.views):
        raise ValueError("View/codec identity mismatch")
    audit = read(root / args.audit)
    if not audit["passed"] or audit["views_sha256"] != sha256(root / args.views):
        raise ValueError("Full image/label/leakage audit has not passed")
    aliases = views.get("group_aliases", {})
    groups = {}
    pixels = set()
    rows = sorted(data["pairs"], key=lambda r: r["id"])
    for r in rows:
        group = aliases.get(r["source_group"], r["source_group"])
        if group in groups and groups[group] != r["split"]:
            raise ValueError("Source group crosses a split")
        groups[group] = r["split"]
        if r["hr_rgb_sha256"] in pixels:
            raise ValueError("Duplicate HR pixels")
        pixels.add(r["hr_rgb_sha256"])
    out = root / args.output
    if out.exists():
        raise ValueError("Packing requires a fresh output directory")
    out.mkdir()
    (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    arrays = {
        role: np.lib.format.open_memmap(
            out / (role + ".npy"),
            mode="w+",
            dtype=np.uint8,
            shape=(len(rows), height, width, 3),
        )
        for role, height, width in [("lr", 540, 960), ("hr", 1080, 1920)]
    }
    boxes = []
    offsets = [0]
    for index, row in enumerate(rows):
        for role in ["lr", "hr"]:
            path = root / row[role]
            if sha256(path) != row[role + "_sha256"]:
                raise ValueError("Changed pair image: " + str(path))
            rgb = np.array(Image.open(path).convert("RGB"))
            if hashlib.sha256(rgb.tobytes()).hexdigest() != row[role + "_rgb_sha256"]:
                raise ValueError("Decoded pixels differ")
            if rgb.shape != arrays[role].shape[1:]:
                raise ValueError("Wrong packed pair geometry")
            arrays[role][index] = rgb
        if sha256(root / row["labels"]) != row["labels_sha256"]:
            raise ValueError("Text geometry labels changed")
        labels = read(root / row["labels"])
        if labels["hr_rgb_sha256"] != row["hr_rgb_sha256"]:
            raise ValueError("Labels do not describe these pixels")
        values = []
        for word in labels["words"]:
            x, y, w, h = word["xywh"]
            weight = (
                3 if any(c.isdigit() or not c.isalnum() for c in word["text"]) else 1
            )
            values.append(
                [int(x), int(y), int(np.ceil(x + w)), int(np.ceil(y + h)), weight]
            )
        boxes.append(np.asarray(values, dtype=np.uint16))
        offsets.append(offsets[-1] + len(values))
        if (index + 1) % 100 == 0:
            for array in arrays.values():
                array.flush()
            atomic_json(
                out / "status.json",
                dict(
                    phase="packing",
                    written=index + 1,
                    total=len(rows),
                    pid=__import__("os").getpid(),
                    updated=time.time(),
                ),
            )
            print("PACK", index + 1, "/", len(rows), flush=True)
    for array in arrays.values():
        array.flush()
    np.save(out / "boxes.npy", np.concatenate(boxes))
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    identities = {
        name: dict(
            path=name + ".npy",
            sha256=sha256(out / (name + ".npy")),
            shape=list(np.load(out / (name + ".npy"), mmap_mode="r").shape),
        )
        for name in ["lr", "hr", "boxes", "offsets"]
    }
    manifest = dict(
        schema="isharescreen_text_packed_v1",
        complete=True,
        count=len(rows),
        source_pairs_sha256=sha256(source),
        views_sha256=sha256(root / args.views),
        quality_audit_sha256=sha256(root / args.audit),
        files=identities,
        rows=rows,
        group_aliases=aliases,
        scope="New source-group-split corpus only; legacy video pairs are retained separately, not silently mixed into this first scratch experiment.",
        finished=time.time(),
    )
    atomic_json(out / "manifest.json", manifest)
    atomic_json(
        out / "COMPLETED.json", dict(manifest_sha256=sha256(out / "manifest.json"))
    )
    print("PACK COMPLETE", len(rows), "views", offsets[-1], "text boxes", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--pairs", default="hevc-v2/pairs.json")
    p.add_argument("--views", default="dataset-views-v3.json")
    p.add_argument("--audit", default="quality-audit-v3.json")
    p.add_argument("--output", default="packed-v1")
    run(p.parse_args())


if __name__ == "__main__":
    main()
