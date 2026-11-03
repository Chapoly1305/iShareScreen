"""Create a source-bound PDF-only screening subset from the frozen mixed pack."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def run(root: Path) -> None:
    source = root / "packed-pilot-v1"
    manifest_path = source / "manifest.json"
    receipt = json.loads((source / "COMPLETED.json").read_text())
    if sha256(manifest_path) != receipt["manifest_sha256"]:
        raise ValueError("Source packed receipt differs")
    manifest = json.loads(manifest_path.read_text())
    if manifest["training_views"] != 1637 or manifest["validation_views"] != 36:
        raise ValueError("Unexpected frozen source mixture")
    out = root / "packed-pdf-screen-v1"
    out.mkdir(exist_ok=False)
    selected = [
        i
        for i, row in enumerate(manifest["rows"])
        if row["split"] == "validation"
        or (row["split"] == "train" and row["kind"] == "paper")
    ]
    rows = [manifest["rows"][i] for i in selected]
    if len(rows) != 1073 or sum(row["split"] == "train" for row in rows) != 1037:
        raise ValueError("Wrong screening subset")
    arrays = {
        name: np.load(source / spec["path"], mmap_mode="r", allow_pickle=False)
        for name, spec in manifest["files"].items()
    }
    low = np.lib.format.open_memmap(
        out / "lr.npy", mode="w+", dtype=np.uint8, shape=(len(rows), 540, 960, 3)
    )
    high = np.lib.format.open_memmap(
        out / "hr.npy", mode="w+", dtype=np.uint8, shape=(len(rows), 1080, 1920, 3)
    )
    boxes = []
    offsets = [0]
    for position, index in enumerate(selected):
        low[position] = arrays["lr"][index]
        high[position] = arrays["hr"][index]
        start, end = arrays["offsets"][index : index + 2]
        box = np.asarray(arrays["boxes"][start:end])
        boxes.append(box)
        offsets.append(offsets[-1] + len(box))
        if (position + 1) % 100 == 0:
            low.flush()
            high.flush()
            atomic_json(
                out / "status.json",
                dict(
                    phase="packing",
                    written=position + 1,
                    total=len(rows),
                    updated=time.time(),
                ),
            )
    low.flush()
    high.flush()
    np.save(out / "boxes.npy", np.concatenate(boxes))
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    files = {
        name: dict(
            path=name + ".npy",
            sha256=sha256(out / (name + ".npy")),
            shape=list(np.load(out / (name + ".npy"), mmap_mode="r").shape),
        )
        for name in ("lr", "hr", "boxes", "offsets")
    }
    subset = dict(
        schema="isharescreen_text_packed_v1",
        complete=True,
        count=len(rows),
        files=files,
        rows=rows,
        group_aliases={},
        source_pairs_sha256=manifest["source_pairs_sha256"],
        views_sha256=manifest["views_sha256"],
        quality_audit_sha256=manifest["quality_audit_sha256"],
        training_views=1037,
        validation_views=36,
        source_groups=len({x["source_group"] for x in rows}),
        parent_manifest_sha256=sha256(manifest_path),
        selected_indices_sha256=hashlib.sha256(
            np.asarray(selected, dtype=np.int64).tobytes()
        ).hexdigest(),
        scope="Screening-only PDF train subset; same independent validation as full mixed pack. Training count and domain both differ from mixed pack, so this is a data-combination screen, not a controlled single-factor effect estimate.",
        runner_sha256=sha256(Path(__file__)),
        finished=time.time(),
    )
    atomic_json(out / "manifest.json", subset)
    atomic_json(
        out / "COMPLETED.json",
        dict(
            manifest_sha256=sha256(out / "manifest.json"),
            count=len(rows),
            finished=time.time(),
        ),
    )
    atomic_json(
        out / "status.json",
        dict(
            phase="completed", written=len(rows), total=len(rows), updated=time.time()
        ),
    )
    print("PDF SCREEN PACK COMPLETE", len(rows), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
