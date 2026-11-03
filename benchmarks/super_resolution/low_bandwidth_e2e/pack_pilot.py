"""Pack hash-bound 540p training/validation pairs for the existing QuickSRNet trainer."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def choose_rate(identifier: str) -> int:
    value = int(
        hashlib.sha256(
            ("isharescreen-validation-rate-v1:" + identifier).encode()
        ).hexdigest()[:8],
        16,
    )
    return 600000 if value % 2 else 1000000


def records(root: Path) -> tuple[list[dict], dict]:
    pairs_path = root / "pilot/hevc-pilot-v1/pairs.json"
    pair_receipt = read(root / "pilot/hevc-pilot-v1/PAIRS-COMPLETED.json")
    if sha256(pairs_path) != pair_receipt["pairs_sha256"]:
        raise ValueError("Pilot pairs receipt differs")
    pilot = read(pairs_path)
    if not pilot["complete"] or pilot["count"] != 2000:
        raise ValueError("Pilot pairs incomplete")
    training = []
    excluded = Counter()
    for pair in pilot["pairs"]:
        if pair["kind"] == "web":
            excluded["old_web_license_pending"] += 1
            continue
        if pair["kind"] == "paper" and pair.get("doc_category") == "patents":
            excluded["patent_diagram_domain_mismatch"] += 1
            continue
        row = dict(pair)
        row["split"] = "train"
        row["hr"] = str((root / "pilot" / pair["hr"]).resolve())
        row["lr"] = str((root / "pilot" / pair["lr"]).resolve())
        row["labels"] = str((root / "pilot" / pair["labels"]).resolve())
        training.append(row)
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = read(protocol_path)
    validation = []
    for case in protocol["cases"]:
        if case["split"] != "validation":
            continue
        bps = choose_rate(case["id"])
        folder = root / "evaluation-inputs" / case["id"] / f"960x540-{bps // 1000}k"
        receipt = read(folder / "COMPLETED.json")
        if receipt["binding"]["protocol_sha256"] != sha256(protocol_path):
            raise ValueError("Validation encode belongs to another protocol")
        frame = next(x for x in receipt["sampled_frames"] if x["index"] == 90)
        if sha256(folder / frame["path"]) != frame["sha256"]:
            raise ValueError("Validation decode frame differs")
        labels = read(root / case["labels"])
        validation.append(
            dict(
                id="validation-" + case["id"],
                kind=case["source_kind"],
                split="validation",
                source_group=case["source_group"],
                hr=str((root / case["hr"]).resolve()),
                hr_sha256=case["hr_sha256"],
                hr_rgb_sha256=labels["hr_rgb_sha256"],
                lr=str((folder / frame["path"]).resolve()),
                lr_sha256=frame["sha256"],
                labels=str((root / case["labels"]).resolve()),
                labels_sha256=case["labels_sha256"],
                requested_bps=bps,
                actual_episode_video_bps=receipt[
                    "measured_video_bps_after_first_second"
                ],
                validation_frame=90,
                official_independent_validation=True,
            )
        )
    rows = training + validation
    if len(training) != 1637 or len(validation) != 36:
        raise ValueError(
            f"Unexpected train/validation mixture: {len(training)}/{len(validation)}"
        )
    return rows, dict(
        excluded=dict(excluded),
        pilot_pairs_sha256=sha256(pairs_path),
        eval_protocol_sha256=sha256(protocol_path),
        independent_input_manifest_sha256=sha256(
            root / "evaluation-inputs/manifest.json"
        ),
    )


def run(root: Path) -> None:
    rows, binding = records(root)
    output = root / "packed-pilot-v1"
    if output.exists():
        raise FileExistsError(
            "Packed pilot exists; inspect partial status rather than silently restarting"
        )
    output.mkdir()
    atomic_json(output / "source-binding.json", binding)
    (output / "runner.py").write_bytes(Path(__file__).read_bytes())
    groups = {}
    pixels = set()
    for row in rows:
        group = row["source_group"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError("Source group crosses train/validation")
        groups[group] = row["split"]
        if row["hr_rgb_sha256"] in pixels and row["split"] == "train":
            raise ValueError("Duplicate HR training pixels")
        pixels.add(row["hr_rgb_sha256"])
    arrays = {
        role: np.lib.format.open_memmap(
            output / (role + ".npy"),
            mode="w+",
            dtype=np.uint8,
            shape=(len(rows), height, width, 3),
        )
        for role, height, width in [("lr", 540, 960), ("hr", 1080, 1920)]
    }
    boxes = []
    offsets = [0]
    for index, row in enumerate(rows):
        for role in ("lr", "hr"):
            path = Path(row[role])
            if sha256(path) != row[role + "_sha256"]:
                raise ValueError(f"Pair pixels changed: {path}")
            image = np.array(Image.open(path).convert("RGB"))
            if image.shape != arrays[role].shape[1:]:
                raise ValueError(f"Wrong {role} geometry: {path}")
            identity = hashlib.sha256(image.tobytes()).hexdigest()
            if role + "_rgb_sha256" in row and identity != row[role + "_rgb_sha256"]:
                raise ValueError(f"Pixel sequence differs: {path}")
            row[role + "_rgb_sha256"] = identity
            arrays[role][index] = image
        label_path = Path(row["labels"])
        if sha256(label_path) != row["labels_sha256"]:
            raise ValueError("Text geometry changed")
        labels = read(label_path)
        if labels["hr_rgb_sha256"] != row["hr_rgb_sha256"]:
            raise ValueError("Labels do not describe original HR pixels")
        values = []
        for word in labels["words"]:
            x, y, width, height = word["xywh"]
            weight = (
                3
                if any(char.isdigit() or not char.isalnum() for char in word["text"])
                else 1
            )
            values.append(
                [int(x), int(y), math.ceil(x + width), math.ceil(y + height), weight]
            )
        boxes.append(np.asarray(values, dtype=np.uint16))
        offsets.append(offsets[-1] + len(values))
        if (index + 1) % 50 == 0:
            for array in arrays.values():
                array.flush()
            atomic_json(
                output / "status.json",
                dict(
                    phase="packing",
                    written=index + 1,
                    total=len(rows),
                    updated=time.time(),
                ),
            )
            print("PACK PILOT", index + 1, "/", len(rows), flush=True)
    for array in arrays.values():
        array.flush()
    np.save(output / "boxes.npy", np.concatenate(boxes))
    np.save(output / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    identities = {
        name: dict(
            path=name + ".npy",
            sha256=sha256(output / (name + ".npy")),
            shape=list(np.load(output / (name + ".npy"), mmap_mode="r").shape),
        )
        for name in ("lr", "hr", "boxes", "offsets")
    }
    manifest = dict(
        schema="isharescreen_text_packed_v1",
        complete=True,
        count=len(rows),
        files=identities,
        rows=rows,
        group_aliases={},
        source_pairs_sha256=binding["pilot_pairs_sha256"],
        views_sha256=binding["eval_protocol_sha256"],
        quality_audit_sha256=sha256(root / "protocol-verification.json"),
        training_views=len(training := [r for r in rows if r["split"] == "train"]),
        validation_views=len(rows) - len(training),
        excluded=binding["excluded"],
        source_groups=len(groups),
        scope="1637 train views: 1037 new DocLayNet non-patent PDFs plus 400 old train code, 100 train table, 100 train terminal. 163 diagram-heavy patent and 200 old web views excluded for predeclared domain/license reasons. 36 independent validation cases; no independent test view enters training or checkpoint selection.",
        runner_sha256=sha256(Path(__file__)),
        finished=time.time(),
    )
    atomic_json(output / "manifest.json", manifest)
    atomic_json(
        output / "COMPLETED.json",
        dict(
            manifest_sha256=sha256(output / "manifest.json"),
            count=len(rows),
            finished=time.time(),
        ),
    )
    print("PACK PILOT COMPLETE", len(rows), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
