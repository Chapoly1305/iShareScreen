"""Index visible numeric-percent word geometry in the frozen training pack."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import time

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def numeric_percent_boxes(words: list[dict]) -> list[list[int]]:
    result = []
    for i, word in enumerate(words):
        value = str(word.get("text", ""))
        x, y, width, height = word["xywh"]
        if re.search(r"\d+(?:\.\d+)?%", value):
            left = x
        elif "%" in value and i > 0:
            previous = words[i - 1]
            previous_text = str(previous.get("text", ""))
            px, py, pw, ph = previous["xywh"]
            if (
                not re.search(r"\d$", previous_text)
                or abs(py - y) > 2
                or x - (px + pw) > 9
            ):
                continue
            left = px
        else:
            continue
        a = max(0, math.floor(left) - 6)
        b = max(0, math.floor(y) - 4)
        c = min(1920, math.ceil(x + width) + 6)
        d = min(1080, math.ceil(y + height) + 4)
        if c > a and d > b:
            result.append([a, b, c, d])
    return result


def run(packed: Path, output: Path) -> None:
    packed = packed.resolve()
    data = json.loads(packed.read_text())
    if data["schema"] != "isharescreen_text_packed_v1" or not data["complete"]:
        raise ValueError("Expected complete pinned packed data")
    rows, counts, groups = [], Counter(), {}
    for index, row in enumerate(data["rows"]):
        label_path = Path(row["labels"])
        if sha256(label_path) != row["labels_sha256"]:
            raise ValueError(f"Label identity differs for {row['id']}")
        labels = json.loads(label_path.read_text())
        boxes = numeric_percent_boxes(labels["words"])
        if boxes:
            rows.append(
                dict(
                    index=index,
                    id=row["id"],
                    split=row["split"],
                    kind=row["kind"],
                    source_group=row["source_group"],
                    labels_sha256=row["labels_sha256"],
                    boxes=boxes,
                )
            )
            counts[(row["split"], row["kind"])] += 1
            groups.setdefault((row["split"], row["kind"]), set()).add(
                row["source_group"]
            )
    report = dict(
        schema="isharescreen_percent_word_index_v1",
        packed_sha256=sha256(packed),
        runner_sha256=sha256(Path(__file__)),
        rows=rows,
        counts={f"{a}/{b}": n for (a, b), n in counts.items()},
        source_groups={f"{a}/{b}": len(v) for (a, b), v in groups.items()},
        train_symbol_word_boxes=sum(
            len(x["boxes"]) for x in rows if x["split"] == "train"
        ),
        selection="Visible source-derived word geometry containing numeric percent; adjacent number and percent tokens combined when separate. No OCR labels.",
        finished=time.time(),
    )
    atomic_json(output, report)
    print(
        "PERCENT INDEX", report["counts"], report["train_symbol_word_boxes"], flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.packed, args.output)
