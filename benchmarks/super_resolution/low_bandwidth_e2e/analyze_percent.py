"""Analyze exact numeric-percent tokens on paired frozen holdout rows."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import time
import unicodedata

import numpy as np

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


TOKEN = re.compile(r"\d+(?:\.\d+)?\s*%")


def tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value)
    return [re.sub(r"\s+", "", found) for found in TOKEN.findall(normalized)]


def key(row: dict) -> tuple:
    return row["id"], row["base_height"], row["requested_bps"], row["frame_index"]


def load(folder: Path, expected: int) -> dict:
    receipt = json.loads((folder / "COMPLETED.json").read_text())
    path = folder / "ocr-rows.json"
    if sha256(path) != receipt["rows_sha256"]:
        raise ValueError(f"OCR receipt differs: {folder}")
    rows = json.loads(path.read_text())
    result = {key(row): row for row in rows}
    if len(rows) != expected or len(result) != expected:
        raise ValueError(f"Incomplete or duplicate rows: {folder}")
    return result


def token_exact(row: dict) -> bool:
    truth = tokens(row["ground_truth"])
    if not truth:
        raise ValueError("Expected a source-known numeric-percent token")
    return truth == tokens(row["prediction"])


def summarize(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_group"]].append(row)
    return dict(
        rows=len(rows),
        source_groups=len(groups),
        token_exact_rate=float(np.mean([token_exact(row) for row in rows])),
        source_equal_token_exact_rate=float(
            np.mean(
                [
                    np.mean([token_exact(row) for row in group])
                    for group in groups.values()
                ]
            )
        ),
        percent_present_rate=float(np.mean(["%" in row["prediction"] for row in rows])),
        pooled_cer=sum(row["character_edits"] for row in rows)
        / sum(row["characters"] for row in rows),
        source_equal_cer=float(
            np.mean(
                [
                    sum(row["character_edits"] for row in group)
                    / sum(row["characters"] for row in group)
                    for group in groups.values()
                ]
            )
        ),
        digit_edits=sum(row["digit_edits"] for row in rows),
        punctuation_edits=sum(row["punctuation_edits"] for row in rows),
        exact_line_rate=float(np.mean([row["exact_line"] for row in rows])),
    )


def paired(keys: list[tuple], base: dict, method: dict) -> dict:
    groups = sorted({base[k]["source_group"] for k in keys})
    token_delta, cer_delta = [], []
    for group in groups:
        selected = [k for k in keys if base[k]["source_group"] == group]
        token_delta.append(
            float(
                np.mean(
                    [
                        int(token_exact(method[k])) - int(token_exact(base[k]))
                        for k in selected
                    ]
                )
            )
        )
        cer_delta.append(
            sum(method[k]["character_edits"] for k in selected)
            / sum(method[k]["characters"] for k in selected)
            - sum(base[k]["character_edits"] for k in selected)
            / sum(base[k]["characters"] for k in selected)
        )
    rng = np.random.default_rng(682)
    draws = rng.integers(0, len(groups), size=(2000, len(groups)))
    token_boot = np.asarray(token_delta)[draws].mean(axis=1)
    cer_boot = np.asarray(cer_delta)[draws].mean(axis=1)
    return dict(
        source_groups=len(groups),
        token_exact_delta=float(np.mean(token_delta)),
        token_bootstrap_95=[float(x) for x in np.quantile(token_boot, [0.025, 0.975])],
        cer_delta=float(np.mean(cer_delta)),
        cer_bootstrap_95=[float(x) for x in np.quantile(cer_boot, [0.025, 0.975])],
        source_token_deltas=token_delta,
        source_cer_deltas=cer_delta,
        bootstrap_replicates=2000,
        seed=682,
    )


def run(root: Path, split: str, method_args: list[str], output: Path) -> None:
    root = root.resolve()
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = json.loads(protocol_path.read_text())
    cases = [case for case in protocol["cases"] if case["split"] == split]
    expected = len(cases) * 12
    methods = {"baseline": load(root / ("baseline-" + split), expected)}
    paths = {"baseline": root / ("baseline-" + split) / "ocr-rows.json"}
    for spec in method_args:
        name, folder_text = spec.split("=", 1)
        folder = Path(folder_text).resolve()
        if name in methods:
            raise ValueError(f"Duplicate method: {name}")
        methods[name] = load(folder, expected)
        paths[name] = folder / "ocr-rows.json"
    baseline = methods["baseline"]
    if any(set(rows) != set(baseline) for rows in methods.values()):
        raise ValueError("Unpaired method rows")
    clean_path = root / "clean-reference/ocr-rows.json"
    clean_receipt = json.loads((root / "clean-reference/COMPLETED.json").read_text())
    if sha256(clean_path) != clean_receipt["rows_sha256"]:
        raise ValueError("Clean reference changed")
    clean = [row for row in json.loads(clean_path.read_text()) if row["split"] == split]
    if len(clean) != len(cases):
        raise ValueError("Clean reference lacks frozen cases")
    report = dict(
        schema="isharescreen_percent_holdout_analysis_v1",
        split=split,
        protocol_sha256=sha256(protocol_path),
        input_sha256={name: sha256(path) for name, path in paths.items()},
        clean_ocr_sha256=sha256(clean_path),
        clean_reference=dict(
            views=len(clean),
            numeric_percent_token_exact=sum(token_exact(row) for row in clean),
            percent_present=sum("%" in row["prediction"] for row in clean),
            pooled_cer=sum(row["character_edits"] for row in clean)
            / sum(row["characters"] for row in clean),
        ),
        methods={},
        cells=[],
        rates=[],
        limits="Synthetic code-like views and fixed English Tesseract; no live or human proof.",
    )
    keys = list(baseline)
    for name, rows in methods.items():
        report["methods"][name] = summarize([rows[k] for k in keys])
        if name != "baseline":
            report["methods"][name]["paired_vs_baseline"] = paired(keys, baseline, rows)
    for height in (540, 720, 1080):
        for bps in (600000, 1000000):
            selected = [k for k in keys if k[1] == height and k[2] == bps]
            report["cells"].append(
                dict(
                    base_height=height,
                    requested_bps=bps,
                    methods={
                        name: summarize([rows[k] for k in selected])
                        for name, rows in methods.items()
                    },
                )
            )
    for bps in (600000, 1000000):
        selected = [k for k in keys if k[2] == bps]
        report["rates"].append(
            dict(
                requested_bps=bps,
                methods={
                    name: summarize([rows[k] for k in selected])
                    for name, rows in methods.items()
                },
                paired={
                    name: paired(selected, baseline, rows)
                    for name, rows in methods.items()
                    if name != "baseline"
                },
            )
        )
    report["finished"] = time.time()
    atomic_json(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument(
        "--method", action="append", default=[], help="NAME=OCR_DIRECTORY"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.split, args.method, args.output)
