"""Compare the two EDSR validation columns with pinned, paired v1 rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from benchmarks.super_resolution.low_bandwidth_e2e.analyze_validation import (
    load_rows,
    paired_interval,
    summarize,
)
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def source_equal(rows: dict, baseline: dict, keys: list) -> float:
    groups = sorted({baseline[k]["source_group"] for k in keys})
    values = []
    for group in groups:
        selected = [rows[k] for k in keys if baseline[k]["source_group"] == group]
        values.append(
            sum(x["character_edits"] for x in selected)
            / sum(x["characters"] for x in selected)
        )
    return float(np.mean(values))


def strata_summary(rows: dict, baseline: dict, keys: list, is_baseline: bool) -> dict:
    summary = summarize([rows[k] for k in keys], is_baseline)
    summary["equal_source_cer"] = source_equal(rows, baseline, keys)
    summary["empty_ocr_rows"] = sum(not rows[k]["prediction"].strip() for k in keys)
    return summary


def run(root: Path, output: Path) -> None:
    followup = root / "edsr-followup-v1"
    plan_path = followup / "PLAN.json"
    plan = json.loads(plan_path.read_text())
    protocol = root / "EVAL_PROTOCOL.json"
    if (
        plan["original_protocol_sha256"] != sha256(protocol)
        or plan["split"] != "validation"
    ):
        raise ValueError("EDSR follow-up does not match frozen validation protocol")
    paths = {
        "baseline": root / "validation-baseline",
        "quicksrnet_official": root / "ocr-results/official-validation",
        "quicksrnet_new_seed_682": root / "ocr-results/formal-seed-682-validation",
        "edsr_official": followup / "ocr/official-validation",
        "edsr_historical_1500ep": followup / "ocr/finetuned-validation",
    }
    rows = {
        name: load_rows(root, folder, name == "baseline")
        for name, folder in paths.items()
    }
    base = rows["baseline"]
    if any(set(data) != set(base) for data in rows.values()):
        raise ValueError("Method rows are not paired to the frozen baseline")
    report = {
        "schema": "isharescreen_edsr_followup_validation_analysis_v1",
        "scope": "Validation-only follow-up after v1 test; no independent claim or deployment selection.",
        "plan_sha256": sha256(plan_path),
        "protocol_sha256": sha256(protocol),
        "ocr_rows_sha256": {
            name: sha256(path / "ocr-rows.json") for name, path in paths.items()
        },
        "methods": {},
    }
    for name, data in rows.items():
        is_base = name == "baseline"
        keys = list(base)
        overall = summarize([data[k] for k in keys], is_base)
        overall["equal_source_six_cell_cer"] = source_equal(data, base, keys)
        if not is_base:
            overall["paired_vs_baseline"] = paired_interval(keys, base, data)
            if name.startswith("edsr"):
                overall["paired_vs_quicksrnet_official"] = paired_interval(
                    keys, rows["quicksrnet_official"], data
                )
        cells = []
        for height in (540, 720, 1080):
            for bps in (600000, 1000000):
                selected = [k for k in keys if k[1] == height and k[2] == bps]
                item = dict(
                    base_height=height,
                    requested_bps=bps,
                    **summarize([data[k] for k in selected], is_base),
                )
                item["equal_source_cer"] = source_equal(data, base, selected)
                if not is_base:
                    item["paired_vs_baseline"] = paired_interval(selected, base, data)
                item["by_source_kind"] = {
                    kind: summarize(
                        [data[k] for k in selected if data[k]["source_kind"] == kind],
                        is_base,
                    )
                    for kind in ("paper", "code", "web")
                    if any(data[k]["source_kind"] == kind for k in selected)
                }
                cells.append(item)
        report["methods"][name] = dict(overall=overall, cells=cells)
    report["content_strata"] = {}
    for kind in ("code", "paper", "web"):
        selected = [k for k in base if base[k]["source_kind"] == kind]
        summary = {
            "scope": "Prespecified source_kind stratum; validation only.",
            "methods": {
                name: strata_summary(data, base, selected, name == "baseline")
                for name, data in rows.items()
            },
            "historical_edsr_paired_vs_baseline": paired_interval(
                selected, base, rows["edsr_historical_1500ep"]
            ),
            "historical_edsr_paired_vs_quicksrnet_official": paired_interval(
                selected, rows["quicksrnet_official"], rows["edsr_historical_1500ep"]
            ),
            "cells": [],
        }
        for height in (540, 720, 1080):
            for bps in (600000, 1000000):
                cell_keys = [k for k in selected if k[1] == height and k[2] == bps]
                summary["cells"].append(
                    dict(
                        base_height=height,
                        requested_bps=bps,
                        methods={
                            name: strata_summary(
                                data, base, cell_keys, name == "baseline"
                            )
                            for name, data in rows.items()
                        },
                    )
                )
        report["content_strata"][kind] = summary
    code_without_first = [
        k
        for k in base
        if base[k]["source_kind"] == "code" and k[0] != "code-validation-00"
    ]
    report["sensitivity"] = {
        "scope": "Post hoc sensitivity only: omit code-validation-00, which has frequent empty OCR; not a new primary metric or model selection rule.",
        "code_without_validation_00": {
            "rows": len(code_without_first),
            "quicksrnet_source_cer": source_equal(
                rows["quicksrnet_official"], base, code_without_first
            ),
            "historical_edsr_source_cer": source_equal(
                rows["edsr_historical_1500ep"], base, code_without_first
            ),
            "paired_interval": paired_interval(
                code_without_first,
                rows["quicksrnet_official"],
                rows["edsr_historical_1500ep"],
            ),
        },
    }
    report["finished"] = time.time()
    atomic_json(output, report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
