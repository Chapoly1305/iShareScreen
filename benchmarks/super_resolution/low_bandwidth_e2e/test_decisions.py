"""Apply frozen validation choices and predeclared evidence rules to test rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from benchmarks.super_resolution.low_bandwidth_e2e.analyze_test import load_rows
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def subset(rows: dict, *, bps: int, height: int | None = None) -> dict:
    return {
        key: value
        for key, value in rows.items()
        if key[2] == bps and (height is None or key[1] == height)
    }


def equal_source_cer(rows: dict) -> float:
    groups = sorted({value["source_group"] for value in rows.values()})
    return sum(
        sum(
            value["character_edits"]
            for value in rows.values()
            if value["source_group"] == group
        )
        / sum(
            value["characters"]
            for value in rows.values()
            if value["source_group"] == group
        )
        for group in groups
    ) / len(groups)


def paired(base: dict, other: dict) -> dict:
    if base.keys() != other.keys():
        # Cross-bitrate settings differ in their rate key; pair by case, height
        # within a selected comparison, and sampled frame instead.
        aligned_base = {(k[0], k[1], k[3]): v for k, v in base.items()}
        aligned_other = {(k[0], k[1], k[3]): v for k, v in other.items()}
        if aligned_base.keys() != aligned_other.keys():
            raise ValueError("Paired independent test cases differ")
    else:
        aligned_base, aligned_other = base, other
    groups = sorted({v["source_group"] for v in aligned_base.values()})
    diffs = []
    for group in groups:
        left = [v for v in aligned_base.values() if v["source_group"] == group]
        right = [v for v in aligned_other.values() if v["source_group"] == group]
        diffs.append(
            sum(v["character_edits"] for v in right)
            / sum(v["characters"] for v in right)
            - sum(v["character_edits"] for v in left)
            / sum(v["characters"] for v in left)
        )
    diffs = np.asarray(diffs)
    rng = np.random.default_rng(682)
    replicates = diffs[rng.integers(0, len(diffs), size=(2000, len(diffs)))].mean(
        axis=1
    )
    return dict(
        source_groups=len(groups),
        equal_source_cer_delta=float(diffs.mean()),
        bootstrap_95_delta=[float(x) for x in np.quantile(replicates, [0.025, 0.975])],
        pooled_digit_punct_edits_base=sum(
            v["digit_edits"] + v["punctuation_edits"] for v in aligned_base.values()
        ),
        pooled_digit_punct_edits_other=sum(
            v["digit_edits"] + v["punctuation_edits"] for v in aligned_other.values()
        ),
        resamples=2000,
        seed=682,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("benchmarks/super_resolution/artifacts/text-low-bandwidth-e2e-v1"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    selection = read(root / "FINAL_SELECTION.json")
    rules = read(root / "DECISION_RULES.json")
    if selection["decision_rules_sha256"] != sha256(root / "DECISION_RULES.json"):
        raise ValueError("Decision rules changed after selection")
    analysis_path = root / "test-analysis.json"
    analysis = read(analysis_path)
    if analysis["protocol_sha256"] != selection["eval_protocol_sha256"]:
        raise ValueError("Test analysis uses another protocol")
    folders = selection["test_result_folders"]
    if set(analysis["methods"]) != {"baseline", *folders.values()}:
        raise ValueError("Test analysis does not contain exactly the frozen roster")
    rows = {"baseline": load_rows(root, root / "test-baseline", True)}
    rows.update(
        {
            method: load_rows(root, root / "test-ocr-results" / folder, False)
            for method, folder in folders.items()
        }
    )
    nonnew = [
        "baseline",
        "official_quicksrnet_x2",
        "existing_official_finetune_x2",
        selection["best_historical_scratch"],
    ]
    chosen = selection["best_new_seed"]
    gain_by_rate = {}
    for bps in (600000, 1000000):
        score_rows = selection["validation_cell_scores"][str(bps)]
        nonnew_scores = {
            method: sum(
                item["equal_source_cer"]
                for item in score_rows
                if item["method"] == method
            )
            / 3
            for method in nonnew
        }
        comparator = min(nonnew_scores, key=nonnew_scores.get)
        comparison = paired(
            subset(rows[comparator], bps=bps), subset(rows[chosen], bps=bps)
        )
        comparison["validation_chosen_comparator"] = comparator
        comparison["candidate"] = chosen
        comparison["reliable_new_gain"] = (
            comparison["bootstrap_95_delta"][1] < 0
            and comparison["pooled_digit_punct_edits_other"]
            <= comparison["pooled_digit_punct_edits_base"]
        )
        gain_by_rate[str(bps)] = comparison
    reliable_both_rates = all(v["reliable_new_gain"] for v in gain_by_rate.values())
    low = selection["validation_rate_winners"]["600000"]
    high = selection["validation_rate_winners"]["1000000"]
    left = subset(rows[high["method"]], bps=1000000, height=high["base_height"])
    right = subset(rows[low["method"]], bps=600000, height=low["base_height"])
    if low["base_height"] != high["base_height"]:
        # Permit different base geometries by aligning only original source and frame.
        left = {(k[0], 0, k[2], k[3]): v for k, v in left.items()}
        right = {(k[0], 0, k[2], k[3]): v for k, v in right.items()}
    bandwidth = paired(left, right)
    bandwidth["validation_selected_600kbps"] = low
    bandwidth["validation_selected_1mbps"] = high
    bandwidth["equivalent_within_margin"] = (
        bandwidth["bootstrap_95_delta"][1] <= 0.02
        and bandwidth["pooled_digit_punct_edits_other"]
        <= bandwidth["pooled_digit_punct_edits_base"]
    )
    all_test_cells = {}
    for bps in (600000, 1000000):
        all_test_cells[str(bps)] = sorted(
            [
                dict(
                    method=method,
                    base_height=height,
                    requested_bps=bps,
                    equal_source_cer=equal_source_cer(
                        subset(method_rows, bps=bps, height=height)
                    ),
                )
                for method, method_rows in rows.items()
                for height in (540, 720, 1080)
            ],
            key=lambda item: item["equal_source_cer"],
        )
    result = dict(
        schema="isharescreen_low_bandwidth_test_decisions_v1",
        selection_sha256=sha256(root / "FINAL_SELECTION.json"),
        rules_sha256=sha256(root / "DECISION_RULES.json"),
        analysis_sha256=sha256(analysis_path),
        new_candidate_gain_by_rate=gain_by_rate,
        reliable_new_gain_both_rates=reliable_both_rates,
        validation_selected_lower_bandwidth_comparison=bandwidth,
        descriptive_test_cell_rankings=all_test_cells,
        ranking_policy="Descriptive full-test rankings only; validation-selected configurations and model roster remain frozen.",
        eligible_cells_rule=rules["primary_test_cells"],
        finished=time.time(),
    )
    atomic_json(
        args.output.resolve() if args.output else root / "test-decisions.json", result
    )
    print("TEST DECISIONS", reliable_both_rates, bandwidth["equivalent_within_margin"])


if __name__ == "__main__":
    main()
