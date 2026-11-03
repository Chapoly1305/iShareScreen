"""Summarize paired, source-level validation OCR without touching test cases."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path):
    return json.loads(path.read_text())


def load_rows(root: Path, folder: Path, baseline: bool) -> dict:
    receipt = read(folder / "COMPLETED.json")
    rows_path = folder / "ocr-rows.json"
    if sha256(rows_path) != receipt["rows_sha256"]:
        raise ValueError(f"OCR receipt differs: {folder}")
    if receipt.get("rows", 432) != 432:
        raise ValueError("Validation result lacks full 36x6x2 cells")
    result = {}
    for row in read(rows_path):
        height = row["base"][1] if baseline else row["base_height"]
        key = (row["id"], height, row["requested_bps"], row["frame_index"])
        if key in result:
            raise ValueError(f"Duplicate OCR row: {key}")
        result[key] = row
    if len(result) != 432:
        raise ValueError(f"Incomplete validation OCR: {folder}")
    return result


def summarize(rows: list[dict], baseline: bool) -> dict:
    chars = sum(row["characters"] for row in rows)
    words = sum(row["words"] for row in rows)
    return dict(
        rows=len(rows),
        source_groups=len({row["source_group"] for row in rows}),
        pooled_cer=sum(row["character_edits"] for row in rows) / chars,
        pooled_wer=sum(row["word_edits"] for row in rows) / words,
        exact_line_rate=sum(row["exact_line"] for row in rows) / len(rows),
        digit_edits=sum(row["digit_edits"] for row in rows),
        punctuation_edits=sum(row["punctuation_edits"] for row in rows),
        measured_video_bps_mean=float(
            np.mean([row["measured_video_bps"] for row in rows])
        ),
        encoder_eligible=sum(row["encoder_eligible"] for row in rows),
        mean_psnr=float(
            np.mean(
                [row["quality"]["psnr"] if baseline else row["psnr"] for row in rows]
            )
        ),
        mean_ssim=float(
            np.mean(
                [row["quality"]["ssim"] if baseline else row["ssim"] for row in rows]
            )
        ),
        median_forward_ms=None
        if baseline
        else float(np.median([row["forward_ms"] for row in rows])),
        median_full_processing_ms=None
        if baseline
        else float(np.median([row["full_processing_ms"] for row in rows])),
    )


def paired_interval(keys: list, base: dict, method: dict, seed: int = 682) -> dict:
    groups = sorted({base[key]["source_group"] for key in keys})
    bgroup, mgroup = defaultdict(list), defaultdict(list)
    for key in keys:
        bgroup[base[key]["source_group"]].append(base[key])
        mgroup[method[key]["source_group"]].append(method[key])
    source_delta = np.asarray(
        [
            sum(x["character_edits"] for x in mgroup[group])
            / sum(x["characters"] for x in mgroup[group])
            - sum(x["character_edits"] for x in bgroup[group])
            / sum(x["characters"] for x in bgroup[group])
            for group in groups
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(groups), size=(2000, len(groups)))
    samples = source_delta[indices].mean(axis=1)
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return dict(
        source_groups=len(groups),
        equal_source_cer_delta=float(source_delta.mean()),
        bootstrap_95_delta=[float(lower), float(upper)],
        probability_better=float(np.mean(samples < 0)),
        source_delta_min=float(source_delta.min()),
        source_delta_max=float(source_delta.max()),
        bootstrap_replicates=2000,
        seed=seed,
    )


def run(root: Path, output: Path) -> None:
    protocol = read(root / "EVAL_PROTOCOL.json")
    if not (root / "validation-baseline/COMPLETED.json").exists():
        raise ValueError("Unenhanced validation baseline is incomplete")
    baseline = load_rows(root, root / "validation-baseline", True)
    completed = sorted((root / "ocr-results").glob("*/COMPLETED.json"))
    methods = {p.parent.name: load_rows(root, p.parent, False) for p in completed}
    for name, rows in methods.items():
        if rows.keys() != baseline.keys():
            raise ValueError(f"Paired case/frame matrix differs: {name}")
    cases = {c["id"]: c for c in protocol["cases"] if c["split"] == "validation"}
    if len(cases) != 36 or set(k[0] for k in baseline) != cases.keys():
        raise ValueError("Frozen validation cases differ")
    language_path = root / "LANGUAGE_ANNOTATIONS.json"
    language_audit = read(language_path)
    if language_audit["eval_protocol_sha256"] != sha256(root / "EVAL_PROTOCOL.json"):
        raise ValueError("Language annotations refer to another frozen protocol")
    languages = {
        row["id"]: row["language"]
        for row in language_audit["annotations"]
        if row["split"] == "validation"
    }
    if set(languages) != cases.keys():
        raise ValueError("Language strata do not cover all validation cases")
    report = dict(
        schema="isharescreen_low_bandwidth_validation_analysis_v1",
        protocol_sha256=sha256(root / "EVAL_PROTOCOL.json"),
        baseline_rows_sha256=sha256(root / "validation-baseline/ocr-rows.json"),
        language_annotations_sha256=sha256(language_path),
        model_rows_sha256={
            name: sha256(root / "ocr-results" / name / "ocr-rows.json")
            for name in methods
        },
        scope="Validation only, source-group paired bootstrap; independent test remains unopened.",
        methods={},
    )
    for name, rows in [("baseline", baseline), *methods.items()]:
        is_baseline = name == "baseline"
        cells = []
        for height in (540, 720, 1080):
            for bps in (600000, 1000000):
                keys = [key for key in baseline if key[1] == height and key[2] == bps]
                values = [rows[key] for key in keys]
                item = dict(
                    base_height=height,
                    requested_bps=bps,
                    **summarize(values, is_baseline),
                )
                if not is_baseline:
                    item["paired_vs_baseline"] = paired_interval(keys, baseline, rows)
                item["by_source_kind"] = {
                    kind: summarize(
                        [row for row in values if row["source_kind"] == kind],
                        is_baseline,
                    )
                    for kind in sorted({row["source_kind"] for row in values})
                }
                item["by_frame"] = {
                    str(frame): summarize(
                        [row for row in values if row["frame_index"] == frame],
                        is_baseline,
                    )
                    for frame in (60, 90)
                }
                item["by_language"] = {
                    language: summarize(
                        [row for row in values if languages[row["id"]] == language],
                        is_baseline,
                    )
                    for language in sorted({languages[row["id"]] for row in values})
                }
                item["by_content_category"] = {
                    category: summarize(
                        [row for row in values if row["content_category"] == category],
                        is_baseline,
                    )
                    for category in sorted({row["content_category"] for row in values})
                }

                def font_bin(row):
                    return (
                        "small_lt18"
                        if row["line_box_height"] < 18
                        else (
                            "medium_18_31"
                            if row["line_box_height"] < 32
                            else "large_ge32"
                        )
                    )

                item["by_line_box_height"] = {
                    font: summarize(
                        [row for row in values if font_bin(row) == font], is_baseline
                    )
                    for font in sorted({font_bin(row) for row in values})
                }
                cells.append(item)
        all_keys = list(baseline)
        overall = summarize([rows[key] for key in all_keys], is_baseline)
        source_means = []
        for group in sorted({x["source_group"] for x in baseline.values()}):
            cell_cers = []
            for height in (540, 720, 1080):
                for bps in (600000, 1000000):
                    subset = [
                        rows[k]
                        for k in all_keys
                        if k[1] == height
                        and k[2] == bps
                        and baseline[k]["source_group"] == group
                    ]
                    cell_cers.append(
                        sum(x["character_edits"] for x in subset)
                        / sum(x["characters"] for x in subset)
                    )
            source_means.append(float(np.mean(cell_cers)))
        overall["equal_source_six_cell_cer"] = float(np.mean(source_means))
        if not is_baseline:
            overall["paired_vs_baseline"] = paired_interval(all_keys, baseline, rows)
            worst = sorted(
                all_keys,
                key=lambda key: rows[key]["cer"] - baseline[key]["cer"],
                reverse=True,
            )[:20]
            overall["worst_paired_cases"] = [
                dict(
                    id=k[0],
                    base_height=k[1],
                    requested_bps=k[2],
                    frame=k[3],
                    source_group=baseline[k]["source_group"],
                    baseline_cer=baseline[k]["cer"],
                    method_cer=rows[k]["cer"],
                    baseline_prediction=baseline[k]["prediction"],
                    method_prediction=rows[k]["prediction"],
                    ground_truth=rows[k]["ground_truth"],
                    crop=rows[k]["native_crop"],
                )
                for k in worst
            ]
        report["methods"][name] = dict(overall=overall, cells=cells)
    report["finished"] = time.time()
    atomic_json(output, report)
    print("VALIDATION ANALYSIS", len(methods), "model methods", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
