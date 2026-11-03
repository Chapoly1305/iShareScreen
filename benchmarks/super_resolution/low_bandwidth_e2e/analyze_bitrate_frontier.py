"""Frozen paired OCR analysis and validation-only minimum-rate selection."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def key(row: dict) -> tuple:
    return row["id"], row["base_height"], row["requested_bps"], row["frame_index"]


def verified_rows(folder: Path, expected: int) -> dict:
    receipt = read(folder / "COMPLETED.json")
    path = folder / "ocr-rows.json"
    if sha256(path) != receipt["rows_sha256"]:
        raise ValueError(f"OCR rows changed: {folder}")
    rows = read(path)
    keyed = {key(row): row for row in rows}
    if len(rows) != expected or len(keyed) != expected:
        raise ValueError(f"Incomplete or duplicate OCR rows: {folder}")
    return keyed


def summary(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("Empty stratum")
    groups = defaultdict(list)
    for row in rows:
        groups[row["source_group"]].append(row)

    def ratio(values: list[dict], edits: str, total: str) -> float:
        return sum(v[edits] for v in values) / max(1, sum(v[total] for v in values))

    return dict(
        rows=len(rows),
        source_groups=len(groups),
        source_equal_cer=float(
            np.mean(
                [ratio(g, "character_edits", "characters") for g in groups.values()]
            )
        ),
        pooled_cer=ratio(rows, "character_edits", "characters"),
        pooled_wer=ratio(rows, "word_edits", "words"),
        exact_line_rate=float(np.mean([r["exact_line"] for r in rows])),
        digit_edits=sum(r["digit_edits"] for r in rows),
        punctuation_edits=sum(r["punctuation_edits"] for r in rows),
        digits=sum(r["digits"] for r in rows),
        punctuation=sum(r["punctuation"] for r in rows),
        mean_measured_video_bps=float(np.mean([r["measured_video_bps"] for r in rows])),
        eligible=sum(r["encoder_eligible"] for r in rows),
        mean_psnr=(
            float(np.mean([r["psnr"] for r in rows]))
            if all(r["psnr"] is not None for r in rows)
            else None
        ),
        mean_ssim=(
            float(np.mean([r["ssim"] for r in rows]))
            if all(r["ssim"] is not None for r in rows)
            else None
        ),
    )


def paired(keys: list[tuple], baseline: dict, enhanced: dict) -> dict:
    group_deltas = []
    for source_group in sorted({baseline[k]["source_group"] for k in keys}):
        selected = [k for k in keys if baseline[k]["source_group"] == source_group]
        a = summary([baseline[k] for k in selected])["pooled_cer"]
        b = summary([enhanced[k] for k in selected])["pooled_cer"]
        group_deltas.append(b - a)
    rng = np.random.default_rng(682)
    draw = rng.integers(0, len(group_deltas), size=(2000, len(group_deltas)))
    means = np.asarray(group_deltas)[draw].mean(axis=1)
    return dict(
        source_groups=len(group_deltas),
        cer_delta=float(np.mean(group_deltas)),
        bootstrap_95=[float(x) for x in np.quantile(means, [0.025, 0.975])],
        source_deltas=group_deltas,
        replicates=2000,
        seed=682,
    )


def run(root: Path, split: str) -> None:
    root = root.resolve()
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = read(protocol_path)
    cases = [c for c in protocol["cases"] if c["split"] == split]
    expected = (
        len(cases)
        * len(protocol["main_grid"]["base_geometry"])
        * len(protocol["main_grid"]["requested_video_bps"])
        * 2
    )
    base = verified_rows(root / ("baseline-" + split), expected)
    enhanced = verified_rows(root / "ocr-results" / ("official-" + split), expected)
    if set(base) != set(enhanced):
        raise ValueError("Unpaired evaluation rows")
    quality_folder = root / "quality-results" / ("direct-" + split)
    quality_receipt = read(quality_folder / "COMPLETED.json")
    quality_path = quality_folder / "results.json"
    if sha256(quality_path) != quality_receipt["results_sha256"]:
        raise ValueError("Direct display quality receipt differs")
    quality_result = read(quality_path)
    if quality_result["binding"]["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Direct quality used another protocol")
    quality = {
        (r["case"], r["base_height"], r["requested_bps"], r["frame_index"]): r
        for r in quality_result["rows"]
    }
    if set(quality) != set(base):
        raise ValueError("Direct quality lacks paired rows")
    for key_, row in base.items():
        row["psnr"] = quality[key_]["psnr"]
        row["ssim"] = quality[key_]["ssim"]
    clean_path = root / "clean-reference/ocr-rows.json"
    clean_receipt = read(root / "clean-reference/COMPLETED.json")
    if sha256(clean_path) != clean_receipt["rows_sha256"]:
        raise ValueError("Clean OCR receipt changed")
    clean = [r for r in read(clean_path) if r["split"] == split]
    if len(clean) != len(cases):
        raise ValueError("Missing clean source rows")
    cells = []
    for kind in ("code", "paper"):
        for height in (720, 1080):
            for bps in protocol["main_grid"]["requested_video_bps"]:
                keys = [
                    k
                    for k in base
                    if base[k]["source_kind"] == kind and k[1] == height and k[2] == bps
                ]
                cells.append(
                    dict(
                        source_kind=kind,
                        base_height=height,
                        requested_bps=bps,
                        direct=summary([base[k] for k in keys]),
                        quick=summary([enhanced[k] for k in keys]),
                        quick_vs_direct=paired(keys, base, enhanced),
                    )
                )
    output = root / (split.upper() + "_ANALYSIS.json")
    report = dict(
        schema="isharescreen_bitrate_frontier_analysis_v1",
        split=split,
        protocol_sha256=sha256(protocol_path),
        input_sha256={
            "baseline": sha256(root / ("baseline-" + split) / "ocr-rows.json"),
            "quick": sha256(
                root / "ocr-results" / ("official-" + split) / "ocr-rows.json"
            ),
            "direct_quality": sha256(quality_path),
        },
        clean={
            kind: summary(
                [
                    dict(r, measured_video_bps=0, encoder_eligible=True, psnr=0, ssim=0)
                    for r in clean
                    if next(c for c in cases if c["id"] == r["id"])["source_kind"]
                    == kind
                ]
            )
            for kind in ("code", "paper")
        },
        cells=cells,
        finished=time.time(),
    )
    atomic_json(output, report)
    if split == "validation":
        selected = None
        gates = []
        for bps in protocol["main_grid"]["requested_video_bps"]:
            code = next(
                c
                for c in cells
                if c["source_kind"] == "code"
                and c["base_height"] == 720
                and c["requested_bps"] == bps
            )
            paper = next(
                c
                for c in cells
                if c["source_kind"] == "paper"
                and c["base_height"] == 720
                and c["requested_bps"] == bps
            )
            paper_1080 = next(
                c
                for c in cells
                if c["source_kind"] == "paper"
                and c["base_height"] == 1080
                and c["requested_bps"] == bps
            )
            checks = dict(
                code_cer=code["quick"]["source_equal_cer"] <= 0.10,
                paired_gain=code["quick_vs_direct"]["bootstrap_95"][1] < 0,
                symbols=(
                    code["quick"]["digit_edits"] + code["quick"]["punctuation_edits"]
                    <= code["direct"]["digit_edits"]
                    + code["direct"]["punctuation_edits"]
                ),
                paper_guardrail=paper["quick"]["source_equal_cer"]
                <= paper_1080["direct"]["source_equal_cer"] + 0.02,
                encoder=code["quick"]["eligible"] == code["quick"]["rows"]
                and paper["quick"]["eligible"] == paper["quick"]["rows"],
            )
            passed = all(checks.values())
            gates.append(dict(requested_bps=bps, checks=checks, passed=passed))
            if passed and selected is None:
                selected = bps
        atomic_json(
            root / "SELECTION.json",
            dict(
                schema="isharescreen_bitrate_frontier_selection_v1",
                eval_protocol_sha256=sha256(protocol_path),
                validation_analysis_sha256=sha256(output),
                selected_720p_quick_bps=selected,
                gates=gates,
                test_methods=["official_quicksrnet_x2"],
                frozen=time.time(),
            ),
        )
    print(
        "FRONTIER ANALYSIS",
        split,
        "selected" if split == "validation" else "tested",
        selected if split == "validation" else len(cells),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    args = p.parse_args()
    run(args.root, args.split)
