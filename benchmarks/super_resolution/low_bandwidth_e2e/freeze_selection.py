"""Freeze validation choices and checkpoint identities before opening test OCR."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmarks.super_resolution.low_bandwidth_e2e.analyze_validation import load_rows
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def source_equal_cell(rows: dict, height: int, bps: int) -> float:
    groups = defaultdict(list)
    for key, row in rows.items():
        if key[1] == height and key[2] == bps:
            groups[row["source_group"]].append(row)
    return sum(
        sum(row["character_edits"] for row in values)
        / sum(row["characters"] for row in values)
        for values in groups.values()
    ) / len(groups)


def main() -> None:
    root = Path(
        "benchmarks/super_resolution/artifacts/text-low-bandwidth-e2e-v1"
    ).resolve()
    target = root / "FINAL_SELECTION.json"
    if target.exists():
        raise FileExistsError("Final validation selection already frozen")
    analysis_path = root / "validation-analysis.json"
    analysis = read(analysis_path)
    expected = [
        "formal-seed-682-validation",
        "formal-seed-683-validation",
        "formal-seed-684-validation",
    ]
    if not all(name in analysis["methods"] for name in expected):
        raise ValueError("All three formal OCR validations must finish first")
    if analysis["protocol_sha256"] != sha256(root / "EVAL_PROTOCOL.json"):
        raise ValueError("Validation analysis changed its source protocol")
    best_formal = min(
        expected,
        key=lambda n: (
            analysis["methods"][n]["overall"]["equal_source_six_cell_cer"],
            analysis["methods"][n]["overall"]["digit_edits"]
            + analysis["methods"][n]["overall"]["punctuation_edits"],
        ),
    )
    scratch = [
        name
        for name in analysis["methods"]
        if name.startswith("scratch-") or name == "existing-scratch-control-validation"
    ]
    best_scratch = min(
        scratch,
        key=lambda n: (
            analysis["methods"][n]["overall"]["equal_source_six_cell_cer"],
            analysis["methods"][n]["overall"]["digit_edits"]
            + analysis["methods"][n]["overall"]["punctuation_edits"],
        ),
    )
    aliases = {
        "official-validation": "official_quicksrnet_x2",
        "existing-official-finetune-validation": "existing_official_finetune_x2",
        "existing-scratch-control-validation": "existing_scratch_control_100ep",
    }
    for flavor in ("l1", "text-edge"):
        for seed in (682, 683, 684):
            aliases[f"scratch-{flavor}-{seed}-validation"] = (
                f"existing_scratch_{flavor}-seed-{seed}_x2"
            )
    for seed in (682, 683, 684):
        aliases[f"formal-seed-{seed}-validation"] = (
            f"new_formal_official_mixed_seed_{seed}_x2"
        )
    source_names = [
        "official-validation",
        "existing-official-finetune-validation",
        best_scratch,
        *expected,
    ]
    roster = [aliases[name] for name in source_names]
    if len(roster) != len(set(roster)):
        raise ValueError("Duplicate model in test roster")
    lab_weights = read(root / "staging/lab-validation-full/weights-formal.json")[
        "models"
    ]
    rtx_weights = read(root / "staging/rtx-validation-full/weights-formal.json")[
        "models"
    ]
    checkpoint_sha = {}
    for name in roster:
        bound = lab_weights.get(name) or rtx_weights.get(name)
        if bound is None:
            raise ValueError(f"Selected model missing from both workers: {name}")
        checkpoint_sha[name] = bound["sha256"]
    baseline = load_rows(root, root / "validation-baseline", True)
    candidate_rows = {"baseline": baseline}
    for source_name in source_names:
        candidate_rows[aliases[source_name]] = load_rows(
            root, root / "ocr-results" / source_name, False
        )
    cell_scores = {}
    rate_winners = {}
    for bps in (600000, 1000000):
        scores = []
        for height in (540, 720, 1080):
            for name, rows in candidate_rows.items():
                score = source_equal_cell(rows, height, bps)
                scores.append(
                    dict(
                        method=name,
                        base_height=height,
                        requested_bps=bps,
                        equal_source_cer=score,
                    )
                )
        cell_scores[str(bps)] = scores
        rate_winners[str(bps)] = min(scores, key=lambda x: x["equal_source_cer"])
    folders = {
        name: (
            "formal-" + str(seed) + "-test"
            if name.startswith("new_formal_official_mixed_seed_")
            else name.replace("_x2", "") + "-test"
        )
        for name in roster
        for seed in (
            [name.split("_seed_")[1].split("_")[0]]
            if name.startswith("new_formal_official_mixed_seed_")
            else [None]
        )
    }
    selection = dict(
        schema="isharescreen_low_bandwidth_final_selection_v1",
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        eval_protocol_sha256=sha256(root / "EVAL_PROTOCOL.json"),
        decision_rules_sha256=sha256(root / "DECISION_RULES.json"),
        analysis_sha256=sha256(analysis_path),
        validation_ocr_sha256={
            name: sha256(root / "ocr-results" / name / "ocr-rows.json")
            for name in source_names
        },
        best_historical_scratch=aliases[best_scratch],
        best_new_seed=aliases[best_formal],
        test_methods=roster,
        test_checkpoint_sha256=checkpoint_sha,
        test_result_folders=folders,
        validation_cell_scores=cell_scores,
        validation_rate_winners=rate_winners,
        unenhanced_baseline="test-baseline",
        scope="Frozen before reading any independent-test OCR outcomes. Test roster includes all three formal seeds; no post-test selection.",
    )
    atomic_json(target, selection)
    print("SELECTION FROZEN", best_scratch, best_formal, rate_winners)


if __name__ == "__main__":
    main()
