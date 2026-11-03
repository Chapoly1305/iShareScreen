"""Verify the frozen percent experiment and produce its H100 audit receipt."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

import torch


FILES = (
    "PLAN.json",
    "PLAN-v2.json",
    "METRIC_PROTOCOL.json",
    "PERCENT_INDEX.json",
    "WEIGHTS-FORMAL.json",
    "WEIGHTS-OFFICIAL-QUICK.json",
    "VALIDATION-SCREEN.json",
    "VALIDATION-FORMAL.json",
    "VALIDATION-FULL.json",
    "SUPPLEMENTARY-VALIDATION.json",
    "SELECTION.json",
    "REPORT.md",
    "RESOURCE_LEDGER.json",
    "gallery/percent-validation-examples.png",
    "gallery/GALLERY.json",
    "holdout/EVAL_PROTOCOL.json",
    "holdout/OVERLAP_AUDIT.json",
    "holdout/TRAIN-OVERLAP-AUDIT.json",
    "holdout/evaluation-inputs/INPUTS-COMPLETED.json",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def run(root: Path) -> None:
    root = root.resolve()
    holdout = root / "holdout"
    for name in FILES:
        if not (root / name).is_file():
            raise FileNotFoundError(name)
    metric = read(root / "METRIC_PROTOCOL.json")
    runner = Path(__file__).with_name("analyze_percent.py")
    assert metric["analysis_runner_sha256"] == sha(runner)
    selected = read(root / "SELECTION.json")
    assert selected["test_methods"] == [] and not selected["test_opened"]
    assert selected["eval_protocol_sha256"] == sha(holdout / "EVAL_PROTOCOL.json")
    assert selected["validation_analysis_sha256"] == sha(
        root / "VALIDATION-FORMAL.json"
    )
    assert selected["formal_plan_sha256"] == sha(root / "PLAN-v2.json")
    assert not any("test" in p.name for p in (holdout / "gpu-results").iterdir())
    assert not any("test" in p.name for p in (holdout / "ocr").iterdir())

    training = {}
    for kind in ("generic", "symbol"):
        path = root / f"remote-results/lab-formal-{kind}-800"
        receipt = read(path / "COMPLETED.json")
        assert receipt["steps"] == 800
        assert sha(path / "last.pt") == receipt["last_sha256"]
        assert sha(path / "best.pt") == receipt["best_sha256"]
        checkpoint = torch.load(path / "last.pt", map_location="cpu", weights_only=True)
        assert checkpoint["step"] == 800
        assert checkpoint["protocol_binding"] == receipt["protocol_binding"]
        assert {"optimizer", "scheduler", "numpy_rng", "torch_rng", "cuda_rng"} <= set(
            checkpoint
        )
        training[kind] = dict(
            last_sha256=receipt["last_sha256"],
            protocol_binding=receipt["protocol_binding"],
            optimizer_scheduler_rng_present=True,
            receipt_sha256=sha(path / "COMPLETED.json"),
        )

    ocr_paths = {
        "baseline": holdout / "baseline-validation",
        "official_quick": holdout / "ocr/official-quick-validation",
        "historical_edsr": holdout / "ocr/old-validation",
        "generic_800": holdout / "ocr/formal-generic-validation",
        "symbol_800": holdout / "ocr/formal-symbol-validation",
    }
    evaluation = {}
    for name, path in ocr_paths.items():
        receipt = read(path / "COMPLETED.json")
        assert receipt["rows"] == 144
        assert sha(path / "ocr-rows.json") == receipt["rows_sha256"]
        assert len(read(path / "ocr-rows.json")) == 144
        evaluation[name] = dict(
            rows=144,
            ocr_sha256=receipt["rows_sha256"],
            ocr_receipt_sha256=sha(path / "COMPLETED.json"),
        )
    gpu_paths = {
        "official_quick": "official-quick-validation",
        "historical_edsr": "old-validation",
        "generic_800": "formal-generic-validation",
        "symbol_800": "formal-symbol-validation",
    }
    for name, folder in gpu_paths.items():
        path = holdout / "gpu-results" / folder
        receipt = read(path / "COMPLETED.json")
        assert (
            receipt["rows"] == 144
            and sha(path / "results.json") == receipt["results_sha256"]
        )
        rows = read(path / "results.json")["rows"]
        assert len(rows) == 144
        for row in rows:
            assert sha(path / row["crop"]) == row["crop_sha256"]
            if row["display"]:
                assert sha(path / row["display"]["path"]) == row["display"]["sha256"]
        evaluation[name]["gpu_result_sha256"] = receipt["results_sha256"]
    ledger = read(root / "RESOURCE_LEDGER.json")
    assert ledger["combined_gpu_hours"] < ledger["first_round_limit_gpu_hours"]
    output = dict(
        schema="isharescreen_percent_final_audit_v1",
        audited_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        host="gpuserverzeng H100",
        files_sha256={name: sha(root / name) for name in FILES},
        audit_runner_sha256=sha(Path(__file__)),
        metric_runner_sha256=metric["analysis_runner_sha256"],
        formal_training=training,
        paired_validation=evaluation,
        validation_cells=144,
        validation_source_views=12,
        test_model_predictions_opened=False,
        clean_reference_test_ocr_was_premeasured=True,
        test_gate_status="rejected_by_frozen_validation_criteria",
        combined_first_round_gpu_hours=ledger["combined_gpu_hours"],
        first_round_cap_gpu_hours=ledger["first_round_limit_gpu_hours"],
        failed_jobs_preserved=["lab-official-quick-validation-failed-launch"],
        limits=[
            "four validation source-template families",
            "synthetic code-like percent content",
            "clean 1080p OCR fails 4/12 validation views",
            "no candidate independent test",
            "no live network/Mac client or full-video temporal validation",
        ],
    )
    (root / "FINAL-AUDIT.json").write_text(json.dumps(output, indent=2) + "\n")
    print("AUDIT PASSED", ledger["combined_gpu_hours"], "GPU hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    run(args.root)
