"""Freeze all evaluation identities and decisions before new candidate training."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import time

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def run(root: Path, old_root: Path, tessdata: Path) -> None:
    target = root / "EVAL_PROTOCOL.json"
    if target.exists():
        raise FileExistsError("Evaluation protocol is already frozen")
    pdf_path = root / "independent-pdf/independent-pdf-protocol.json"
    html_path = root / "independent-html-protocol.json"
    pdf_audit_path = root / "independent-pdf/overlap-audit.json"
    html_audit_path = root / "html-overlap-audit.json"
    pdf, html = load(pdf_path), load(html_path)
    for path in (pdf_audit_path, html_audit_path):
        audit = load(path)
        if not audit["passed"]:
            raise ValueError(f"Near-duplicate audit did not pass: {path}")
    cases = []
    for kind, source in [("pdf", pdf), ("html", html)]:
        for row in source["cases"]:
            cases.append(
                dict(
                    id=row["id"],
                    source_kind=row["kind"],
                    truth_origin=kind,
                    split=row["split"],
                    source_group=row["source_group"],
                    hr=str(
                        (
                            root
                            / ("independent-pdf" if kind == "pdf" else "")
                            / row["hr"]
                        )
                        .resolve()
                        .relative_to(root)
                    ),
                    hr_sha256=row["hr_sha256"],
                    labels=str(
                        (
                            root
                            / ("independent-pdf" if kind == "pdf" else "")
                            / row["labels"]
                        )
                        .resolve()
                        .relative_to(root)
                    ),
                    labels_sha256=row["labels_sha256"],
                    line=row["line"],
                    content_category=row.get("doc_category", row["kind"]),
                    source_original=row.get(
                        "original_filename", row.get("source_file")
                    ),
                    font_size=row.get("font_size"),
                    rendering_width=row.get("rendering_width"),
                )
            )
    if len(cases) != len({row["id"] for row in cases}):
        raise ValueError("Duplicate evaluation case ID")
    by_split = {
        name: sorted((c["id"] for c in cases if c["split"] == name))
        for name in ("validation", "test")
    }
    if not by_split["validation"] or not by_split["test"]:
        raise ValueError("Missing evaluation split")
    val_groups = {c["source_group"] for c in cases if c["split"] == "validation"}
    test_groups = {c["source_group"] for c in cases if c["split"] == "test"}
    if val_groups & test_groups:
        raise ValueError("Source group crosses independent validation/test")
    protocol = dict(
        schema="isharescreen_low_bandwidth_eval_protocol_v1",
        created=time.time(),
        runner_sha256=sha256(Path(__file__)),
        pdf_protocol_sha256=sha256(pdf_path),
        html_protocol_sha256=sha256(html_path),
        pdf_overlap_audit_sha256=sha256(pdf_audit_path),
        html_overlap_audit_sha256=sha256(html_audit_path),
        old_views_sha256=sha256(old_root / "dataset-views-v3.json"),
        cases=cases,
        counts={
            f"{split}/{kind}": n
            for (split, kind), n in Counter(
                (c["split"], c["source_kind"]) for c in cases
            ).items()
        },
        fixed_display_ids={
            name: ids[: min(8, len(ids))] for name, ids in by_split.items()
        },
        main_grid=dict(
            base_geometry=[[960, 540], [1280, 720], [1920, 1080]],
            requested_video_bps=[600000, 1000000],
            display_geometry=[1920, 1080],
            x2_native_geometry=[[1920, 1080], [2560, 1440], [3840, 2160]],
            output_resampler="OpenCV Lanczos4; baseline uses bicubic upsample or native 1080p direct",
            models=[
                "baseline",
                "official_quicksrnet_x2",
                "existing_best_scratch_x2",
                "existing_official_finetune_x2",
                "new_data_candidate_x2",
            ],
        ),
        codec=dict(
            fps=30,
            pixel_format="yuv420p",
            color="full-range BT.709",
            encoder="libx265 fast/zerolatency",
            gop_frames=60,
            bframes=0,
            lookahead=0,
            episode_seconds=4,
            sampled_frame_indices=[60, 90],
            static_layout="Identical FHD source pixels and font/layout for every base/rate/model.",
            eligible_video_rate="Retained frames after first second: average video packets <=105% of requested rate and zero VBV underflows.",
            network_boundary="Packet video rate excludes RTP/IP/control/audio; no constrained-link claim without real capture.",
        ),
        ocr=dict(
            engine=subprocess.check_output(
                ["tesseract", "--version"], stderr=subprocess.STDOUT, text=True
            ),
            traineddata_sha256=sha256(tessdata / "eng.traineddata"),
            language="eng",
            oem=1,
            psm=7,
            dpi=300,
            crop="Source-derived FHD line XYWH, Lanczos x3, 12px white border; fixed for every method.",
            scoring="Case-sensitive Unicode NFKC; CER removes whitespace only; WER tokenizes whitespace-collapsed source/prediction; exact line ignores whitespace only. Digits and punctuation keep exact characters.",
            no_gt_from_ocr=True,
        ),
        metrics=[
            "pooled_CER",
            "WER",
            "exact_line_rate",
            "digit_edits",
            "punctuation_edits",
            "equal_source_CER",
            "PSNR",
            "SSIM",
        ],
        uncertainty="Paired bootstrap over original source groups, 2000 replicates, seed 682; report 95% percentile intervals and source counts.",
        strata=[
            "requested_rate",
            "measured_rate",
            "base_resolution",
            "source_kind",
            "content_category",
            "line_box_height",
            "language",
            "sampled_frame_type",
        ],
        checkpoint_selection="Minimum fixed validation text-weighted pixel/edge loss within frozen update budget; never select by test.",
        configuration_selection="On validation only, minimize equal-source CER across six primary cells; for per-rate recommendations minimize CER, then digit+punctuation edits, then complete-path runtime. Final test run once after selection.",
        reliable_gain="One-time paired test source-bootstrap 95% interval for CER improvement excludes zero at both main rates, with no increased pooled digit+punctuation edits. Otherwise keep the best validated fallback.",
        no_post_test_changes="Do not change case list, labels, metrics, encoder rules, model selection or display cases after inspecting test predictions. Record invalid/failed cells explicitly.",
        limits="Synthetic HTML reader and generated code complement original PDF pages; no natural website, Chinese OCR, Mac playback or real network proof.",
    )
    atomic_json(target, protocol)
    (root / "freeze-metrics-runner.py").write_bytes(Path(__file__).read_bytes())
    print("FROZEN EVALUATION", len(cases), protocol["counts"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--tessdata", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.old_root.resolve(), args.tessdata.resolve())


if __name__ == "__main__":
    main()
