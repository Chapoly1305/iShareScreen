"""Score returned EDSR frames with the frozen Linux OCR and create pixel gallery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.base_text_quality import ocr
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def run(root: Path) -> None:
    plan = read(root / "PLAN.json")
    remote = read(root / "results.json")
    done = read(root / "COMPLETED.json")
    if (
        sha256(root / "results.json") != done["results_sha256"]
        or remote["plan_sha256"] != sha256(root / "PLAN.json")
        or len(remote["rows"]) != 6
    ):
        raise ValueError("Remote completion invalid")
    prior = root.parent / "historical-video-frontier-v1"
    if sha256(prior / "manifest.json") != plan["historical_manifest_sha256"]:
        raise ValueError("Previous video manifest changed")
    truth_path = Path(
        "benchmarks/super_resolution/artifacts/base-sweep-v1/text-ground-truth.json"
    )
    if sha256(truth_path) != plan["truth_sha256"]:
        raise ValueError("Ground truth changed")
    truth = read(truth_path)
    scores = []
    for row in remote["rows"]:
        scene = next(s for s in plan["scenes"] if s["id"] == row["scene"])
        if (
            row["input_sha256"] != scene["input_sha256"]
            or row["checkpoint_sha256"]
            != read(root / "weights.json")["models"][row["method"]]["sha256"]
        ):
            raise ValueError("Output identity differs")
        image = root / row["output"]
        if sha256(image) != row["output_sha256"]:
            raise ValueError("Returned pixels changed")
        if Image.open(image).size != (1920, 1080):
            raise ValueError("Wrong display geometry")
        for region in truth["regions"]:
            if region["scene"] != row["scene"]:
                continue
            crop = (
                root
                / "ocr-crops"
                / row["method"]
                / row["scene"]
                / f"{region['id']}.png"
            )
            crop.parent.mkdir(parents=True, exist_ok=True)
            reference = (
                Path("benchmarks/super_resolution/artifacts/base-sweep-v1")
                / row["scene"]
                / "reference-2s.png"
            )
            task = dict(
                region,
                config="720p-1250k-420-30",
                method=row["method"],
                image=str(image),
                reference=str(reference),
                crop=str(crop),
                root=str(root),
            )
            scored = ocr(task)
            scored["source_fold"] = scene["source_fold"]
            scored["fold0_training_overlap"] = scene["fold0_training_overlap"]
            scores.append(scored)
    if len(scores) != 12:
        raise ValueError("Incomplete OCR rows")
    atomic_json(
        root / "OCR_RESULTS.json",
        {"plan_sha256": sha256(root / "PLAN.json"), "rows": scores},
    )
    grouped = []
    for method in plan["methods"]:
        for scene in plan["scenes"]:
            rows = [
                x for x in scores if x["method"] == method and x["scene"] == scene["id"]
            ]
            grouped.append(
                {
                    "method": method,
                    "scene": scene["id"],
                    "source_fold": scene["source_fold"],
                    "fold0_training_overlap": scene["fold0_training_overlap"],
                    "cer": sum(x["edit_distance"] for x in rows)
                    / sum(x["characters"] for x in rows),
                    "exact_blocks": sum(x["exact_nonspace"] for x in rows),
                    "digit_errors": sum(x["digit_errors"] for x in rows),
                    "punctuation_errors": sum(x["punctuation_errors"] for x in rows),
                }
            )
    atomic_json(
        root / "SUMMARY.json",
        {
            "plan_sha256": sha256(root / "PLAN.json"),
            "ocr_results_sha256": sha256(root / "OCR_RESULTS.json"),
            "per_scene": grouped,
        },
    )
    gallery = root / "gallery"
    gallery.mkdir(exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    prior_scores = read(prior / "results.json")["text_rows"]
    shown = []
    for scene_id, region_id, zoom in (
        ("light-code", "syntax", 3),
        ("terminal-light", "process-errors", 2),
        ("web-document", "paragraph", 2),
    ):
        scene = next(s for s in plan["scenes"] if s["id"] == scene_id)
        region = next(
            r
            for r in truth["regions"]
            if r["scene"] == scene_id and r["id"] == region_id
        )
        x, y, w, h = region["xywh"]
        paths = [
            (
                "Source",
                Path("benchmarks/super_resolution/artifacts/base-sweep-v1")
                / scene_id
                / "reference-2s.png",
                None,
            ),
            (
                "720p direct",
                prior
                / "display-frames"
                / scene_id
                / "720p-1250k-420-30"
                / "720_direct.png",
                "720_direct",
            ),
            (
                "Official QuickSRNet",
                prior
                / "display-frames"
                / scene_id
                / "720p-1250k-420-30"
                / "720_quicksrnet.png",
                "720_quicksrnet",
            ),
        ]
        paths += [
            (
                "Official EDSR",
                root / "outputs" / "official_edsr_x2" / f"{scene_id}.png",
                "official_edsr_x2",
            ),
            (
                "EDSR 1500 ep, fold 0",
                root / "outputs" / "old_finetuned_edsr_x2" / f"{scene_id}.png",
                "old_finetuned_edsr_x2",
            ),
        ]
        slot = h * zoom + 58
        canvas = Image.new("RGB", (w * zoom + 28, slot * len(paths) + 20), "#f1f5f9")
        draw = ImageDraw.Draw(canvas)
        for i, (label, path, method) in enumerate(paths):
            if method in plan["methods"]:
                match = next(
                    r
                    for r in scores
                    if r["scene"] == scene_id
                    and r["region"] == region_id
                    and r["method"] == method
                )
            elif method:
                match = next(
                    r
                    for r in prior_scores
                    if r["scene"] == scene_id
                    and r["region"] == region_id
                    and r["method"] == method
                    and r["config"] == "720p-1250k-420-30"
                )
            else:
                match = None
            label += (
                "" if match is None else f"  ·  block CER {match['cer'] * 100:.1f}%"
            )
            top = 10 + i * slot
            draw.text((14, top), label, font=font, fill="#17212b")
            crop = Image.open(path).convert("RGB").crop((x, y, x + w, y + h))
            canvas.paste(
                crop.resize((w * zoom, h * zoom), Image.Resampling.NEAREST),
                (14, top + 40),
            )
        target = gallery / f"{scene_id}-1250k-edsr1500.png"
        canvas.save(target)
        shown.append(
            {
                "scene": scene_id,
                "region": region_id,
                "source_fold": scene["source_fold"],
                "fold0_training_overlap": scene["fold0_training_overlap"],
                "image": str(target.relative_to(root)),
                "sha256": sha256(target),
            }
        )
    atomic_json(
        gallery / "GALLERY.json",
        {
            "scope": "Illustrative raw-pixel crops at 2s, enlarged nearest-neighbor; not metrics input",
            "items": shown,
        },
    )
    lines = [
        "# EDSR 第 1500 轮同帧补测",
        "",
        "目标 1.25 Mbps；相同 720p HEVC 输入，x2 到 1440p 后统一 area 缩到 1080p。",
        "同一 Linux Tesseract 后端和六块已核实英文文字真值；只评估第 2 秒。",
        "",
        "| 场景 | 原始来源折 | 是否进入 fold 0 训练 | 官方 EDSR CER | 1500 轮 EDSR CER |",
        "|---|---:|---|---:|---:|",
    ]
    for scene in plan["scenes"]:
        vals = {g["method"]: g["cer"] for g in grouped if g["scene"] == scene["id"]}
        lines.append(
            f"| {scene['id']} | {scene['source_fold']} | {'是' if scene['fold0_training_overlap'] else '否'} "
            f"| {vals['official_edsr_x2']:.3f} | {vals['old_finetuned_edsr_x2']:.3f} |"
        )
    lines += [
        "",
        "代码和终端录屏在 fold 0 的训练来源中；不能把它们的文字分数当作独立泛化证明。",
        "网页录屏是该权重按折未训练过的来源，但这批历史材料也已反复查看，仍不是新独立测试。",
        "EDSR 1500 轮权重来自旧录屏配对训练，并未用本轮新下载的大量文字材料训练。",
        "1500 轮 fold 0 为 current 权重；不是所有五折的挑优组合。",
        "",
        "同帧裁剪见 `gallery/`；逐块 OCR 预测在 `OCR_RESULTS.json`。GPU 结果、权重与输入哈希见 `results.json` 和 `PLAN.json`。",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        root / "FINAL_AUDIT.json",
        {
            "plan_sha256": sha256(root / "PLAN.json"),
            "remote_results_sha256": sha256(root / "results.json"),
            "ocr_results_sha256": sha256(root / "OCR_RESULTS.json"),
            "summary_sha256": sha256(root / "SUMMARY.json"),
            "report_sha256": sha256(root / "REPORT.md"),
            "gallery_sha256": sha256(gallery / "GALLERY.json"),
            "runner_sha256": sha256(Path(__file__)),
            "remote_runner_sha256": sha256(
                Path(
                    "benchmarks/super_resolution/low_bandwidth_e2e/infer_historical_edsr_gpu.py"
                )
            ),
            "remote_gpu_seconds": remote["ended_unix"] - remote["started_unix"],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root.resolve())
