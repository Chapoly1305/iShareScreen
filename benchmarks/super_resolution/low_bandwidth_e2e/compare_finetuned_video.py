"""Descriptive, source-bound comparison of QuickSRNet weights on old videos."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.base_text_quality import ocr
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.candidates import TorchUpscaler
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def run(root: Path) -> None:
    plan_path = root / "PLAN.json"
    plan = read(plan_path)
    previous = Path(plan["historical_root"])
    if sha256(previous / "manifest.json") != plan["manifest_sha256"]:
        raise ValueError("Video manifest changed")
    if sha256(previous / "results.json") != plan["previous_results_sha256"]:
        raise ValueError("Previous paired scores changed")
    if sha256(Path(plan["truth_path"])) != plan["truth_sha256"]:
        raise ValueError("Text truth changed")
    manifest = read(previous / "manifest.json")
    truth = read(Path(plan["truth_path"]))
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    rows = []
    identities = {}
    for key, spec in plan["models"].items():
        checkpoint = Path(spec["checkpoint"])
        receipt = Path(spec["receipt"])
        if (
            sha256(checkpoint) != spec["sha256"]
            or sha256(receipt) != spec["receipt_sha256"]
        ):
            raise ValueError(f"Checkpoint or training receipt changed: {key}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        done = read(receipt)
        if (
            payload["protocol_binding"] != done["protocol_binding"]
            or done["best_sha256"] != spec["sha256"]
            or payload["epoch"] != done["best_epoch"]
        ):
            raise ValueError(f"Training/checkpoint mismatch: {key}")
        adapter = TorchUpscaler("quicksrnet", device="cpu", precision="fp32")
        if adapter.provenance["checkpoint_sha256"] != plan["official_sha256"]:
            raise ValueError("Official initializer changed")
        torch.nn.Module.load_state_dict(
            adapter.model, payload["state_dict"], strict=True
        )
        adapter.model.eval()
        identities[key] = {
            "checkpoint_sha256": spec["sha256"],
            "epoch": payload["epoch"],
            "protocol_binding": payload["protocol_binding"],
        }
        for cell in manifest["rows"]:
            if cell["height"] != 720 or cell["target_kbps"] not in plan["target_kbps"]:
                continue
            if sha256(Path(cell["video"])) != cell["video_sha256"]:
                raise ValueError("Video changed")
            input_path = Path(cell["frames"]["2"])
            if sha256(input_path) != cell["frame_sha256"]["2"]:
                raise ValueError("Decoded input changed")
            rgb = np.asarray(Image.open(input_path).convert("RGB"))
            enhanced, _ = adapter(rgb)
            if enhanced.shape != (1440, 2560, 3):
                raise ValueError("Wrong x2 model geometry")
            display = cv2.resize(enhanced, (1920, 1080), interpolation=cv2.INTER_AREA)
            display_path = (
                root / "display" / key / cell["scene"] / f"{cell['target_kbps']}k.png"
            )
            display_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(display).save(display_path)
            for region in truth["regions"]:
                if region["scene"] != cell["scene"]:
                    continue
                crop_path = (
                    root
                    / "crops"
                    / key
                    / cell["scene"]
                    / f"{cell['target_kbps']}k-{region['id']}.png"
                )
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                task = dict(
                    region,
                    config=cell["config"],
                    method=key,
                    image=str(display_path),
                    reference=str(
                        Path(plan["reference_root"])
                        / cell["scene"]
                        / "reference-2s.png"
                    ),
                    crop=str(crop_path),
                    root=str(root),
                )
                row = ocr(task)
                row["checkpoint_sha256"] = spec["sha256"]
                row["display_sha256"] = sha256(display_path)
                rows.append(row)
            print(f"SCORED {key} {cell['scene']} {cell['target_kbps']}k", flush=True)
    expected = len(plan["models"]) * len(plan["target_kbps"]) * len(truth["regions"])
    if len(rows) != expected:
        raise ValueError(f"Incomplete OCR rows: {len(rows)} != {expected}")
    atomic_json(
        root / "results.json",
        {
            "schema": "historical_video_finetuned_comparison_v1",
            "plan_sha256": sha256(plan_path),
            "models": identities,
            "rows": rows,
            "torch": torch.__version__,
        },
    )
    previous_rows = read(previous / "results.json")["text_rows"]
    combined = rows + [
        r for r in previous_rows if r["method"] in ("720_direct", "720_quicksrnet")
    ]
    grouped = defaultdict(list)
    for row in combined:
        if (
            row["config"].startswith("720p-")
            and int(row["config"].split("-")[1][:-1]) in plan["target_kbps"]
        ):
            grouped[(row["method"], row["config"], row["scene"])].append(row)
    per_scene = []
    for (method, config, scene), values in sorted(grouped.items()):
        per_scene.append(
            {
                "method": method,
                "config": config,
                "scene": scene,
                "cer": sum(r["edit_distance"] for r in values)
                / sum(r["characters"] for r in values),
                "wer": sum(r["word_errors"] for r in values)
                / sum(r["words"] for r in values),
                "exact_blocks": sum(r["exact_nonspace"] for r in values),
                "digit_errors": sum(r["digit_errors"] for r in values),
                "punctuation_errors": sum(r["punctuation_errors"] for r in values),
            }
        )
    summary = []
    for bitrate in plan["target_kbps"]:
        config = f"720p-{bitrate}k-420-30"
        for method in ("720_direct", "720_quicksrnet", *plan["models"]):
            values = [
                r for r in per_scene if r["method"] == method and r["config"] == config
            ]
            if len(values) != 3:
                raise ValueError(f"Missing paired scene: {method} {config}")
            summary.append(
                {
                    "target_kbps": bitrate,
                    "method": method,
                    "cer": float(np.mean([r["cer"] for r in values])),
                    "wer": float(np.mean([r["wer"] for r in values])),
                    "exact_blocks": sum(r["exact_blocks"] for r in values),
                    "digit_errors": sum(r["digit_errors"] for r in values),
                    "punctuation_errors": sum(r["punctuation_errors"] for r in values),
                }
            )
    atomic_json(
        root / "SUMMARY.json",
        {
            "plan_sha256": sha256(plan_path),
            "results_sha256": sha256(root / "results.json"),
            "per_scene": per_scene,
            "summary": summary,
        },
    )
    lines = [
        "# 720p QuickSRNet 微调权重的旧视频回归",
        "",
        "同三段历史视频、同一编码帧和同一 Linux OCR。旧视频已经反复查看，不能据此选产品默认模型。",
        "两份微调权重分别来自既有文字数据训练 100 epoch，以及本轮混合新文字数据正式训练 80 epoch 的 seed 682（最佳 epoch 42）。",
        "这里只评估每段第 2 秒的六块已核实英文真值；不包含 `%`，未测整段文字准确性。",
        "",
        "| 目标码率 | 方法 | CER ↓ | WER ↓ | 正确块/6 | 数字错误 | 标点错误 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {r['target_kbps']} Kbps | {r['method']} | {r['cer']:.3f} | {r['wer']:.3f} | "
            f"{r['exact_blocks']}/6 | {r['digit_errors']} | {r['punctuation_errors']} |"
        )
    lines += [
        "",
        "码率为目标视频码率；每个模型复用同一个已校验 720p HEVC 输入，实测码率见原历史视频报告。",
        "CER 为三个场景各自汇总后等权平均；六块小样本仅作描述，不能证明普遍收益。",
        "",
        "逐场景及逐文字块结果见 `SUMMARY.json` 与 `results.json`；原始显示帧与 OCR 裁剪保存在 `display/` 和 `crops/`。",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        root / "COMPLETED.json",
        {
            "plan_sha256": sha256(plan_path),
            "result_sha256": sha256(root / "results.json"),
            "summary_sha256": sha256(root / "SUMMARY.json"),
            "report_sha256": sha256(root / "REPORT.md"),
            "runner_sha256": sha256(Path(__file__)),
            "ocr_rows": len(rows),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root.resolve())
