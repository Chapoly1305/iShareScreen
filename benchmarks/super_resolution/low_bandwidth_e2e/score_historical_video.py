"""Run the frozen 1/1.25/1.5 Mbps historical video regression on H100.

The old excerpts are development material. The scorer uses only the existing
manually checked 2-second text blocks, never OCR output as ground truth.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.base_text_quality import ocr
from benchmarks.super_resolution.bitrate_poc import packet_stats
from benchmarks.super_resolution.metrics import metrics
from benchmarks.super_resolution.prepare_video import probe
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.candidates import TorchUpscaler
from isharescreen.enhancement.models import sha256


BASE = Path("benchmarks/super_resolution/artifacts/base-sweep-v1")
TIMES = (1, 2, 3)


def _inputs(root: Path) -> tuple[dict, dict, dict]:
    plan_path = root / "PLAN.json"
    status = json.loads((root / "STATUS.json").read_text())
    if sha256(plan_path) != status["plan_sha256"]:
        raise ValueError("frozen plan changed")
    plan = json.loads(plan_path.read_text())
    base = json.loads((BASE / "manifest.json").read_text())
    truth = json.loads((BASE / "text-ground-truth.json").read_text())
    if sha256(BASE / "manifest.json") != plan["source_manifest_sha256"]:
        raise ValueError("historical manifest changed")
    if sha256(BASE / "text-ground-truth.json") != plan["ground_truth_sha256"]:
        raise ValueError("historical text labels changed")
    for source in plan["sources"]:
        if sha256(Path(source["reference"])) != source["sha256"]:
            raise ValueError(f"source changed: {source['id']}")
    return plan, base, truth


def _sample(path: Path, folder: Path, width: int, height: int) -> dict[str, str]:
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    count = 0
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if frame.is_corrupt or (frame.width, frame.height) != (width, height):
                raise ValueError(f"invalid decoded frame {path}:{i}")
            if abs(float(frame.pts * frame.time_base) - i / 30) > 0.002:
                raise ValueError(f"nonaligned frame PTS {path}:{i}")
            if i in (30, 60, 90):
                target = folder / f"input-{i // 30}s.png"
                Image.fromarray(frame.to_ndarray(format="rgb24")).save(target)
                paths[str(i // 30)] = str(target)
            count += 1
    if count != 120 or len(paths) != 3:
        raise ValueError(f"incomplete clip: {path}, frames={count}")
    return paths


def encode(root: Path) -> None:
    plan, base, _ = _inputs(root)
    output = root / "encodes"
    output.mkdir(exist_ok=True)
    rows = []
    historical = {(x["scene"], x["config"]): x for x in base["runs"]}
    for scene in plan["matrix"]["scenes"]:
        source = next(x for x in plan["sources"] if x["id"] == scene)
        for height in plan["matrix"]["base_heights"]:
            width = height * 16 // 9
            old = historical[(scene, f"{height}p-1000k-420-30")]
            for kbps in plan["matrix"]["target_kbps"]:
                ident = f"{height}p-{kbps}k-420-30"
                folder = output / scene / ident
                if kbps == 1000:
                    old_file = BASE / old["encoded"]
                    expected = next(
                        x
                        for x in plan["reused_1mbps"]
                        if x["scene"] == scene and x["config"] == old["config"]
                    )
                    if sha256(old_file) != expected["sha256"]:
                        raise ValueError(f"old encode changed: {old_file}")
                    frames = {
                        str(t): str(BASE / scene / old["config"] / f"input-{t}s.png")
                        for t in TIMES
                    }
                    for t, path in frames.items():
                        if sha256(Path(path)) != old["input_png_hashes"][t]:
                            raise ValueError(f"old sampled frame changed: {path}")
                    row = dict(
                        scene=scene,
                        config=ident,
                        height=height,
                        target_kbps=kbps,
                        source="reused_identity_verified",
                        video=str(old_file),
                        video_sha256=expected["sha256"],
                        frames=frames,
                        frame_sha256=old["input_png_hashes"],
                        rate=old["rate"],
                        vbv_underflows=old["vbv_underflows"],
                        budget_pass=old["budget_pass"],
                    )
                else:
                    folder.mkdir(parents=True, exist_ok=True)
                    target = folder / "encoded.mp4"
                    receipt = folder / "COMPLETED.json"
                    if receipt.exists():
                        row = json.loads(receipt.read_text())
                        if (
                            row["scene"] != scene
                            or row["config"] != ident
                            or sha256(target) != row["video_sha256"]
                        ):
                            raise ValueError(f"failed resume audit: {folder}")
                        for t, path in row["frames"].items():
                            if sha256(Path(path)) != row["frame_sha256"][t]:
                                raise ValueError(f"sample changed: {path}")
                    else:
                        if target.exists():
                            raise ValueError(
                                f"partial encode preserved for inspection: {target}"
                            )
                        command = list(old["command"])
                        command[command.index("-i") + 1] = source["reference"]
                        command[-1] = str(target)
                        for flag in ("-b:v", "-minrate", "-maxrate", "-bufsize"):
                            command[command.index(flag) + 1] = str(kbps * 1000)
                        position = command.index("-x265-params") + 1
                        command[position] = re.sub(
                            r"(?<![\w-])(bitrate|vbv-maxrate|vbv-bufsize)=\d+",
                            lambda match: match[1] + "=" + str(kbps),
                            command[position],
                        )
                        with (folder / "encode.log").open("w") as log:
                            subprocess.run(command, check=True, stdout=log, stderr=log)
                        stream = probe(target)
                        if (
                            (stream["width"], stream["height"]) != (width, height)
                            or stream["pix_fmt"] not in ("yuv420p", "yuvj420p")
                            or Fraction(stream["avg_frame_rate"]) != 30
                            or int(stream["nb_read_frames"]) != 120
                            or stream["has_b_frames"] != 0
                        ):
                            raise ValueError(f"unexpected encode format: {target}")
                        frames = _sample(target, folder, width, height)
                        rate = packet_stats(target, Fraction(30), 30)
                        underflows = (
                            (folder / "encode.log").read_text().count("VBV underflow")
                        )
                        row = dict(
                            scene=scene,
                            config=ident,
                            height=height,
                            target_kbps=kbps,
                            source="new_matching_historical_encoder",
                            video=str(target),
                            video_sha256=sha256(target),
                            frames=frames,
                            frame_sha256={
                                t: sha256(Path(path)) for t, path in frames.items()
                            },
                            command=command,
                            stream=stream,
                            rate=rate,
                            vbv_underflows=underflows,
                            budget_pass=rate["retained_video_bps"] <= kbps * 1000 * 1.05
                            and underflows == 0,
                            log_sha256=sha256(folder / "encode.log"),
                        )
                        atomic_json(receipt, row)
                rows.append(row)
                print(
                    f"ENCODED {len(rows)}/18 {scene} {ident}: "
                    f"{row['rate']['retained_video_bps'] / 1000:.1f} kbps",
                    flush=True,
                )
    manifest = dict(
        schema="historical_video_frontier_inputs_v1",
        plan_sha256=sha256(root / "PLAN.json"),
        rows=rows,
        source_manifest_sha256=plan["source_manifest_sha256"],
    )
    atomic_json(root / "manifest.json", manifest)
    atomic_json(
        root / "ENCODES-COMPLETED.json",
        dict(manifest_sha256=sha256(root / "manifest.json"), cells=len(rows)),
    )
    status = json.loads((root / "STATUS.json").read_text())
    status.update(
        phase="encodes_complete",
        complete_cells=len(rows),
        next_step="Run the frozen Linux OCR and image scorer.",
    )
    atomic_json(root / "STATUS.json", status)


def _score_cell(args: tuple[Path, dict, list[dict], str, str]) -> dict:
    root, cell, regions, plan_sha, manifest_sha = args
    import torch

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    cell_receipt = (
        root / "scored-cells" / cell["scene"] / cell["config"] / "COMPLETED.json"
    )
    if cell_receipt.exists():
        cached = json.loads(cell_receipt.read_text())
        if (
            cached["plan_sha256"] != plan_sha
            or cached["manifest_sha256"] != manifest_sha
            or cached["video_sha256"] != cell["video_sha256"]
        ):
            raise ValueError(f"scored cell identity changed: {cell_receipt}")
        return cached
    if sha256(Path(cell["video"])) != cell["video_sha256"]:
        raise ValueError(f"video changed: {cell['video']}")
    model = None
    if cell["height"] == 720:
        model = TorchUpscaler("quicksrnet", device="cpu", precision="fp32")
        if (
            model.provenance["checkpoint_sha256"]
            != "d95d70f1d2366cb9c28d99f8c7aa5bb07e1ffeaf5d7e30d9c66ab0fa28c6d0f8"
        ):
            raise ValueError("official model changed")
    quality = []
    text_rows = []
    scene = cell["scene"]
    methods = ("720_direct", "720_quicksrnet") if model else ("1080_direct",)
    for second in TIMES:
        input_path = Path(cell["frames"][str(second)])
        if sha256(input_path) != cell["frame_sha256"][str(second)]:
            raise ValueError(f"sample changed: {input_path}")
        image = np.asarray(Image.open(input_path).convert("RGB"))
        reference_path = BASE / scene / f"reference-{second}s.png"
        reference = np.asarray(Image.open(reference_path).convert("RGB"))
        if reference.shape != (1080, 1920, 3):
            raise ValueError("unexpected reference shape")
        outputs = {}
        if model:
            outputs["720_direct"] = cv2.resize(
                image, (1920, 1080), interpolation=cv2.INTER_CUBIC
            )
            enhanced, _ = model(image)
            if enhanced.shape != (1440, 2560, 3):
                raise ValueError("unexpected native x2 output")
            outputs["720_quicksrnet"] = cv2.resize(
                enhanced, (1920, 1080), interpolation=cv2.INTER_AREA
            )
        else:
            outputs["1080_direct"] = image
        for method in methods:
            output = outputs[method]
            q = metrics(reference, output)
            quality.append(
                dict(
                    scene=scene,
                    config=cell["config"],
                    second=second,
                    method=method,
                    psnr=q["psnr"],
                    ssim=q["ssim"],
                )
            )
            if second != 2:
                continue
            display = root / "display-frames" / scene / cell["config"] / f"{method}.png"
            display.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(output).save(display)
            for region in regions:
                crop = (
                    root
                    / "ocr-crops"
                    / scene
                    / cell["config"]
                    / f"{method}-{region['id']}.png"
                )
                crop.parent.mkdir(parents=True, exist_ok=True)
                task = dict(
                    region,
                    config=cell["config"],
                    method=method,
                    image=str(display),
                    reference=str(reference_path),
                    crop=str(crop),
                    root=str(root),
                )
                text_rows.append(ocr(task))
    output = dict(
        plan_sha256=plan_sha,
        manifest_sha256=manifest_sha,
        video_sha256=cell["video_sha256"],
        quality=quality,
        text_rows=text_rows,
    )
    cell_receipt.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(cell_receipt, output)
    return output


def score(root: Path) -> None:
    plan, _, truth = _inputs(root)
    manifest_path = root / "manifest.json"
    receipt = json.loads((root / "ENCODES-COMPLETED.json").read_text())
    if receipt["manifest_sha256"] != sha256(manifest_path) or receipt["cells"] != 18:
        raise ValueError("encode receipt invalid")
    manifest = json.loads(manifest_path.read_text())
    import torch

    tasks = [
        (
            root,
            cell,
            [r for r in truth["regions"] if r["scene"] == cell["scene"]],
            sha256(root / "PLAN.json"),
            sha256(manifest_path),
        )
        for cell in manifest["rows"]
    ]
    quality = []
    text_rows = []
    with ProcessPoolExecutor(max_workers=6) as executor:
        for i, scored_cell in enumerate(executor.map(_score_cell, tasks), 1):
            quality.extend(scored_cell["quality"])
            text_rows.extend(scored_cell["text_rows"])
            print(
                f"SCORED {i}/18 {manifest['rows'][i - 1]['scene']} {manifest['rows'][i - 1]['config']}",
                flush=True,
            )
    model = TorchUpscaler("quicksrnet", device="cpu", precision="fp32")
    if (
        model.provenance["checkpoint_sha256"]
        != plan["official_quicksrnet_checkpoint_sha256"]
    ):
        raise ValueError("official model changed")
    for region in truth["regions"]:
        scene = region["scene"]
        reference_path = BASE / scene / "reference-2s.png"
        crop = root / "ocr-crops" / scene / f"reference-{region['id']}.png"
        crop.parent.mkdir(parents=True, exist_ok=True)
        text_rows.append(
            ocr(
                dict(
                    region,
                    config="reference",
                    method="reference",
                    image=str(reference_path),
                    reference=str(reference_path),
                    crop=str(crop),
                    root=str(root),
                )
            )
        )
    if len(quality) != 81 or len(text_rows) != 60:
        raise ValueError(
            f"incomplete scoring quality={len(quality)} text={len(text_rows)}"
        )
    results = dict(
        schema="historical_video_frontier_results_v1",
        plan_sha256=sha256(root / "PLAN.json"),
        manifest_sha256=sha256(manifest_path),
        model=model.provenance,
        quality=quality,
        text_rows=text_rows,
        environment={"torch": torch.__version__, "opencv": cv2.__version__},
        completed_utc=time.time(),
    )
    atomic_json(root / "results.json", results)
    atomic_json(
        root / "SCORES-COMPLETED.json",
        dict(
            results_sha256=sha256(root / "results.json"),
            quality_rows=len(quality),
            text_rows=len(text_rows),
        ),
    )
    status = json.loads((root / "STATUS.json").read_text())
    status.update(
        phase="scores_complete", next_step="Aggregate and audit paired video results."
    )
    atomic_json(root / "STATUS.json", status)


def report(root: Path) -> None:
    plan, _, truth = _inputs(root)
    results_path = root / "results.json"
    scored = json.loads((root / "SCORES-COMPLETED.json").read_text())
    if scored["results_sha256"] != sha256(results_path):
        raise ValueError("score receipt invalid")
    result = json.loads(results_path.read_text())
    manifest = json.loads((root / "manifest.json").read_text())
    for cell in manifest["rows"]:
        if sha256(Path(cell["video"])) != cell["video_sha256"]:
            raise ValueError("video changed during final audit")
        for second, path in cell["frames"].items():
            if sha256(Path(path)) != cell["frame_sha256"][second]:
                raise ValueError("sampled frame changed during final audit")
        if cell["source"] == "new_matching_historical_encoder":
            receipt = json.loads(
                (
                    root / "encodes" / cell["scene"] / cell["config"] / "COMPLETED.json"
                ).read_text()
            )
            if receipt["video_sha256"] != cell["video_sha256"]:
                raise ValueError("new encode receipt changed")
        scored_cell = json.loads(
            (
                root
                / "scored-cells"
                / cell["scene"]
                / cell["config"]
                / "COMPLETED.json"
            ).read_text()
        )
        if (
            scored_cell["plan_sha256"] != sha256(root / "PLAN.json")
            or scored_cell["manifest_sha256"] != sha256(root / "manifest.json")
            or scored_cell["video_sha256"] != cell["video_sha256"]
        ):
            raise ValueError("cell score receipt changed")
    grouped = defaultdict(list)
    for row in result["text_rows"]:
        grouped[(row["config"], row["method"], row["scene"])].append(row)
    per_scene = []
    for (config, method, scene), values in grouped.items():
        per_scene.append(
            dict(
                config=config,
                method=method,
                scene=scene,
                cer=sum(v["edit_distance"] for v in values)
                / sum(v["characters"] for v in values),
                wer=sum(v["word_errors"] for v in values)
                / sum(v["words"] for v in values),
                exact_blocks=sum(v["exact_nonspace"] for v in values),
                digit_errors=sum(v["digit_errors"] for v in values),
                punctuation_errors=sum(v["punctuation_errors"] for v in values),
            )
        )
    summary = []
    for kbps in plan["matrix"]["target_kbps"]:
        for height, method in (
            (720, "720_direct"),
            (720, "720_quicksrnet"),
            (1080, "1080_direct"),
        ):
            rows = [
                r
                for r in per_scene
                if r["config"] == f"{height}p-{kbps}k-420-30" and r["method"] == method
            ]
            q = [
                r
                for r in result["quality"]
                if r["config"] == f"{height}p-{kbps}k-420-30" and r["method"] == method
            ]
            cells = [
                r
                for r in manifest["rows"]
                if r["height"] == height and r["target_kbps"] == kbps
            ]
            if len(rows) != 3 or len(q) != 9 or len(cells) != 3:
                raise ValueError("incomplete paired summary")
            summary.append(
                dict(
                    target_kbps=kbps,
                    height=height,
                    method=method,
                    cer=float(np.mean([r["cer"] for r in rows])),
                    wer=float(np.mean([r["wer"] for r in rows])),
                    exact_blocks=sum(r["exact_blocks"] for r in rows),
                    digit_errors=sum(r["digit_errors"] for r in rows),
                    punctuation_errors=sum(r["punctuation_errors"] for r in rows),
                    psnr=float(np.mean([r["psnr"] for r in q])),
                    ssim=float(np.mean([r["ssim"] for r in q])),
                    mean_actual_kbps=float(
                        np.mean([r["rate"]["retained_video_bps"] / 1000 for r in cells])
                    ),
                    max_actual_kbps=max(
                        r["rate"]["retained_video_bps"] / 1000 for r in cells
                    ),
                    budget_pass=all(r["budget_pass"] for r in cells),
                )
            )
    clean = [r for r in per_scene if r["config"] == "reference"]
    clean_cer = float(np.mean([r["cer"] for r in clean]))
    report_data = dict(
        schema="historical_video_frontier_summary_v1",
        plan_sha256=sha256(root / "PLAN.json"),
        results_sha256=sha256(results_path),
        clean_reference_cer=clean_cer,
        per_scene=per_scene,
        summary=summary,
    )
    atomic_json(root / "SUMMARY.json", report_data)
    lines = [
        "# 既有录屏的码率补充评估",
        "",
        "三段历史开发片段（代码、终端、网页），各 4 秒，30 fps，HEVC 4:2:0，固定内容和字号。",
        "1 Mbps 沿用已校验编码；1.25/1.5 Mbps 按相同编码器命令重新编码。所有 OCR 在同一 Linux Tesseract 后端重跑。",
        "文字真值仅为 2 秒处六块此前人工核对的英文区域；不是新独立测试，也不是完整视频转录。",
        "全图 PSNR/SSIM 取 1、2、3 秒同源配对帧；视频码率为首秒后视频 packet，不含实际网络协议开销。",
        "720p QuickSRNet 为官方预训练 x2，原生 1440p 输出后统一缩至 1080p；CPU 推理只用于离线质量评估，不代表实时性能。",
        "",
        f"干净参考图相同 OCR 流程 CER：{clean_cer:.3f}。",
        "",
        "| 目标 Kbps | 方法 | 实际视频 Kbps | CER ↓ | WER ↓ | 完全正确块 / 6 | 数字错误 | 标点错误 | PSNR ↑ | SSIM ↑ | 编码合格 |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['target_kbps']} | {row['method']} | {row['mean_actual_kbps']:.1f} "
            f"| {row['cer']:.3f} | {row['wer']:.3f} | {row['exact_blocks']}/6 "
            f"| {row['digit_errors']} | {row['punctuation_errors']} "
            f"| {row['psnr']:.2f} | {row['ssim']:.4f} | {'是' if row['budget_pass'] else '否'} |"
        )
    lines += [
        "",
        "## 1.25 Mbps 的内容差异",
        "",
        "| 方法 | 代码 CER | 终端 CER | 网页 CER |",
        "|---|---:|---:|---:|",
    ]
    for height, method in (
        (720, "720_direct"),
        (720, "720_quicksrnet"),
        (1080, "1080_direct"),
    ):
        vals = {
            r["scene"]: r["cer"]
            for r in per_scene
            if r["config"] == f"{height}p-1250k-420-30" and r["method"] == method
        }
        lines.append(
            f"| {method} | {vals['light-code']:.3f} | {vals['terminal-light']:.3f} | {vals['web-document']:.3f} |"
        )
    lines += [
        "",
        "差异主要集中在历史终端录屏：1080p 在该目标码率下出现严重的细字损伤。原像素裁剪显示 720p 路径更平滑，这可能是固定 OCR 在该片段上更容易识别的原因；未做独立的预滤波消融。",
        "QuickSRNet 提高了全图 PSNR/SSIM，但 1.25 Mbps 的总 CER 高于 720p 直通；它没有稳定的文字准确性优势。",
        "这六块真值不包含 `%`，不能用本次结果回答百分号问题。",
        "",
    ]
    temporal_path = root / "TEMPORAL_DIAGNOSTIC.json"
    temporal_sha = None
    if temporal_path.exists():
        temporal_receipt = json.loads((root / "TEMPORAL-COMPLETED.json").read_text())
        if temporal_receipt["result_sha256"] != sha256(temporal_path):
            raise ValueError("temporal diagnostic changed")
        temporal_sha = sha256(temporal_path)
        temporal = json.loads(temporal_path.read_text())
        lines += [
            "## 整段视频补充诊断",
            "",
            "每个直通配置的全部 120 帧与同时间源帧配对，得到全图 PSNR 和固定文字区域的相邻帧差误差。后者仅为像素诊断，不是主观闪烁或准确文字指标；本项是在冻结选择协议之外增加的描述性检查。",
            "",
            "| 目标 Kbps | 基础分辨率 | 全段 PSNR ↑ | 文字区域帧差误差 ↓ |",
            "|---:|---:|---:|---:|",
        ]
        for kbps in plan["matrix"]["target_kbps"]:
            for height in (720, 1080):
                vals = [
                    r
                    for r in temporal["rows"]
                    if r["target_kbps"] == kbps and r["height"] == height
                ]
                lines.append(
                    f"| {kbps} | {height}p | {np.mean([r['mean_psnr'] for r in vals]):.2f} "
                    f"| {np.mean([r['roi_temporal_error_mean'] for r in vals]):.2f} |"
                )
        lines += ["", "完整逐帧数据见 `TEMPORAL_DIAGNOSTIC.json`。", ""]
    gallery = []
    for scene, kbps in (
        ("light-code", 1250),
        ("terminal-light", 1000),
        ("terminal-light", 1250),
        ("terminal-light", 1500),
    ):
        video = root / "gallery" / f"{scene}-{kbps}k-comparison.mp4"
        receipt = video.with_suffix(".json")
        preview = video.with_name(video.stem.replace("comparison", "preview") + ".png")
        if video.exists():
            data = json.loads(receipt.read_text())
            if sha256(video) != data["sha256"] or data["frames"] != 120:
                raise ValueError("comparison video changed")
            gallery.append(
                dict(
                    scene=scene,
                    target_kbps=kbps,
                    video=str(video.relative_to(root)),
                    video_sha256=sha256(video),
                    preview=str(preview.relative_to(root)),
                    preview_sha256=sha256(preview),
                )
            )
    if gallery:
        lines += [
            "## 同帧视频对照",
            "",
            "查看副本按同一个源帧拼接原录屏、720p 直通、720p QuickSRNet 和 1080p 直通；H.264 查看视频绝不作为指标输入。",
            "",
            "[代码 1.25 Mbps](gallery/light-code-1250k-comparison.mp4) · [终端 1 Mbps](gallery/terminal-light-1000k-comparison.mp4) · [终端 1.25 Mbps](gallery/terminal-light-1250k-comparison.mp4) · [终端 1.5 Mbps](gallery/terminal-light-1500k-comparison.mp4) · [浏览页](gallery/index.html)",
            "",
        ]
        page = [
            '<!doctype html><html lang="zh"><meta charset="utf-8"><title>历史视频 1.25 Mbps 对照</title>',
            "<style>body{font-family:system-ui;margin:2rem auto;max-width:1000px;background:#f8fafc;color:#16202d}video,img{max-width:100%;height:auto}section{margin:2rem 0;padding:1rem;background:white;border-radius:10px}</style>",
            "<h1>历史视频 1.25 Mbps 对照</h1><p>原录屏、720p 直通、720p QuickSRNet、1080p 直通。视频是查看副本，不参与评分。</p>",
        ]
        for item in gallery:
            title = (
                "代码" if item["scene"] == "light-code" else "终端"
            ) + f" · {item['target_kbps']} Kbps"
            page.append(
                f'<section><h2>{title}</h2><video controls preload="metadata" poster="{Path(item["preview"]).name}" src="{Path(item["video"]).name}"></video><p><a href="{Path(item["preview"]).name}">查看原像素裁剪预览</a></p></section>'
            )
        page.append('<p><a href="../REPORT.md">评估结果与限制</a></p></html>')
        (root / "gallery" / "index.html").write_text("\n".join(page) + "\n")
    lines += [
        "",
        "## 解释边界",
        "",
        "这些视频、裁剪与转录在旧实验中已反复看过，仅能做回归描述；不能用其小样本差值选择新默认配置、推断总体收益或认定可用最低码率。",
        "同一录屏的三帧不是三个独立来源。整段时序文字、真人辨认、网络开销与真实弱网均未验证。新的独立静态测试在代码上的 1080p 直通结论仍有效；这里的单个旧终端优势不能覆盖它。",
        "",
        "逐源和逐块预测见 `SUMMARY.json`、`results.json`；所有视频身份、编码命令与实际 packet 码率见 `manifest.json`。",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    atomic_json(
        root / "FINAL_AUDIT.json",
        dict(
            plan_sha256=sha256(root / "PLAN.json"),
            manifest_sha256=sha256(root / "manifest.json"),
            results_sha256=sha256(results_path),
            summary_sha256=sha256(root / "SUMMARY.json"),
            report_sha256=sha256(root / "REPORT.md"),
            generator_sha256=sha256(Path(__file__)),
            code_identity_sha256=(
                sha256(root / "CODE_IDENTITY.json")
                if (root / "CODE_IDENTITY.json").exists()
                else None
            ),
            temporal_sha256=temporal_sha,
            gallery=gallery,
            complete_cells=len(manifest["rows"]),
            quality_rows=len(result["quality"]),
            text_rows=len(result["text_rows"]),
        ),
    )
    status = json.loads((root / "STATUS.json").read_text())
    status.update(
        phase="complete_historical_video_regression",
        next_step="Review gallery; retain independent-test recommendation unchanged.",
    )
    atomic_json(root / "STATUS.json", status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("encode", "score", "report"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    {"encode": encode, "score": score, "report": report}[args.stage](args.root)


if __name__ == "__main__":
    main()
