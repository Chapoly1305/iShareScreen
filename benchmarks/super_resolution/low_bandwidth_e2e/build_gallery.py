"""Build a browsable fixed and worst-case test gallery without filtering rows."""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote

from benchmarks.super_resolution.low_bandwidth_e2e.analyze_test import load_rows
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def link(path: str) -> str:
    return "../" + quote(path, safe="/")


def esc(value: object) -> str:
    return html.escape(str(value))


def main() -> None:
    root = Path(
        "benchmarks/super_resolution/artifacts/text-low-bandwidth-e2e-v1"
    ).resolve()
    protocol = read(root / "EVAL_PROTOCOL.json")
    selection = read(root / "FINAL_SELECTION.json")
    analysis = read(root / "test-analysis.json")
    method_folders = selection["test_result_folders"]
    if set(analysis["methods"]) != {"baseline", *method_folders.values()}:
        raise ValueError("Incomplete frozen test roster")
    rows = {"baseline": load_rows(root, root / "test-baseline", True)}
    rows.update(
        {
            method: load_rows(root, root / "test-ocr-results" / folder, False)
            for method, folder in method_folders.items()
        }
    )
    display_methods = [
        "baseline",
        "official_quicksrnet_x2",
        selection["best_new_seed"],
        selection["validation_rate_winners"]["600000"]["method"],
    ]
    display_methods = list(dict.fromkeys(display_methods))
    style = """body{font:15px/1.45 system-ui,sans-serif;max-width:1500px;margin:auto;padding:24px;color:#15202b;background:#f7f9fc}h1,h2{color:#153c65}table{border-collapse:collapse;background:white;margin:12px 0;width:100%}th,td{border:1px solid #ccd5df;padding:6px;text-align:left;vertical-align:top}thead{background:#e7f0f8}.crop{max-width:100%;image-rendering:pixelated;background:white;border:1px solid #b9c8d8}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}.card{background:white;padding:10px;border:1px solid #ccd5df;overflow-wrap:anywhere}.truth{font-family:monospace;white-space:pre-wrap}.bad{color:#a12924}details{background:#fff;border:1px solid #ccd5df;margin:12px 0;padding:8px}summary{cursor:pointer;font-weight:700}video{width:100%;max-width:1200px}"""
    parts = [
        '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>iShareScreen independent text test</title>',
        "<style>"
        + style
        + "</style><main><h1>iShareScreen 低带宽文字：独立测试图库</h1>",
        "<p>所有 72 个预先冻结测试案例、六个码率/分辨率单元和两帧都参与机器指标。下方固定展示和最差案例仅帮助检查笔画、数字、标点和路径；评估结果未据此筛选或重算。原生裁剪使用像素显示，放大时不插值。</p>",
    ]
    parts.append(
        "<h2>机器指标</h2><table><thead><tr><th>方法</th><th>基础分辨率</th><th>目标码率</th><th>实测视频码率均值</th><th>CER</th><th>WER</th><th>整行准确率</th><th>数字错误</th><th>标点错误</th></tr></thead><tbody>"
    )
    for method, folder in [("baseline", "baseline"), *method_folders.items()]:
        for cell in analysis["methods"][folder]["cells"]:
            parts.append(
                "<tr>"
                + "".join(
                    f"<td>{esc(x)}</td>"
                    for x in (
                        method,
                        f"{cell['base_height']}p",
                        f"{cell['requested_bps'] // 1000} Kbps",
                        f"{cell['measured_video_bps_mean'] / 1000:.0f} Kbps",
                        f"{cell['pooled_cer']:.3f}",
                        f"{cell['pooled_wer']:.3f}",
                        f"{cell['exact_line_rate']:.3f}",
                        cell["digit_edits"],
                        cell["punctuation_edits"],
                    )
                )
                + "</tr>"
            )
    parts.append("</tbody></table>")
    media_path = root / "comparison-media/COMPLETED.json"
    if media_path.exists():
        media = read(root / "comparison-media/results.json")
        inspection_path = root / "comparison-media/temporal-inspection/INSPECTION.json"
        inspection = (
            {record["video"]: record for record in read(inspection_path)["records"]}
            if inspection_path.exists()
            else {}
        )
        parts.append(
            "<h2>预先选定的视频对比</h2><p>左侧是未增强的 FHD 展示，右侧是验证阶段选中的模型。MP4 仅供浏览；文字评分使用原始像素。</p>"
        )
        for item in media["rows"]:
            detail = inspection.get(item["video"])
            parts.append(
                f'<details><summary>{esc(item["source_kind"])} · {esc(item["case"])} · {item["requested_bps"] // 1000} Kbps · {esc(item["method"])}</summary><video controls preload="none" src="{link("comparison-media/" + item["video"])}"></video><p>文字框相邻帧 RGB 差均值：未增强 {item["mean_text_crop_temporal_abs_rgb_base"]:.2f}；增强 {item["mean_text_crop_temporal_abs_rgb_model"]:.2f}。此值仅为闪烁辅助指标。</p>'
                + (
                    f'<p>自动最差相邻帧 {detail["worst_second_frame"]}：上排前帧，下排后帧，左原始右增强。</p><img class="crop" loading="lazy" src="{link(detail["crop"])}">'
                    if detail
                    else ""
                )
                + "</details>"
            )
    parts.append("<h2>冻结展示案例：原生文字裁剪</h2>")
    for case_id in protocol["fixed_display_ids"]["test"]:
        parts.append(f"<details><summary>{esc(case_id)}</summary>")
        case = next(c for c in protocol["cases"] if c["id"] == case_id)
        parts.append(
            f'<p>来源组：{esc(case["source_group"])}；真值：<span class="truth">{esc(case["line"]["text"])}</span></p>'
        )
        for bps in (600000, 1000000):
            for height in (540, 720, 1080):
                key = (case_id, height, bps, 60)
                parts.append(
                    f'<h3>{height}p / {bps // 1000} Kbps / 帧 60</h3><div class="cards">'
                )
                for method in display_methods:
                    row = rows[method][key]
                    parts.append(
                        f'<div class="card"><b>{esc(method)}</b><br><img class="crop" loading="lazy" src="{link(row["native_crop"])}"><p>OCR：<span class="truth">{esc(row["prediction"].strip())}</span></p><p>CER {row["cer"]:.3f}</p></div>'
                    )
                parts.append("</div>")
        parts.append("</details>")
    parts.append(
        "<h2>自动汇总的最差案例</h2><p>按相同样本相对未增强 CER 增幅排序，每个模型展示前 20 条。所有案例仍保留在总体指标中。</p>"
    )
    for method, folder in method_folders.items():
        parts.append(f"<details><summary>{esc(method)}</summary>")
        for worst in analysis["methods"][folder]["overall"]["worst_paired_cases"]:
            key = (
                worst["id"],
                worst["base_height"],
                worst["requested_bps"],
                worst["frame"],
            )
            before = rows["baseline"][key]
            after = rows[method][key]
            parts.append(
                f'<div class="card"><b>{esc(worst["id"])} · {worst["base_height"]}p · {worst["requested_bps"] // 1000} Kbps · 帧 {worst["frame"]}</b><p>真值：<span class="truth">{esc(worst["ground_truth"])}</span></p><div class="cards"><div>未增强 CER {before["cer"]:.3f}<br><img class="crop" loading="lazy" src="{link(before["native_crop"])}"><p class="truth">{esc(before["prediction"].strip())}</p></div><div>增强 CER {after["cer"]:.3f}<br><img class="crop" loading="lazy" src="{link(after["native_crop"])}"><p class="truth bad">{esc(after["prediction"].strip())}</p></div></div></div>'
            )
        parts.append("</details>")
    parts.append("</main></html>")
    output = root / "gallery"
    output.mkdir(parents=True, exist_ok=True)
    page = output / "index.html"
    page.write_text("\n".join(parts))
    atomic_json(
        output / "GALLERY.json",
        dict(
            schema="isharescreen_low_bandwidth_test_gallery_v1",
            html_sha256=sha256(page),
            analysis_sha256=sha256(root / "test-analysis.json"),
            selection_sha256=sha256(root / "FINAL_SELECTION.json"),
            fixed_display_ids=protocol["fixed_display_ids"]["test"],
            method_folders=method_folders,
            worst_per_method=20,
            videos=len(read(root / "comparison-media/results.json")["rows"])
            if media_path.exists()
            else 0,
        ),
    )
    print("GALLERY", page)


if __name__ == "__main__":
    main()
