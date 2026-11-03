"""Build exportable evidence and a local, offline A/B visual review viewer."""

import csv
import html
import json
from pathlib import Path
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.visual_validation import card, psnr, read, MODELS
from benchmarks.super_resolution.training_state import atomic_json


def build(root):
    regions = read(root / "REGIONS.json")
    scenes = read(root / "SCENES.json")
    frames = {r["clip"]: r["frame"] for r in regions}
    notes = [
        dict(
            id="code-improved",
            model="edsr",
            clip="2026-10-15_18-12-43",
            box=[210, 200, 530, 360],
            category="improved",
            title="浅色代码：边缘和杂边改善",
            observation="参考图中的蓝色关键字和黑色变量名边界更接近微调版；base的笔画周围模糊与杂边更明显。此例2倍放大可见，不能据此推算所有文字的可见改善率。",
        ),
        dict(
            id="web-cleaner",
            model="edsr",
            clip="2026-10-15_18-25-36",
            box=[672, 600, 992, 760],
            category="improved",
            title="网页正文：背景更干净，文字仍有损伤",
            observation="微调后标题和正文周围的散点/杂边减少，仍有明显压缩损伤；不能把去杂边等同于文字内容已恢复。",
        ),
        dict(
            id="web-limited",
            model="edsr",
            clip="2026-10-15_17-44-40",
            box=[672, 280, 992, 440],
            category="small_difference",
            title="已严重糊化的表格：两版都恢复有限",
            observation="两版的数字、表头和拖影都明显偏离参考图，字形改善不明显。没有进行OCR字符正确率验证。",
        ),
        dict(
            id="flat-ui",
            model="edsr",
            clip="2026-10-15_18-13-40",
            box=[672, 440, 992, 600],
            category="small_difference",
            title="空白列表背景：涨分不等于读字收益",
            observation="这是空白列表/边界，几乎没有可读文字；微调边界更干净但原尺寸下差别小。局部PSNR提升较大仍不证明读字体验显著改善。",
        ),
        dict(
            id="dark-regressed",
            model="edsr",
            clip="2026-10-15_18-03-53",
            box=[352, 280, 672, 440],
            category="local_regression",
            title="深色等宽字：EDSR局部退步",
            observation="参考图是较细、干净的笔画；微调版部分字符更厚、边缘锯齿更明显，base局部更接近参考。此片段为受控合成桌面页。整段视频的PSNR仍上涨，局部与全局结论不同。",
        ),
        dict(
            id="terminal-regressed",
            model="quicksrnet",
            clip="2026-10-15_18-26-43",
            box=[352, 280, 672, 440],
            category="local_regression",
            title="黄色终端：QuickSRNet未改善且略有退步",
            observation="两版均有严重失真；微调版局部条纹和字符形状未改善，局部分数下降。整段视频PSNR/SSIM也下降。仅凭截图不能证明哪版字符识别准确率更高。",
        ),
        dict(
            id="status-metric-trap",
            model="quicksrnet",
            clip="2026-10-15_18-12-43",
            box=[672, 900, 992, 1060],
            category="metric_visual_disagreement",
            title="细UI边缘：PSNR大幅下降，肉眼影响很局部",
            observation="局部PSNR下降约8.8dB，变化主要集中在底部细线附近的色带/边缘，不能解读为整幅画面灾难性退化。指标变化和可见影响不按比例对应。",
        ),
    ]
    for row in notes:
        row["frame"] = frames[row["clip"]]
        row["card"] = card(
            root,
            row["model"],
            row["clip"],
            row["frame"],
            row["box"],
            row["id"],
            row["title"].split("：")[0] + " / " + row["category"],
        )
        # Re-create card title using an ASCII label for portable PIL font coverage.
        titles = {
            "code-improved": "Code: cleaner text edges",
            "web-cleaner": "Web: cleaner background, residual text damage",
            "web-limited": "Web table: little recovery in either version",
            "flat-ui": "Blank UI: higher PSNR, little reading benefit",
            "dark-regressed": "Dark code: local EDSR regression",
            "terminal-regressed": "Yellow terminal: QuickSRNet regression",
            "status-metric-trap": "Thin UI boundary: large metric drop, localized visual change",
        }
        card(
            root,
            row["model"],
            row["clip"],
            row["frame"],
            row["box"],
            row["id"],
            titles[row["id"]],
        )
        x, y, xx, yy = row["box"]
        folder = root / "renders" / row["model"] / row["clip"]
        reference = np.asarray(
            Image.open(root / "references" / row["clip"] / "reference.png").convert(
                "RGB"
            )
        )[y:yy, x:xx]
        base = np.asarray(Image.open(folder / "base.png").convert("RGB"))[y:yy, x:xx]
        trained = np.asarray(Image.open(folder / "trained.png").convert("RGB"))[
            y:yy, x:xx
        ]
        row.update(
            base_crop_psnr=psnr(reference, base),
            trained_crop_psnr=psnr(reference, trained),
            delta_crop_psnr=psnr(reference, trained) - psnr(reference, base),
        )
        # Short1x display card for convenient inline use.2x detail remains in full card.
        crop_h = 128
        compact = Image.new("RGB", (680, 3 * (crop_h * 2 + 44)), "white")
        draw = ImageDraw.Draw(compact)
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
        for i, (name, image) in enumerate(
            zip(["Reference", "Official base", "Finetuned"], [reference, base, trained])
        ):
            im = Image.fromarray(image[:crop_h])
            draw.text((12, i * 300 + 8), name, fill="black", font=font)
            compact.paste(
                im.resize((640, crop_h * 2), Image.Resampling.NEAREST),
                (12, i * 300 + 36),
            )
        compact.save(root / "cards" / f"{row['id']}-compact.png")
    atomic_json(
        root / "QUALITATIVE-REVIEW.json",
        dict(
            reviewer="Assistant visual inspection, not a human participant study",
            methods="Reference-aware side-by-side inspection at1x and2x nearest; selected examples include favorable, small-change and unfavorable cases.",
            reviewed_examples=notes,
            limits="Examples are exploratory and selected, not random visibility samples. No subjective significance, OCR accuracy or temporal flicker evaluation. PlainUSR/RT4KSR still training at protocol freeze.",
        ),
    )
    counts = []
    for model in MODELS:
        subset = [r for r in regions if r["model"] == model]
        clips = [r for r in scenes["clips"] if r["model"] == model]
        counts.append(
            dict(
                model=model,
                clips=len(clips),
                clips_psnr_up_gt_point1=sum(r["delta_psnr"] > 0.1 for r in clips),
                clips_psnr_down_lt_minus_point1=sum(
                    r["delta_psnr"] < -0.1 for r in clips
                ),
                clips_lpips_worse=sum(r["delta_lpips"] > 0 for r in clips),
                local_regions=len(subset),
                regions_up_gt_point2=sum(r["delta_psnr"] > 0.2 for r in subset),
                regions_small_abs_le_point2=sum(
                    abs(r["delta_psnr"]) <= 0.2 for r in subset
                ),
                regions_down_lt_minus_point2=sum(
                    r["delta_psnr"] < -0.2 for r in subset
                ),
            )
        )
    atomic_json(
        root / "SCREENING-COUNTS.json",
        dict(
            counts=counts,
            not_visibility_rates=True,
            notes="Fixed-grid regions overlap in places and are not independent. Thresholds ±0.1dB clip / ±0.2dB ROI are only exploratory numeric screening, not perceptual/JND thresholds.",
        ),
    )
    with (root / "region-metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(regions[0]))
        w.writeheader()
        w.writerows(regions)
    # Encode all viewer paths as relative local files; no network/API dependencies.
    clips = sorted({r["clip"] for r in regions})
    contents = {r["clip"]: r["content"] for r in regions}
    data = json.dumps(
        dict(
            models=MODELS,
            clips=[dict(id=c, content=contents[c], frame=frames[c]) for c in clips],
            notes=notes,
            scenes=scenes["clips"],
        ),
        ensure_ascii=False,
    )
    template = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>SR视觉核验</title>
<style>body{font:16px system-ui;max-width:1450px;margin:24px auto;padding:0 16px;background:#f4f5f7;color:#18212b}button,select{font:inherit;padding:8px;margin:4px}article{background:white;padding:18px;margin:20px 0;border-radius:10px}img{max-width:100%;height:auto}.panel{overflow:auto;max-height:80vh;background:#ddd}.panel img{max-width:none;display:block}p{line-height:1.65}.tools{position:sticky;top:0;background:white;padding:12px;z-index:2}small{color:#52606d}</style>
<h1>官方Base vs 专项微调：视觉核验</h1><p>24段Mac validation，每段同一中间时刻；720p /600Kbps /420，统一1080p输出。仅评估已完成的5模型。下方A/B顺序固定打散，不预先显示哪张是微调。可原尺寸切换核对，再揭示。它是自查工具，不是已经完成的真人盲测。所有资源在本地，无网络请求。</p>
<div class="tools"><select id="model"></select><select id="clip"></select><button id="a">看A</button><button id="b">看B</button><button id="ref">看Reference</button><button id="reveal">揭示A/B</button><select id="zoom"><option value="1">100%原尺寸</option><option value="0.5">50%概览</option><option value="2">200%最近邻</option></select><b id="label"></b><p id="metric"></p></div>
<div class="panel"><img id="image"></div><p><small>100%对应文件像素；实际视觉角度仍取决于屏幕、系统缩放和观看距离。左右方向键切换A/B。视频时间闪烁和OCR未测试。</small></p>
<h2>已核查的正例、小变化和反例</h2><div id="notes"></div>
<script>const D=__DATA__;let reveal=false,current='a';const $=id=>document.getElementById(id);D.models.forEach(v=>$('model').add(new Option(v,v)));D.clips.forEach(v=>$('clip').add(new Option(v.content+' / '+v.id,v.id)));
function order(){let s=$('model').value+$('clip').value;return [...s].reduce((v,c)=>v+c.charCodeAt(0),0)%2?['base','trained']:['trained','base']}
function show(which){current=which;let m=$('model').value,c=$('clip').value,o=order(),phase=which==='ref'?'reference':o[which==='a'?0:1];$('image').src=phase==='reference'?'references/'+c+'/reference.png':'renders/'+m+'/'+c+'/'+phase+'.png';$('image').style.width=(1920*Number($('zoom').value))+'px';$('image').style.imageRendering=Number($('zoom').value)>1?'pixelated':'auto';$('label').textContent=which==='ref'?'Reference':which.toUpperCase()+(reveal?' = '+phase:'');let row=D.scenes.find(r=>r.model===m&&r.clip===c);$('metric').textContent=reveal?'整段3帧平均PSNR：'+row.base_psnr.toFixed(3)+' → '+row.trained_psnr.toFixed(3)+' dB；这不是当前单帧/局部分数。':'先比较，再揭示；未用PSNR给你提示哪张应该更好。'}
['model','clip'].forEach(id=>$(id).onchange=()=>{reveal=false;show('a')});$('zoom').onchange=()=>show(current);$('a').onclick=()=>show('a');$('b').onclick=()=>show('b');$('ref').onclick=()=>show('ref');$('reveal').onclick=()=>{reveal=true;show(current)};document.onkeydown=e=>{if(e.key==='ArrowLeft')show('a');if(e.key==='ArrowRight')show('b')};
for(const n of D.notes){const a=document.createElement('article');const title=document.createElement('h3');title.textContent=n.title;const p=document.createElement('p');p.textContent=n.observation+' 局部PSNR：'+n.base_crop_psnr.toFixed(2)+' → '+n.trained_crop_psnr.toFixed(2)+' dB。';const im=document.createElement('img');im.src=n.card;im.loading='lazy';a.append(title,p,im);$('notes').append(a)}show('a');</script></html>"""
    (root / "visual-review.html").write_text(template.replace("__DATA__", data))
    # Standalone scientific figures: aggregate gain and full-frame/ROI distinction.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    contents_order = ["code", "web", "terminal", "mixed"]
    x = np.arange(4)
    for i, model in enumerate(MODELS):
        vals = [
            next(
                v["delta_psnr"]
                for v in scenes["content_summary"]
                if v["model"] == model and v["content"] == c
            )
            for c in contents_order
        ]
        ax.bar(x + (i - 2) * 0.16, vals, width=0.16, label=model)
    ax.set_xticks(x, contents_order)
    ax.set_ylabel("Mean clip PSNR gain over official base (dB)")
    ax.set_title("Mac validation: scene averages hide local regressions")
    ax.legend(ncol=5)
    ax.axhline(0, color="black", lw=0.7)
    fig.tight_layout()
    for ext in ["png", "svg"]:
        fig.savefig(root / f"scene-gains.{ext}", dpi=180)
    plt.close(fig)
    lines = [
        "# Validation visual audit",
        "",
        "Scope: five completed models, official pretrained vs best Mac-validation-selected checkpoint.24 clips,72 metric frames/model;24 matched middle-frame image comparisons/model;533 reference-selected grid regions/model (2,665 total). PlainUSR and RT4KSR are pending this audit.",
        "",
        "This is assistant qualitative inspection plus paired metrics, not a completed human participant/blind study. A/B viewer is available for user inspection; no subjective significance, OCR accuracy or temporal-quality claims. The Mac source is a processed RGB8 proxy,24 recordings share one session, and several clips show controlled synthetic pages. Test is untouched.",
        "",
        "[Open offline A/B viewer](visual-review.html)",
        "",
        "| Model | Code ΔPSNR | Web ΔPSNR | Terminal ΔPSNR | Mixed ΔPSNR |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in MODELS:
        vals = [
            next(
                v["delta_psnr"]
                for v in scenes["content_summary"]
                if v["model"] == model and v["content"] == c
            )
            for c in contents_order
        ]
        lines.append(
            "| " + model + " | " + " | ".join(f"{v:+.3f}" for v in vals) + " |"
        )
    lines += [
        "",
        "These are average image-error changes, not visibility rates. QuickSRNet regresses on one terminal clip by0.519dB on average, with SSIM also worse; its LPIPS worsens on3/24 clips. EDSR/IMDN/EfRLFN improve full-clip PSNR/SSIM/LPIPS on all24 clips, but local regions can still be worse.",
        "",
        "| Model | Clip gain >0.1dB | Clip loss >0.1dB | ROI gain >0.2dB | ROI small change | ROI loss >0.2dB |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c in counts:
        lines.append(
            f"| {c['model']} | {c['clips_psnr_up_gt_point1']}/24 | {c['clips_psnr_down_lt_minus_point1']}/24 | {c['regions_up_gt_point2']}/533 | {c['regions_small_abs_le_point2']}/533 | {c['regions_down_lt_minus_point2']}/533 |"
        )
    lines += [
        "",
        "Thresholds are numeric screening conventions only. Fixed-grid regions overlap and are not independent. Counts must not be described as visually improved/degraded percentages.",
        "",
        "![Scene PSNR gains](scene-gains.png)",
        "",
    ]
    for n in notes:
        lines += [
            f"## {n['title']}",
            "",
            n["observation"],
            f"\n{n['model']}, {n['clip']}, frame{n['frame']}; crop{n['box']}; local PSNR {n['base_crop_psnr']:.3f} → {n['trained_crop_psnr']:.3f}dB.",
            "",
            f"![{n['id']}]({n['card']})",
            "",
        ]
    lines += [
        "## Practical findings",
        "",
        "The current fine-tuning improves many light-background code and mixed UI regions, including reduced halos/noise, but this is not universal glyph recovery. Severe compression damage in the web-table example remains. Dark code strokes and thin color/UI boundaries expose local regressions missed by full-frame averages. Future experiments should explicitly retain dark/light themes, terminals, color boundaries and fine-line patterns, and assess OCR and temporal behavior separately. No additional training or test-set evaluation was started by this audit.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    build(p.parse_args().root.resolve())
