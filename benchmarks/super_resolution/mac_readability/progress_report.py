"""Report real-link failure modes and the independent development baseline."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import html
import json
from pathlib import Path
import random

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.mac_readability.report import quality
from benchmarks.super_resolution.mac_readability.quality import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    truth = json.loads((args.source / "manifest.json").read_text())
    states = {s["state_id"]: s for s in truth["states"]}
    native = json.loads((root / "dense-native-quality-v1/SUMMARY.json").read_text())
    models = json.loads((root / "dense-model-quality-v1/SUMMARY.json").read_text())
    summaries = native + models
    images = (
        json.loads((root / "dense-codec-quality-v1/IMAGES.json").read_text())["images"]
        + json.loads((root / "dense-models-v1/IMAGES-LOCAL.json").read_text())["images"]
    )
    tasks = [
        (r, states[r["state_id"]], args.source)
        for r in images
        if r["config"] != "clean"
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        pixels = list(pool.map(quality, tasks))
    save(root / "DENSE-PIXEL-QUALITY.json", pixels)
    names = {
        "reference": "Clean reference",
        "bicubic": "720p bicubic",
        "native": "1080p native",
        "edsr-official": "EDSR official",
        "edsr-trained": "EDSR trained",
        "imdn-official": "IMDN official",
        "imdn-trained": "IMDN trained",
    }
    order = list(names)
    table = []
    for key in order:
        score = next(s for s in summaries if s["unit"] == "line" and s["method"] == key)
        rows = [r for r in pixels if r["method"] == key]
        table.append(
            dict(
                method=key,
                name=names[key],
                cer=score["macro_cer"],
                exact=score["exact_units"],
                lines=score["units"],
                roi_psnr=float(np.mean([r["text_roi_psnr"] for r in rows]))
                if rows
                else None,
                roi_ssim=float(np.mean([r["text_roi_ssim"] for r in rows]))
                if rows
                else None,
            )
        )
    categories = {}
    for path in [
        "dense-native-quality-v1/CATEGORY-SUMMARY.json",
        "dense-model-quality-v1/CATEGORY-SUMMARY.json",
    ]:
        for category, rows in json.loads((root / path).read_text()).items():
            categories.setdefault(category, []).extend(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), layout="constrained")
    axes[0].barh(
        [r["name"] for r in table],
        [r["cer"] * 100 for r in table],
        color=[
            "#8795a5",
            "#b3b9c0",
            "#4f8d71",
            "#486fa9",
            "#b46c47",
            "#779bc9",
            "#d0a07a",
        ],
    )
    axes[0].invert_yaxis()
    axes[0].set(
        xlabel="Equal-document macro CER (%)",
        title="16 development document groups, 14/18 px fonts",
    )
    x = np.arange(4)
    width = 0.2
    for offset, key in enumerate(
        ["bicubic", "edsr-official", "edsr-trained", "imdn-trained"]
    ):
        values = [
            next(
                r["macro_cer"]
                for r in categories[c]
                if r["method"] == key and r["unit"] == "line"
            )
            * 100
            for c in ["code", "table", "english", "chinese"]
        ]
        axes[1].bar(x + (offset - 1.5) * width, values, width, label=names[key])
    axes[1].set(
        xticks=x,
        xticklabels=["Code", "Tables", "English", "Chinese"],
        ylabel="Macro CER (%)",
        title="Retain category regressions",
    )
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="x" if ax == axes[0] else "y", alpha=0.15)
    for ext in ["png", "svg"]:
        fig.savefig(root / f"dense-baseline-curves.{ext}", dpi=170)
    plt.close(fig)
    links = json.loads((root / "LINK-COMPARISON-V3.json").read_text())["runs"]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for row, label in [
        (links[0], "Native3360, unshaped"),
        (links[3], "Virtual720, 600 Kbps / 250ms queue"),
    ]:
        values = [
            r["first_both_markers_ms"] for r in row["responses"] if not r["warmup"]
        ]
        ax.plot(range(1, 8), values, marker="o", label=label)
    ax.set(
        xlabel="Source change",
        ylabel="Draw to decoded marker agreement (ms)",
        title="Actual Mac sessions; different display geometry, same stimulus script",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    for ext in ["png", "svg"]:
        fig.savefig(root / f"link-response-curves.{ext}", dpi=170)
    plt.close(fig)
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 18)
    gallery = []
    rng = random.Random(20261102)
    for state_id, box, name in [
        (7, (32, 72, 912, 324), "code"),
        (13, (32, 72, 912, 324), "table"),
        (31, (32, 72, 912, 324), "chinese"),
    ]:
        methods = ["reference", "bicubic", "edsr-official", "edsr-trained"]
        rng.shuffle(methods)
        w, h = box[2] - box[0], box[3] - box[1]
        canvas = Image.new("RGB", (w, (h + 30) * 4), "#edf1f5")
        draw = ImageDraw.Draw(canvas)
        mapping = {}
        for i, key in enumerate(methods):
            row = next(
                r for r in images if r["state_id"] == state_id and r["method"] == key
            )
            letter = chr(65 + i)
            mapping[letter] = names[key]
            draw.text((8, i * (h + 30) + 5), letter, fill="#152332", font=font)
            canvas.paste(
                Image.open(row["path"]).convert("RGB").crop(box), (0, i * (h + 30) + 30)
            )
        filename = "dense-blind-" + name + ".png"
        canvas.save(root / filename)
        gallery.append(dict(path=filename, state_id=state_id, box=box, mapping=mapping))
    selection = json.loads((root / "COMPARATOR-FROZEN.json").read_text())
    result = dict(
        schema="mac_phase2_report_v1",
        goal_active=True,
        quality_target_achieved=False,
        script_sha256=sha(Path(__file__)),
        table=table,
        categories=categories,
        links=links,
        comparator=selection,
        gallery=gallery,
        important_limit="Codec controls and live Screen Sharing are separate conditions. Virtual720 changes the source raster; its latency is not a same-geometry quality comparison.",
    )
    save(root / "DENSE-AND-LINK-REPORT.json", result)
    body = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Mac low-bandwidth text: measured findings</title>',
        "<style>body{font:16px system-ui;line-height:1.6;max-width:1120px;margin:36px auto;padding:0 24px;color:#172638}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #ccd4dd;padding:9px;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;height:auto}.status{background:#edf4ff;padding:14px}details{margin-bottom:30px}</style>",
        '<h1>Real Mac link limits and independent text baseline</h1><p class="status">Goal active. The independent 25% error-reduction target has not been achieved. No new model has been trained in this phase. Historical final test remains untouched.</p>',
        "<h2>What the limited-link session actually did</h2><p>The emulator shares one 600,000-bit/s FIFO across incoming media and control, and a separate 600,000-bit/s FIFO for outgoing traffic. It charges UDP payload plus 28 bytes and TCP application segments plus 40 bytes, with 10 ms one-way delay. It is a socket-boundary experiment, excluding physical TCP ACK/retransmission, Wi-Fi and VPN overhead. Mac saturated-packet calibration measured 600.05–600.81 Kbps over 4-second windows; packetization accounts for the small excess. Short-window bursts are bounded by one packet/MTU.</p>",
        "<table><tr><th>Mac session</th><th>Actual canvas</th><th>Queue budget</th><th>Connected</th><th>Current updates</th><th>Response p95</th></tr>",
    ]
    labels = [
        "Unshaped physical mirror",
        "600 Kbps physical mirror",
        "600 Kbps physical mirror",
        "600 Kbps virtual desktop",
    ]
    for row, label in zip(links, labels):
        response = row.get("warm_response_summary")
        latency = (
            response["conditional_ms"]["p95"]
            if response and response["conditional_ms"]
            else None
        )
        body.append(
            f"<tr><td>{label}</td><td>{html.escape(str(row['actual_canvas']))}</td><td>{row.get('link_config', {}).get('queue_ms', '—')} ms</td><td>{row['state']}</td><td>{str(response['current_updates']) + '/' + str(response['changes']) if response else 'No session'}</td><td>{f'{latency:.1f} ms' if latency is not None else 'Unavailable'}</td></tr>"
        )
    body += [
        "</table><p>The short-queue physical mirror failed during startup. Enlarging its queue allowed connection but none of seven post-warmup source changes arrived before the next change. The virtual desktop really transmitted 1280×720 and delivered all seven changes; its tail remains over one second. These are single 30-second Mac-local runs, not a physical WAN or RTX 4090 receiver result. Virtual mode changes source raster and blanks the physical display while connected; the bounded session ended and the Mac was verified unlocked afterward.</p>",
        '<img src="link-response-curves.png" alt="Actual source response samples">',
        "<h2>Readability under actual Mac hardware encoding</h2><p>Sixteen source-document groups, two native font sizes 14/18 px,1,046 scored lines and 1,806 separate table cells per method. The code documents come from repository-isolated CodeSCAN validation; prose and tables are project-authored. No sample enters training or final test. Sample each state at exactly 2.8 seconds, validate both state markers, and compare at 1920×1080. The codec request is 600 Kbps/420; this section is independent of the network experiment above.</p>",
        "<p>Five of 32 encoded clips had 175–179 frames instead of 180. All 64 prespecified 2.8/5.8-second sample timestamps exist exactly and match the source markers. Their real frame drops remain reported; no re-encoding, frame substitution or smooth 30 fps claim was used for these scores. The strict full-frame-count results remain preserved in the codec manifest.</p>",
        "<table><tr><th>Method</th><th>Macro CER ↓</th><th>Exact nonspace lines</th><th>Text ROI PSNR</th><th>Text ROI SSIM</th></tr>",
    ]
    for row in table:
        psnr = f"{row['roi_psnr']:.2f} dB" if row["roi_psnr"] is not None else "—"
        ssim = f"{row['roi_ssim']:.4f}" if row["roi_ssim"] is not None else "1.0000"
        body.append(
            f"<tr><td>{row['name']}</td><td>{row['cer'] * 100:.3f}%</td><td>{row['exact']}/{row['lines']}</td><td>{psnr}</td><td>{ssim}</td></tr>"
        )
    body += [
        '</table><img src="dense-baseline-curves.png" alt="Independent development CER and category results">',
        "<p>The previous EDSR fine-tune improves code and table OCR, but Chinese CER rises from 2.94% to 8.16%. Its higher exact-line count does not offset large errors in some Chinese lines. OCR is a diagnostic, not proof of human preference. Clean-image OCR also makes errors; its 1.82% value is reported rather than removed.</p>",
        f"<p>The comparator is now frozen as official EDSR, selected from the four existing model controls before new training. Fifteen document groups have nonzero baseline error; the remaining zero-error document is retained as a regression guard. Hard-set CER is {selection['hard_macro_cer'] * 100:.3f}%; the 25% target requires at most {selection['target_hard_cer'] * 100:.3f}%. Source-group bootstrap and full-set/zero-error regressions remain required for a new candidate.</p>",
        "<h2>Native-pixel visual inspection</h2><p>The following crops are randomized within each example. No enlargement or sharpening was added. Open the PNG at 100% to inspect original pixels; reveal the key afterward.</p>",
    ]
    for item in gallery:
        body.append(
            f'<p><a href="{item["path"]}">{item["path"]}</a></p><img src="{item["path"]}" alt="Randomized native-pixel comparison"><details><summary>Reveal panel identities</summary><p>{html.escape(str(item["mapping"]))}</p></details>'
        )
    body += [
        "<h2>Next experiment</h2><p>A separate 384-source train-only derivative has been encoded on the same M1 hardware at 720p/600 Kbps/420. It covers 24 languages and 384 distinct training repositories. Eight smoke examples passed exact-time, 16-pixel alignment and Mac/Linux decoding checks. New training should start from the stronger official EDSR comparator, preserve Chinese fidelity, and compare matched-codec loss variants against the historical fine-tune. An additional 128 native Mac glyph variants from 64 new train-only document families have now been captured and encoded. Character frequencies come only from frozen training sources; these random glyph strings are reconstruction training material, not an evaluation corpus. Existing data splits, models and final tests remain frozen.</p>",
        "<p>The earlier RTX 4090 display fix remains separately measured: added first-display p95 0.40ms and stable-to-refinement p95 78.28ms over 24 warm local updates. These local measurements do not include the network delays above.</p>",
        '<p><a href="REPORT.html">Earlier calibration and decoder/display evidence</a></p></html>',
    ]
    (root / "NETWORK-AND-DENSE.html").write_text("\n".join(body) + "\n")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
