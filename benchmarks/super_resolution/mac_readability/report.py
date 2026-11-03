"""Render the first Mac calibration evidence without claiming goal completion."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity

from benchmarks.super_resolution.mac_readability.quality import sha, save


def quality(task):
    row, state, source = task
    reference = np.asarray(
        Image.open(source / state["reference"]["path"]).convert("RGB")
    )
    image = np.asarray(Image.open(row["path"]).convert("RGB"))
    assert image.shape == reference.shape
    mask = np.zeros(reference.shape[:2], bool)
    ssim = []
    weights = []
    for label in state["labels"]:
        x, y, w, h = label["xywh"]
        mask[y : y + h, x : x + w] = True
        ssim.append(
            structural_similarity(
                reference[y : y + h, x : x + w],
                image[y : y + h, x : x + w],
                channel_axis=2,
                data_range=255,
            )
        )
        weights.append(w * h)
    delta = (reference.astype(np.float32) - image.astype(np.float32)) ** 2
    psnr = lambda mse: 10 * math.log10(255**2 / mse) if mse else None
    return {k: row[k] for k in ["id", "config", "method", "state_id", "group"]} | dict(
        full_psnr=psnr(float(delta.mean())),
        text_roi_psnr=psnr(float(delta[mask].mean())),
        text_roi_ssim=float(np.average(ssim, weights=weights)),
        image_sha256=sha(row["path"]),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    source = args.data / "normalized-reference-v2"
    truth = json.loads((source / "manifest.json").read_text())
    control = json.loads((root / "codec-quality-v1/SUMMARY.json").read_text())
    model = json.loads((root / "model-quality-v2/SUMMARY.json").read_text())
    actual = json.loads((root / "actual-quality-v1/SUMMARY.json").read_text())
    traffic = json.loads((args.data / "screen-sharing009/TRAFFIC.json").read_text())
    decoder = json.loads((root / "DECODER-COMPARISON.json").read_text())
    inputs = json.loads((root / "codec-quality-v1/IMAGES.json").read_text())["images"]
    outputs = json.loads((root / "codec-models-v2/IMAGES-LOCAL.json").read_text())[
        "images"
    ]
    methods = [
        "bicubic",
        "edsr-official",
        "edsr-trained",
        "imdn-official",
        "imdn-trained",
    ]
    names = {
        "bicubic": "Bicubic",
        "edsr-official": "EDSR official",
        "edsr-trained": "EDSR trained",
        "imdn-official": "IMDN official",
        "imdn-trained": "IMDN trained",
    }
    chosen = [
        r
        for r in inputs + outputs
        if r.get("config") == "720p-600k" and r["source_age_s"] == 2.8
    ]
    tasks = [
        (r, next(s for s in truth["states"] if s["state_id"] == r["state_id"]), source)
        for r in chosen
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        q = list(pool.map(quality, tasks))
    save(root / "QUALITY-STEADY.json", q)
    table = []
    for method in methods:
        r = next(
            s
            for s in control + model
            if s["method"] == method
            and s["config"] == "720p-600k"
            and s["unit"] == "line"
            and s["source_age_s"] == 2.8
        )
        quality_rows = [s for s in q if s["method"] == method]
        table.append(
            dict(
                method=method,
                name=names[method],
                cer_percent=r["macro_cer"] * 100,
                exact_lines=r["exact_units"],
                lines=r["units"],
                group_cer=r["groups"],
                text_roi_psnr_db=float(
                    np.mean([s["text_roi_psnr"] for s in quality_rows])
                ),
                text_roi_ssim=float(
                    np.mean([s["text_roi_ssim"] for s in quality_rows])
                ),
                full_psnr_db=float(np.mean([s["full_psnr"] for s in quality_rows])),
            )
        )
    clean = next(r for r in control if r["config"] == "clean" and r["unit"] == "line")
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    for method in methods:
        rows = sorted(
            [
                r
                for r in control + model
                if r["method"] == method
                and r["config"] == "720p-600k"
                and r["unit"] == "line"
            ],
            key=lambda x: x["source_age_s"],
        )
        axes[0].plot(
            [r["source_age_s"] for r in rows],
            [r["macro_cer"] * 100 for r in rows],
            marker="o",
            label=names[method],
        )
    axes[0].axhline(
        clean["macro_cer"] * 100, ls=":", color="black", label="Clean-image OCR error"
    )
    axes[0].set(
        xlabel="Time after source change (s)",
        ylabel="Character error rate (%)",
        title="Mac hardware codec controls: 720p / 600 Kbps / 420",
    )
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].grid(alpha=0.2)
    bins = traffic["fixed_windows"]["1.0"]
    axes[1].plot(
        [r["start_s"] + 0.5 for r in bins],
        [r["incoming_payload_kbps"] for r in bins],
        label="All received socket payload",
    )
    axes[1].plot(
        [r["start_s"] + 0.5 for r in bins],
        [
            r["incoming_payload_kbps"] + r["udp_ipv4_overhead_estimated_kbps"]
            for r in bins
        ],
        label="+ estimated UDP/IPv4 overhead",
    )
    axes[1].axhline(600, color="#bd3b31", ls="--", label="600 Kbps video request")
    axes[1].set(
        xlabel="Capture time (s)",
        ylabel="Kbps in fixed 1 s bins",
        title="Actual Screen Sharing: native canvas, unshaped loopback",
    )
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)
    fig.savefig(root / "calibration-curves.png", dpi=170)
    fig.savefig(root / "calibration-curves.svg")
    plt.close(fig)
    font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), 18)
    pictures = []
    for state_id, box, filename in [
        (1, (52, 688, 952, 960), "visual-code-fonts.png"),
        (5, (52, 332, 952, 488), "visual-paper-chinese.png"),
    ]:
        state = next(s for s in truth["states"] if s["state_id"] == state_id)
        panels = [("Clean source", str(source / state["reference"]["path"]))]
        for method in ["bicubic", "edsr-official", "edsr-trained", "imdn-trained"]:
            row = next(
                r for r in chosen if r["state_id"] == state_id and r["method"] == method
            )
            panels.append((names[method], row["path"]))
        w, h = box[2] - box[0], box[3] - box[1]
        canvas = Image.new("RGB", (w, (h + 34) * len(panels)), "#edf1f5")
        draw = ImageDraw.Draw(canvas)
        for i, (label, path) in enumerate(panels):
            draw.text((8, i * (h + 34) + 7), label, fill="#152332", font=font)
            canvas.paste(
                Image.open(path).convert("RGB").crop(box), (0, i * (h + 34) + 34)
            )
        canvas.save(root / filename)
        pictures.append(filename)
    result = dict(
        schema="mac_readability_calibration_report_v1",
        goal_status="active",
        script_sha256=sha(Path(__file__)),
        complete_calibration=True,
        independent_quality_goal_met=False,
        clean_ocr_cer_percent=clean["macro_cer"] * 100,
        table=table,
        decoder_comparison=decoder,
        limitations=[
            "Eight project-authored calibration states only; no generalization or human preference claim",
            "600 Kbps is an encoder request; controls use less and real Screen Sharing exceeds it",
            "Screen Sharing ignores requested 720p canvas in physical mirroring; actual canvas 3360x2100",
            "One 1080p paper control has 179/180 frames and remains in diagnostic outputs only",
            "Real-capture source-state matching does not imply current content; actual selected ages are retained",
            "Mac loopback backlog in timing013/queue014 was resolved by native 420 decoding. The seven-change measurements do not establish WAN or RTX4090 receiver latency",
            "Later timing runs 010-012 occurred after the Mac locked and are invalid; never use them as latency evidence",
        ],
        required_next=[
            "Dense independent text/document families with source truth and group isolation",
            "Fixed 600 Kbps link emulation including video/control with declared queue and overhead accounting",
            "Validate real Mac degradation before training four ablations",
            "Report English/Chinese, punctuation/digits, zero-error regressions and local timing separately",
        ],
    )
    save(root / "CALIBRATION.json", result)
    body = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Mac readability calibration</title>',
        "<style>body{font:16px system-ui;max-width:1120px;margin:40px auto;padding:0 24px;line-height:1.6;color:#152332}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #ccd3dc;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%;height:auto}p{max-width:920px}.status{background:#edf4ff;padding:14px}</style>",
        '<h1>Mac text-readability calibration</h1><p class="status">Goal active. Capture and measurement qualification completed; no claim that the independent readability target has been achieved.</p>',
        "<p>Actual M1 Mac VideoToolbox encoding, 720p / requested 600 Kbps / HEVC 4:2:0. Four content groups, two known text states per group. Every method uses identical decoded inputs and is displayed at 1080p. The selected EDSR and IMDN checkpoints are from the completed 100-epoch campaign. GPU inference ran on the lab RTX 4090 in AMP FP16.</p>",
        "<h2>Steady-state text scores</h2><p>At 2.8 seconds after each source change. Character errors are pooled within each content group and groups receive equal weight. Case, digits and punctuation are retained; whitespace is excluded. Exact lines use the same non-whitespace convention.</p>",
        f"<p>Clean-reference OCR error: <strong>{result['clean_ocr_cer_percent']:.2f}%</strong>. The trained EDSR result is close to that value, so this sparse calibration set has limited evidence of further recoverable text errors. OCR errors do not establish human visual preference.</p>",
        "<table><tr><th>Method</th><th>CER ↓</th><th>Exact lines</th><th>Text ROI PSNR</th><th>Text ROI SSIM</th></tr>",
    ]
    for r in table:
        body.append(
            f"<tr><td>{r['name']}</td><td>{r['cer_percent']:.2f}%</td><td>{r['exact_lines']}/{r['lines']}</td><td>{r['text_roi_psnr_db']:.2f} dB</td><td>{r['text_roi_ssim']:.4f}</td></tr>"
        )
    body += [
        "</table><p>EDSR fine-tuning improves code and terminal OCR in these samples but regresses on mixed English/Chinese prose. The IMDN checkpoint has higher text error despite its historical PSNR gain. These regressions remain part of the result.</p>",
        '<img src="calibration-curves.png" alt="Readability and actual traffic curves">',
        "<h2>Real Screen Sharing qualification</h2><p>A separate 30-second native-canvas capture produced 60 raw tile snapshots and 45 state-aligned window crops. The transmitted SPS confirms 4:2:0. Its requested 720p output was not adopted: physical mirroring sent 3360×2100. Received video UDP payload averaged 694.72 Kbps before network headers. The maximum fixed 1-second incoming payload bin was 2187.78 Kbps. This is unshaped Mac-local traffic, not a 600 Kbps WAN result. Authentication material stayed on the Mac.</p>",
        "<p>Source-state matching does not prove that received content is current. A separate lightweight Mac loopback run showed 1.99–6.86 seconds of growing source-marker lag; the following queue diagnostic confirmed 1,872 packets pending after 30 seconds. This is a Mac sender plus Mac decoder result, not an RTX4090 receiver measurement. The cause was subsequently resolved below.</p>",
        '<h2>Resolved Mac decoder accumulation</h2><p>The decoder expanded the actual 4:2:0 source to 4:4:4 before copying pixels. Preserving native 4:2:0 reduced the final queue from 1,872 packets to 1 packet. Seven non-warmup changes in the production-code run took 26–43 ms from source draw to decoded markers; the separate prototype run took 46–60 ms. These are small-sample, unshaped M1 loopback measurements, excluding rendering and physical scanout. Format/layout and related decoder/display regression checks passed. The existing Mac project was not overwritten; testing used its independent experiment runtime.</p><img src="decoder-latency-curves.png" alt="Mac decoder before and after">',
        "<h2>Receiver responsiveness</h2><p>On the RTX 4090, the revised asynchronous display showed current native content in all 24 changes. First draw p95 was 2.22 ms, added delay p95 0.40 ms, and settled refinement p95 78.28 ms with the actual trained EDSR. The previous display retained old enhanced text on every first draw. These are warm local rendering/readback measurements, excluding network, hardware decoding and panel scanout.</p>",
        '<img src="display-latency-curves.png" alt="Actual RTX4090 display latency curves">',
        "<h2>Pixel-preserving comparison crops</h2><p>All panels use the same crop coordinates. No additional sharpening or enlargement was applied for presentation. Browser scaling may reduce them; open the PNG at 100% to inspect the original pixels.</p>",
    ]
    for picture in pictures:
        body.append(
            f'<p><a href="{picture}">Open original comparison</a></p><img src="{picture}" alt="Same-pixel comparison">'
        )
    body += (
        ["<h2>Limits and next gates</h2><ul>"]
        + [f"<li>{html.escape(v)}</li>" for v in result["limitations"]]
        + ["</ul><ol>"]
        + [f"<li>{html.escape(v)}</li>" for v in result["required_next"]]
        + [
            "</ol><p>Original datasets, frozen train/validation/test splits and previous experiments remain unchanged. These calibration samples are development-only.</p></html>"
        ]
    )
    (root / "REPORT.html").write_text("\n".join(body) + "\n")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
