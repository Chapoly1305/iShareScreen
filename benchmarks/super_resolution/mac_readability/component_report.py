"""Report exact-pattern compression, real-device processing and remaining gaps."""

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.mac_readability.quality import save, sha, summarize


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    a = p.parse_args()
    r = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    sender = read(r / "component-mac-v1/RESULTS.json")
    cost = read(r / "COMPONENT-COSTS.json")
    display = read(r / "component-display-v2/RESULTS.json")
    quality = read(r / "component-quality-v1/PROTOCOL.json")
    frozen = read(r / "dense-clean-v1/PROTOCOL.json")
    assert all(v["complete"] for v in [sender, cost, display, quality])
    assert display["sender_sha256"] == sha(r / "component-mac-v1/RESULTS.json")
    for k in ["tesseract", "tessdata", "preprocessing"]:
        assert quality[k] == frozen[k]
    assert len(display["records"]) == 32 and all(
        v["first_base_exact"] and v["stale_rejected"] and v["render_max_delta"] <= 1
        for v in display["records"]
    )
    assert sum(x["control_first"] for x in display["records"]) == 16
    candidate = read(r / "component-quality-v1/ROWS.json")
    prior = [
        x
        for x in read(r / "precodec-quality-v1/ROWS.json")
        if x["config"] == "1080p-420-no-codec"
    ]
    lookup = {(x["state_id"], x["label_id"]): x for x in prior}
    for x in candidate:
        if x["unit"] == "line":
            assert (
                x["text_predicted"]
                == lookup[x["state_id"], x["label_id"]]["text_predicted"]
            )
    best = read(r / "training-v2-selected-full-quality/ROWS.json")
    official = [
        x
        for x in read(r / "dense-model-quality-v1/ROWS.json")
        if x["method"] == "edsr-official"
    ]
    freeze = read(r / "COMPARATOR-FROZEN.json")
    base = {(x["state_id"], x["label_id"]): x for x in best if x["unit"] == "line"}
    table = []
    for method, rows in [
        ("Official EDSR", official),
        ("Best trained EDSR", best),
        ("Exact source-pattern supplement", candidate),
    ]:
        lines = [x for x in rows if x["unit"] == "line"]
        cells = [x for x in rows if x["unit"] == "cell"]
        assert len(lines) == 1046 and len(cells) == 1806
        summary = summarize(lines)[0]
        groups = {g["group"]: g["cer"] for g in summary["groups"]}
        cer = float(np.mean([groups[g] for g in freeze["hard_groups"]]))
        table.append(
            dict(
                method=method,
                hard_cer=cer,
                relative_gain_vs_official=1 - cer / freeze["hard_macro_cer"],
                lines=summary,
                cells=summarize(cells)[0],
                categories={
                    c: summarize([x for x in lines if x["category"] == c])[0]
                    for c in ["code", "table", "chinese", "english"]
                },
                newly_wrong_lines=[
                    x
                    for x in lines
                    if base[x["state_id"], x["label_id"]]["exact_nonspace"]
                    and not x["exact_nonspace"]
                ],
                repaired_lines=[
                    x
                    for x in lines
                    if not base[x["state_id"], x["label_id"]]["exact_nonspace"]
                    and x["exact_nonspace"]
                ],
            )
        )
    result = dict(
        complete=True,
        test_used=False,
        production_selected=False,
        script_sha256=sha(Path(__file__)),
        quality_table=table,
        quality_target_met=table[-1]["hard_cer"] <= freeze["target_hard_cer"],
        full_goal_met=False,
        measured_mac_sender=dict(
            encode_p50_p95_ms=cost["mac_encode_p50_p95_ms"],
            decode_p50_p95_ms=cost["mac_decode_p50_p95_ms"],
        ),
        modeled_traffic=dict(
            wire_bytes_min_p50_p95_max=cost["wire_bytes_min_p50_p95_max"],
            aggregate_byte_reduction=cost["aggregate_byte_reduction"],
            completed=cost["modeled_complete_before_next_source"],
            total=32,
            conditional_completion_p50_p95_ms=cost[
                "modeled_complete_conditional_p50_p95_ms"
            ],
        ),
        measured_receiver_processing=display["stats"],
        receiver_scope=display["scope"],
        all32_exact_source_roundtrip=cost["all32_raw_bytes_independently_verified"],
        evidence={
            f: sha(r / f)
            for f in [
                "component-mac-v1/RESULTS.json",
                "COMPONENT-COSTS.json",
                "component-display-v1/RESULTS.json",
                "component-display-v2/RESULTS.json",
                "component-quality-v1/ROWS.json",
                "DETAIL-BOUNDARY-AUDIT.json",
                "STRIP-RESULTS.json",
            ]
        },
        remaining=[
            "Integrated live Mac-to4090 base+supplement transport and constrained-link measurement",
            "Quality target and independent confirmation",
            "Full asynchronous SR+supplement integration and local latency guard",
            "Human visual comparison and temporal stability",
            "Final held-out test after development selection",
        ],
    )
    save(r / "COMPONENT-RESULTS.json", result)
    old = read(r / "detail-tiles-v1/RESULTS.json")
    old_by = {x["state_id"]: x for x in old["images"]}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    for category in ["code", "table", "chinese", "english"]:
        items = [x for x in sender["images"] if x["category"] == category]
        axes[0].scatter(
            [old_by[x["state_id"]]["wire_bytes"] / 1000 for x in items],
            [x["wire_bytes"] / 1000 for x in items],
            label=category,
        )
    axes[0].plot([0, 110], [0, 110], "k--")
    axes[0].set(
        xlabel="256px tiles (KB)",
        ylabel="Exact-pattern codec (KB)",
        title="Estimated wire bytes, paired frames",
    )
    axes[0].legend(fontsize=8)
    axes[1].bar(
        ["M1 encode", "4090-host processing"],
        [
            cost["mac_encode_p50_p95_ms"][1],
            display["stats"]["supplement_available_to_readback_ms"]["p95"],
        ],
    )
    axes[1].set(ylabel="Measured p95 (ms)", title="Processing only; transport excluded")
    axes[2].bar(
        ["Official", "Best SR", "Exact source"], [x["hard_cer"] * 100 for x in table]
    )
    axes[2].axhline(freeze["target_hard_cer"] * 100, color="green", ls="--")
    axes[2].set(ylabel="Hard-group CER (%)", title="Frozen OCR retains source errors")
    fig.savefig(r / "component-curves.png", dpi=150)
    plt.close(fig)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    bi = read(r / "TRAINING-V2-SELECTED-IMAGES.json")
    ni = read(r / "precodec-images-v1/IMAGES.json")
    for sid, category in [(1, "code"), (9, "table"), (25, "chinese")]:
        paths = [
            Path(next(x for x in bi["images"] if x["state_id"] == sid)["path"]),
            Path(
                next(
                    x
                    for x in ni["images"]
                    if x["state_id"] == sid and x["config"] == "1080p-420-no-codec"
                )["path"]
            ),
        ]
        canvas = Image.new("RGB", (1144, 152), "#edf0f3")
        draw = ImageDraw.Draw(canvas)
        for i, (name, path) in enumerate(
            zip(["Best trained SR", "Exact received source (4:2:0)"], paths)
        ):
            x = i * 572
            draw.text((x + 4, 3), name, font=font, fill="black")
            canvas.paste(
                Image.open(path).convert("RGB").crop((36, 78, 600, 200)), (x + 4, 26)
            )
        canvas.save(r / f"component-{category}.png")
    rows = "".join(
        f"<tr><td>{html.escape(x['method'])}</td><td>{x['hard_cer'] * 100:.4f}%</td><td>{x['cells']['macro_cer'] * 100:.4f}%</td><td>{x['cells']['exact_units']}/1806</td><td>{len(x['newly_wrong_lines'])}/{len(x['repaired_lines'])}</td></tr>"
        for x in table
    )
    stats = display["stats"]
    network = cost["modeled_complete_conditional_p50_p95_ms"]
    document = f"""<!doctype html><meta charset="utf-8"><title>Exact source-pattern refinement</title>
<style>body{{font:17px system-ui;max-width:1100px;margin:40px auto;line-height:1.55;color:#182334}}td,th{{padding:9px;border:1px solid #ddd}}table{{border-collapse:collapse}}img{{max-width:100%}}.limit{{background:#fff4d8;padding:16px}}</style>
<h1>Exact source-pattern refinement: less data, fast local display</h1>
<p>The experimental codec reuses a foreground pattern only when its dimensions and every luma byte are identical. It performs no OCR, approximate glyph matching, or character substitution. Color remains lossless 4:2:0. All 32 frozen development frames were reconstructed byte-for-byte on the Mac and independently verified on H100; the RTX 4090 display readback differed by at most one RGB level.</p>
<h2>Measured processing and encoded size</h2>
<p>Estimated wire size fell by <b>{cost["aggregate_byte_reduction"] * 100:.1f}% in aggregate</b> versus independent 256px tiles. The per-frame range is 20.0–49.3 KB, with a 30.4 KB median. This includes the declared fragment overhead model; it is not a physical-network byte measurement. No cross-frame cache is used.</p>
<p>Actual M1 encoding p50/p95: <b>{cost["mac_encode_p50_p95_ms"][0]:.1f}/{cost["mac_encode_p50_p95_ms"][1]:.1f} ms</b>. On the RTX 4090 host, an available supplement reached synchronized GPU readback in <b>{stats["supplement_available_to_readback_ms"]["p50"]:.1f}/{stats["supplement_available_to_readback_ms"]["p95"]:.1f} ms</b>. The added first-display p95 was {stats["added_first_ms"]["p95"]:.2f} ms in a counterbalanced comparison. All 32 first frames matched the current base image, and old-generation supplements were rejected.</p>
<p class="limit">The local display test uses precomputed SR base images. It measures supplement decoding, RGB conversion, upload, drawing and GPU readback; it excludes base inference, source capture, network transfer and panel scanout. It does not by itself satisfy the complete end-to-end goal.</p>
<h2>Shared-bandwidth model</h2>
<p>With recorded Mac base-video packet sizes and supplemental data sharing one modeled 600 Kbps serializer, all 32 supplements completed before the next 3-second source change. Completion p50/p95 was {network[0]:.0f}/{network[1]:.0f} ms, versus 1754/2630 ms for the tile prototype, where 4 of 32 did not finish. The model includes a 100 ms stability wait and measured Mac encoding; it excludes loss, control traffic, SSH/VPN overhead, decoder scheduling and display. Real Mac-to-4090 media transport remains to be measured.</p>
<h2>Quality remains a separate gate</h2>
<table><tr><th>Method</th><th>Hard CER</th><th>Cell CER</th><th>Exact cells</th><th>New wrong / repaired lines vs best SR</th></tr>{rows}</table>
<p>Lossless source transmission restores the actual source pixels, but the frozen OCR still makes errors on those pixels. It does not reach the hard-CER target of {freeze["target_hard_cer"] * 100:.4f}%. All errors are retained. No human-preference or overall success claim is made, and the final test set remains untouched.</p>
<p>The boundary audit explains why fixing seams alone was insufficient: 40 of the 51 newly wrong lines in the earlier 60 KB tile experiment were already pixel-identical to the native 4:2:0 source. Source-aligned strips helped the 15 KB case slightly, but did not reduce total bytes or reach the quality target.</p>
<img src="component-curves.png"><h2>Fixed native-size crops</h2><img src="component-code.png"><img src="component-table.png"><img src="component-chinese.png">
<p><a href="STRIP-REPORT.html">Strip control details</a> · <a href="COMPONENT-RESULTS.json">Audited results and evidence hashes</a></p>"""
    (r / "COMPONENT-REPORT.html").write_text(document)
    for x in table:
        print(
            x["method"],
            "hardCER",
            round(x["hard_cer"] * 100, 4),
            "cellCER",
            round(x["cells"]["macro_cer"] * 100, 4),
            "exactcells",
            x["cells"]["exact_units"],
        )
    print("Full goal remains incomplete; no production selection.")


if __name__ == "__main__":
    main()
