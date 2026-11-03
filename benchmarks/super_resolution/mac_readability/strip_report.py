"""Compare source-aligned strips with frozen SR/tile controls and OCR floor."""

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
    sender = read(r / "detail-strips-v1/RESULTS.json")
    replay = read(r / "strip-replay-v1/RESULTS.json")
    protocol = read(r / "strip-quality-v1/PROTOCOL.json")
    frozen = read(r / "dense-clean-v1/PROTOCOL.json")
    assert protocol["complete"] and sender["complete"] and replay["complete"]
    for k in ["tesseract", "tessdata", "preprocessing"]:
        assert protocol[k] == frozen[k]
    freeze = read(r / "COMPARATOR-FROZEN.json")
    hard = freeze["hard_groups"]
    best = read(r / "training-v2-selected-full-quality/ROWS.json")
    official = [
        x
        for x in read(r / "dense-model-quality-v1/ROWS.json")
        if x["method"] == "edsr-official"
    ]
    candidate = read(r / "strip-quality-v1/ROWS.json")
    tile = read(r / "detail-quality-v1/ROWS.json")
    native = [
        x
        for x in read(r / "precodec-quality-v1/ROWS.json")
        if x["config"] == "1080p-420-no-codec"
    ]
    comparisons = [
        ("Official EDSR", official),
        ("Best trained EDSR", best),
        ("Native1080 420 source", native),
    ]
    comparisons += [
        (method, [x for x in tile if x["method"] == method])
        for method in sorted({x["method"] for x in tile})
    ]
    comparisons += [
        (method, [x for x in candidate if x["method"] == method])
        for method in sorted({x["method"] for x in candidate})
    ]
    base = {(x["state_id"], x["label_id"]): x for x in best if x["unit"] == "line"}
    bgroup = {
        x["group"]: x["cer"]
        for x in summarize([x for x in best if x["unit"] == "line"])[0]["groups"]
    }
    draws = np.random.default_rng(20261102).integers(0, len(hard), (10000, len(hard)))
    table = []
    for method, rows in comparisons:
        lines = [x for x in rows if x["unit"] == "line"]
        assert len(lines) == 1046
        summary = summarize(lines)[0]
        groups = {g["group"]: g["cer"] for g in summary["groups"]}
        cer = float(np.mean([groups[g] for g in hard]))
        cells = [x for x in rows if x["unit"] == "cell"]
        delta = np.array([bgroup[g] - groups[g] for g in hard])
        new = [
            x
            for x in lines
            if base[x["state_id"], x["label_id"]]["exact_nonspace"]
            and not x["exact_nonspace"]
        ]
        fixed = [
            x
            for x in lines
            if not base[x["state_id"], x["label_id"]]["exact_nonspace"]
            and x["exact_nonspace"]
        ]
        table.append(
            dict(
                method=method,
                hard_cer=cer,
                relative_gain_vs_official=1 - cer / freeze["hard_macro_cer"],
                full_summary=summary,
                cells=summarize(cells)[0] if cells else None,
                categories={
                    c: summarize([x for x in lines if x["category"] == c])[0]
                    for c in ["code", "table", "chinese", "english"]
                },
                newly_wrong_lines=new,
                repaired_lines=fixed,
                paired_gain_vs_best_ci95=np.quantile(
                    delta[draws].mean(1), [0.025, 0.975]
                ).tolist(),
            )
        )
    old = read(r / "detail-tiles-v1/RESULTS.json")
    by_state = {x["state_id"]: x for x in old["images"]}
    cost = dict(
        mac_encode_p50_p95_ms=np.percentile(
            [x["encode_ms"] for x in sender["images"]], [50, 95]
        ).tolist(),
        aligned_bytes_p50_p95=np.percentile(
            [x["wire_bytes"] for x in sender["images"]], [50, 95]
        ).tolist(),
        fixed_bytes_p50_p95=np.percentile(
            [x["controls"]["fixed_strips_wire_bytes"] for x in sender["images"]],
            [50, 95],
        ).tolist(),
        monolithic_bytes_p50_p95=np.percentile(
            [x["controls"]["monolithic_wire_bytes"] for x in sender["images"]], [50, 95]
        ).tolist(),
        mean_paired_aligned_byte_change_vs_tiles=float(
            np.mean(
                [
                    x["wire_bytes"] / by_state[x["state_id"]]["wire_bytes"] - 1
                    for x in sender["images"]
                ]
            )
        ),
    )
    states = [s for v in replay["link_replays"] for s in v["states"]]
    done = [s["complete_age_s"] for s in states if s["complete_age_s"] is not None]
    network = dict(
        complete_before_next_change=len(done),
        total=32,
        conditional_completion_p50_p95_ms=(
            1000 * np.percentile(done, [50, 95])
        ).tolist(),
        scope="Modeled shared600Kbps, actualMacpacket sizes, basepriority, noinitialcredit;100msstability+Macencode; not physicalnetwork",
    )
    result = dict(
        complete=True,
        script_sha256=sha(Path(__file__)),
        table=table,
        cost=cost,
        network=network,
        quality_target_met=any(
            x["hard_cer"] <= freeze["target_hard_cer"]
            for x in table
            if x["method"].startswith("source-strips")
        ),
        production_selected=False,
        test_used=False,
        local_latency_verified=False,
        end_to_end_verified=False,
        boundary_audit_sha256=sha(r / "DETAIL-BOUNDARY-AUDIT.json"),
        limitations=[
            "Development only; repeated comparisons, not independentconfirmation",
            "OCR errors retained even on native source",
            "Byte-budget composite quality does not establish quality at a wallclockdeadline",
            "Newlywrong/repaired counts do not measure humanpreference",
        ],
        evidence={
            f: sha(r / f)
            for f in [
                "detail-strips-v1/RESULTS.json",
                "strip-replay-v1/RESULTS.json",
                "strip-quality-v1/ROWS.json",
            ]
        },
    )
    save(r / "STRIP-RESULTS.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    labels = [
        "Official",
        "Best SR",
        "Native420",
        "Tiles15KB",
        "Tiles60KB",
        "Strips15KB",
        "Strips60KB",
    ]
    axes[0].bar(labels, [x["hard_cer"] * 100 for x in table])
    axes[0].axhline(freeze["target_hard_cer"] * 100, color="green", ls="--")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_ylabel("Hard CER (%)")
    for c in ["chinese", "code", "table"]:
        axes[1].plot(
            labels,
            [x["categories"][c]["macro_cer"] * 100 for x in table],
            marker="o",
            label=c,
        )
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].set_ylabel("Category CER (%)")
    axes[1].legend()
    fig.savefig(r / "strip-curves.png", dpi=150)
    plt.close(fig)
    columns = "".join(
        f"<tr><td>{html.escape(x['method'])}</td><td>{100 * x['hard_cer']:.4f}%</td><td>{100 * x['cells']['macro_cer']:.4f}%</td><td>{len(x['newly_wrong_lines'])}/{len(x['repaired_lines'])}</td></tr>"
        if x["cells"]
        else f"<tr><td>{html.escape(x['method'])}</td><td>{100 * x['hard_cer']:.4f}%</td><td>Not scored in this control</td><td>{len(x['newly_wrong_lines'])}/{len(x['repaired_lines'])}</td></tr>"
        for x in table
    )
    source = (
        a.project
        / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/reference"
    )
    states = read(source / "manifest.json")["states"]
    index = read(r / "strip-replay-v1/IMAGES.json")
    bi = read(r / "TRAINING-V2-SELECTED-IMAGES.json")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    for sid in [1, 9, 25]:
        state = next(x for x in states if x["state_id"] == sid)
        paths = [
            source / state["reference"]["path"],
            Path(next(x for x in bi["images"] if x["state_id"] == sid)["path"]),
        ]
        paths += [
            Path(
                next(
                    x
                    for x in index["images"]
                    if x["state_id"] == sid and x["method"] == method
                )["path"]
            )
            for method in ["source-strips-15000bytes", "source-strips-60000bytes"]
        ]
        canvas = Image.new("RGB", (1144, 304), "#edf0f3")
        draw = ImageDraw.Draw(canvas)
        for i, (name, path) in enumerate(
            zip(
                [
                    "Source",
                    "Best trained SR",
                    "Source-aligned strips15KB",
                    "Source-aligned strips60KB",
                ],
                paths,
            )
        ):
            x = i % 2 * 572
            y = i // 2 * 152
            draw.text((x + 4, y + 3), name, font=font, fill="black")
            canvas.paste(
                Image.open(path).convert("RGB").crop((36, 78, 600, 200)),
                (x + 4, y + 26),
            )
        canvas.save(r / f"strip-{state['category']}.png")
    doc = f"""<!doctype html><meta charset="utf-8"><title>Source-aligned strip experiment</title><style>body{{font:17px system-ui;max-width:1100px;margin:40px auto;line-height:1.5}}td,th{{padding:8px;border:1px solid #ddd}}img{{max-width:100%}}</style>
<h1>Source-aligned420 strips: quality and bytecost</h1><p>32frozenMacdevelopmentframes. Whole-width bands choose low-contrast horizontal boundaries from sourcepixels, without OCR or category routing. Originaltile experiment retained; finaltest untouched. Quality images are byte-budget composites on frozensteadySR, not measurements at a wallclockdeadline.</p>
<table><tr><th>Method</th><th>Hard CER</th><th>Cell CER</th><th>Newwrong/repaired vsbestSR</th></tr>{columns}</table>
<p>Boundary audit: in the60KBtile control,40/51newlywrong lines already have pixels identical to native420source; all51are also misrecognized on the native420source. Only3mixed lines have moreerrors than both baselines. At15KB,all24newlywrong lines crosspatchregions. Thus notallregression is a boundaryproblem; sourceOCR failures remain included.</p>
<p>Macencodep50/p95={cost["mac_encode_p50_p95_ms"][0]:.1f}/{cost["mac_encode_p50_p95_ms"][1]:.1f}ms. Meanpairedbytes change vs256pxtiles={cost["mean_paired_aligned_byte_change_vs_tiles"] * 100:+.2f}%. Monolithicwholeframe median={cost["monolithic_bytes_p50_p95"][0] / 1000:.1f}KB: shared compression alone doesnot produce a15KBfullrefresh.</p>
<p>Shared600Kbpssimulation:{len(done)}/32complete before3ssourcechange; conditionalp50/p95={network["conditional_completion_p50_p95_ms"][0]:.0f}/{network["conditional_completion_p50_p95_ms"][1]:.0f}ms. No physicalWAN/4090displaytiming orhumanpreference claim.</p>
<img src="strip-curves.png"><img src="strip-code.png"><img src="strip-table.png"><img src="strip-chinese.png">"""
    (r / "STRIP-REPORT.html").write_text(doc)
    for x in table:
        print(
            x["method"],
            round(x["hard_cer"] * 100, 4),
            "cells",
            round(x["cells"]["macro_cer"] * 100, 4) if x["cells"] else None,
            "new",
            len(x["newly_wrong_lines"]),
            "fixed",
            len(x["repaired_lines"]),
        )
    print(cost)
    print(network)


if __name__ == "__main__":
    main()
