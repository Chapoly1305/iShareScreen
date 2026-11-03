"""Audit source-detail feasibility without confusing byte budgets and latency."""

import argparse
import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.mac_readability.detail_replay import shared_link
from benchmarks.super_resolution.mac_readability.quality import save, sha, summarize


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    a = p.parse_args()
    r = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    sender = read(r / "detail-tiles-v1/RESULTS.json")
    replay = read(r / "detail-replay-v1/RESULTS.json")
    quality = read(r / "detail-quality-v1/PROTOCOL.json")
    assert quality["complete"] and sender["complete"] and replay["complete"]
    freeze = read(r / "COMPARATOR-FROZEN.json")
    hard = freeze["hard_groups"]
    baseline = [
        x
        for x in read(r / "training-v2-selected-full-quality/ROWS.json")
        if x["unit"] == "line"
    ]
    by_key = {(x["state_id"], x["label_id"]): x for x in baseline}
    candidate = read(r / "detail-quality-v1/ROWS.json")
    table = []
    methods = [("Best trained EDSR", baseline)]
    methods += [
        (method, [x for x in candidate if x["method"] == method])
        for method in sorted({x["method"] for x in candidate})
    ]
    base_groups = {x["group"]: x["cer"] for x in summarize(baseline)[0]["groups"]}
    draws = np.random.default_rng(20261102).integers(0, len(hard), (10000, len(hard)))
    for method, rows in methods:
        assert len(rows) == 1046
        summary = summarize(rows)[0]
        groups = {g["group"]: g["cer"] for g in summary["groups"]}
        cer = float(np.mean([groups[g] for g in hard]))
        delta = np.array([base_groups[g] - groups[g] for g in hard])
        table.append(
            dict(
                method=method,
                hard_cer=cer,
                relative_gain_vs_official=1 - cer / freeze["hard_macro_cer"],
                relative_gain_vs_best=1 - cer / np.mean([base_groups[g] for g in hard]),
                paired_gain_vs_best_ci95=np.quantile(
                    delta[draws].mean(1), [0.025, 0.975]
                ).tolist(),
                summary=summary,
                categories={
                    c: summarize([x for x in rows if x["category"] == c])[0]
                    for c in ["code", "table", "chinese", "english"]
                },
                newly_wrong_lines=[
                    x
                    for x in rows
                    if by_key[x["state_id"], x["label_id"]]["exact_nonspace"]
                    and not x["exact_nonspace"]
                ],
                repaired_lines=[
                    x
                    for x in rows
                    if not by_key[x["state_id"], x["label_id"]]["exact_nonspace"]
                    and x["exact_nonspace"]
                ],
                zero_error_group_edits=sum(
                    x["edits"]
                    for x in rows
                    if x["group"] in freeze["zero_error_groups"]
                ),
            )
        )
    codec = read(
        a.project
        / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/codec-v1/MANIFEST.json"
    )
    added = []
    base_delays = []
    for run in replay["link_replays"]:
        source = next(
            x
            for x in codec["runs"]
            if x["group"] == run["group"] and x["height"] == 720
        )
        alone = shared_link(source, [])
        baseline_times = {
            e["frame"]: e for e in alone["events"] if e["kind"] == "base" and e["last"]
        }
        for event in run["events"]:
            if (
                event["kind"] == "base"
                and event["last"]
                and event["frame"] in baseline_times
            ):
                ref = baseline_times[event["frame"]]
                added.append((event["finish_s"] - ref["finish_s"]) * 1000)
                base_delays.append((ref["finish_s"] - ref["ready"]) * 1000)
    assert min(added) > -1e-7 and max(added) <= 1244 * 8 / 600 + 1e-6
    states = [s for v in replay["link_replays"] for s in v["states"]]
    finish = [s["complete_age_s"] for s in states if s["complete_age_s"] is not None]
    bounds = [v["serialization_floor_full600_ms"] for v in sender["images"]]
    network = dict(
        states=len(states),
        complete_before_next_source=len(finish),
        missing_complete=len(states) - len(finish),
        conditional_complete_p50_p95_ms=(
            1000 * np.percentile(finish, [50, 95])
        ).tolist(),
        mean_refined_area_at_200ms=float(
            np.mean([s["snapshots"][0]["pixels"] / 1920 / 1080 for s in states])
        ),
        base_only_serialization_delay_p50_p95_ms=np.percentile(
            base_delays, [50, 95]
        ).tolist(),
        detail_added_base_delay_p50_p95_max_ms=np.percentile(
            added, [50, 95, 100]
        ).tolist(),
        model="Shared600Kbps, base priority, max1244wire-byte nonpreemptive fragments; 100msstability + measured Mac encode; no starting token credit",
        caveats=replay["limits"]
        + [
            "No packet losses, no bounded queue, media DTS assumed available immediately; neither actual Screen Sharing nor physical WAN"
        ],
    )
    result = dict(
        complete=True,
        test_used=False,
        production_selected=False,
        quality_table=table,
        mac_encode_p50_p95_ms=np.percentile(
            [v["encode_ms"] for v in sender["images"]], [50, 95]
        ).tolist(),
        full_wire_bytes_min_p50_p95_max=np.percentile(
            [v["wire_bytes"] for v in sender["images"]], [0, 50, 95, 100]
        ).tolist(),
        full600_serialization_floor_p50_p95_ms=np.percentile(bounds, [50, 95]).tolist(),
        network=network,
        quality_target_met=any(
            x["hard_cer"] <= freeze["target_hard_cer"] for x in table[1:]
        ),
        local_latency_verified=False,
        end_to_end_verified=False,
        evidence={
            f: sha(r / f)
            for f in [
                "detail-tiles-v1/RESULTS.json",
                "detail-replay-v1/RESULTS.json",
                "detail-quality-v1/ROWS.json",
            ]
        },
        script_sha256=sha(Path(__file__)),
        conclusion="Full-page lossless refinement cannot fit200ms at600Kbps even with allbandwidth; partialquality andsenderCPU costs measured; trueclientandlink pending",
    )
    save(r / "DETAIL-RESULTS.json", result)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
    axes[0].bar(["Best SR", "+15 KB", "+60 KB"], [x["hard_cer"] * 100 for x in table])
    axes[0].axhline(freeze["target_hard_cer"] * 100, color="green", ls="--")
    axes[0].set_ylabel("Hard-group CER (%)")
    axes[0].set_title("Byte-budget quality, not deadline quality")
    axes[1].hist(bounds, bins=10)
    axes[1].axvline(200, color="red", ls="--")
    axes[1].set_xlabel("Full-detail serialization lower bound (ms)")
    axes[1].set_title("All 600 Kbps to detail; base excluded")
    axes[2].hist(np.array(finish) * 1000, bins=10)
    axes[2].set_xlabel("Modeled full-detail age (ms)")
    axes[2].set_title(f"Shared link: {len(finish)}/32 finish before change")
    fig.savefig(r / "detail-curves.png", dpi=150)
    plt.close(fig)
    index = read(r / "detail-replay-v1/IMAGES.json")
    baseindex = read(r / "TRAINING-V2-SELECTED-IMAGES.json")
    source_dir = (
        a.project
        / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/reference"
    )
    source = read(source_dir / "manifest.json")
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    for sid in [1, 9, 25]:
        state = next(x for x in source["states"] if x["state_id"] == sid)
        paths = [
            source_dir / state["reference"]["path"],
            Path(next(x for x in baseindex["images"] if x["state_id"] == sid)["path"]),
        ]
        paths += [
            Path(
                next(
                    x
                    for x in index["images"]
                    if x["state_id"] == sid and x["method"] == m
                )["path"]
            )
            for m in ["source-detail-15000bytes", "source-detail-60000bytes"]
        ]
        canvas = Image.new("RGB", (1144, 304), "#edf0f3")
        draw = ImageDraw.Draw(canvas)
        for i, (label, path) in enumerate(
            zip(
                ["Source", "Best trained SR", "SR + 15 KB detail", "SR + 60 KB detail"],
                paths,
            )
        ):
            x = i % 2 * 572
            y = i // 2 * 152
            draw.text((x + 4, y + 3), label, font=font, fill="black")
            canvas.paste(
                Image.open(path).convert("RGB").crop((36, 78, 600, 200)),
                (x + 4, y + 26),
            )
        canvas.save(r / f"detail-{state['category']}.png")
    cells = "".join(
        f"<tr><td>{html.escape(x['method'])}</td><td>{100 * x['hard_cer']:.4f}%</td><td>{len(x['newly_wrong_lines'])}</td><td>{len(x['repaired_lines'])}</td></tr>"
        for x in table
    )
    doc = f"""<!doctype html><meta charset="utf-8"><title>Source-assisted 420 detail feasibility</title><style>body{{font:17px system-ui;max-width:1100px;margin:40px auto;line-height:1.5}}td,th{{padding:8px;border:1px solid #ddd}}img{{max-width:100%}}</style>
<h1>Source-assisted 4:2:0 detail: first feasibility experiment</h1>
<p>32 frozen Mac development frames; source-only tile priority; unchanged OCR; final test untouched. Lossless tile codec, exact 420 roundtrip and stale-revision rejection verified. This is an offline prototype, not a deployed Screen Sharing feature.</p>
<table><tr><th>Method</th><th>Hard CER</th><th>Newly wrong lines vs best SR</th><th>Repaired lines vs best SR</th></tr>{cells}</table>
<p>15 KB and60 KB are total detail-wire budgets. The quality images use the frozen steady SR image underneath. They do not demonstrate a200ms or800ms wall-clock result. Base video needs additional bytes.</p>
<p>Actual M1 encoding p50/p95: {result["mac_encode_p50_p95_ms"][0]:.1f}/{result["mac_encode_p50_p95_ms"][1]:.1f}ms. Full refinement requires {min(v["wire_bytes"] for v in sender["images"]) / 1000:.1f}–{max(v["wire_bytes"] for v in sender["images"]) / 1000:.1f}KB. Even giving detail all600Kbps, serialization p50/p95 is {result["full600_serialization_floor_p50_p95_ms"][0]:.0f}/{result["full600_serialization_floor_p50_p95_ms"][1]:.0f}ms, excluding stability/encoding/network overhead beyond the declared model.</p>
<p>Shared-link simulation uses actual Mac video packet sizes plus the detail packets. {len(finish)}/32 pages finish before the3s source change; conditional completion p50/p95={network["conditional_complete_p50_p95_ms"][0]:.0f}/{network["conditional_complete_p50_p95_ms"][1]:.0f}ms. Missing4 pages must not be hidden. Mean refined area at200ms={network["mean_refined_area_at_200ms"] * 100:.2f}%. Added base serialization delay p95={network["detail_added_base_delay_p50_p95_max_ms"][1]:.2f}ms; base-only p95={network["base_only_serialization_delay_p50_p95_ms"][1]:.2f}ms.</p>
<p>Simulation excludes codec setup/control, ACKs, loss, VPN, decode and display. It uses unbounded queues and immediate packet availability at DTS; no physical WAN, GPU display or input-to-photon claim. Full lossless refresh does not meet200ms; sparse changes/cache or a different reconstruction scheme require separate verification.</p>
<img src="detail-curves.png"><h2>Fixed native-size examples</h2><img src="detail-code.png"><img src="detail-table.png"><img src="detail-chinese.png">"""
    (r / "DETAIL-REPORT.html").write_text(doc)
    for x in table:
        print(
            x["method"],
            round(x["hard_cer"] * 100, 4),
            "new",
            len(x["newly_wrong_lines"]),
            "repaired",
            len(x["repaired_lines"]),
        )
    print("network", network)


if __name__ == "__main__":
    main()
