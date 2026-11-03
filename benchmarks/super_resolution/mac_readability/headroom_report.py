"""Report repeated actual Mac captures without hiding late/missing changes."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from benchmarks.super_resolution.mac_readability.analyze_link import analyze
from benchmarks.super_resolution.mac_readability.quality import save, sha


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    runs = []
    for path in sorted(root.glob("capture-headroom-*")):
        row = analyze(path)
        assert row["state"] == "completed" and row["actual_canvas"] == [1280, 720]
        status = read(path / "STATUS.json")
        stimulus = read(path / "stimulus/manifest.json")
        link = read(path / "LINK.json")
        assert link["config"]["rate_bps_per_direction"] == 600000
        rate = int(path.name.rsplit("-", 1)[1][:-1])
        row["requested_kbps"] = rate
        start = stimulus["states"][0]["draw_start_ns"] / 1e9
        end = stimulus["finished_ns"] / 1e9
        incoming = link["directions"]["incoming"]["records"]
        active = [r for r in incoming if start <= r["arrival"] < end]
        row["active_stimulus_link_status_counts"] = dict(
            Counter(r["status"] for r in active)
        )
        row["before_stimulus_link_status_counts"] = dict(
            Counter(r["status"] for r in incoming if r["arrival"] < start)
        )
        row["active_offered_payload_kbps"] = (
            sum(r["payload_bytes"] for r in active) * 8 / (end - start) / 1000
        )
        row["last_observed_ns"] = max(r["time_ns"] for r in status["samples"])
        assert row["last_observed_ns"] >= stimulus["finished_ns"], (
            "Right-censored experiment before stimulus finished"
        )
        row["complete_stimulus_observation"] = True
        for response in row["responses"]:
            if response["warmup"]:
                continue
            state = next(
                s for s in stimulus["states"] if s["state_id"] == response["state_id"]
            )
            t = state["draw_start_ns"] / 1e9
            response["tail_drops_first_second"] = sum(
                r["status"] == "tail_drop"
                for r in incoming
                if t <= r["arrival"] < t + 1
            )
        runs.append(row)
    assert len(runs) == 6
    groups = []
    for rate in [600, 450, 300]:
        selected = [r for r in runs if r["requested_kbps"] == rate]
        assert len(selected) == 2
        responses = [x for r in selected for x in r["responses"] if not x["warmup"]]
        ok = [
            x["first_both_markers_ms"]
            for x in responses
            if x["observed_before_next_source"]
        ]
        groups.append(
            dict(
                requested_kbps=rate,
                runs=2,
                changes=len(responses),
                successful=len(ok),
                missing_or_late=len(responses) - len(ok),
                conditional_median_ms=float(np.median(ok)),
                conditional_p95_ms=float(np.percentile(ok, 95)),
                per_run_p95_ms=[
                    r["warm_response_summary"]["conditional_ms"]["p95"]
                    for r in selected
                ],
                active_tail_drops=sum(
                    r["active_stimulus_link_status_counts"].get("tail_drop", 0)
                    for r in selected
                ),
                before_stimulus_tail_drops=sum(
                    r["before_stimulus_link_status_counts"].get("tail_drop", 0)
                    for r in selected
                ),
                active_delivery_errors=sum(
                    r["active_stimulus_link_status_counts"].get("delivery_error", 0)
                    for r in selected
                ),
            )
        )
    result = dict(
        schema="mac_rate_headroom_repeats_v1",
        complete=True,
        script_sha256=sha(Path(__file__)),
        runs=runs,
        groups=groups,
        conclusion="Lower requested rate reduces active-window tail drops but does not establish consistently fast updates. Do not adopt a lower offer from median latency alone.",
        scope="ActualMac ScreenSharing virtual1280x720 sender andMac native420 decoder;600Kbps per-direction process-local link,250ms queue,10ms one-way; not physicalWAN orRTX4090 receiver.",
        limitations=[
            "Two runs percondition andseven measured changes perrun; repeated calibration content, not independent final-test samples.",
            "Medians/p95 are conditional on matching both markers before the nextsource; missing updates remain explicitly counted.",
            "Marker agreement is not whole-frame atomicity or text clarity; quality tradeoff is not measured by this timing table.",
            "Delivery errors during socket shutdown are separated from active stimulus; allraw records are preserved.",
            "This comparison changes requested encoder budget; it does not change the600Kbps link limit.",
        ],
    )
    save(root / "HEADROOM-REPEATS.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    colors = {600: "#446aab", 450: "#247c87", 300: "#a95340"}
    for rate in [600, 450, 300]:
        selected = [r for r in runs if r["requested_kbps"] == rate]
        for repeat, run in enumerate(selected):
            values = [x for x in run["responses"] if not x["warmup"]]
            for i, x in enumerate(values):
                ok = x["observed_before_next_source"]
                axes[0].scatter(
                    rate + (repeat - 0.5) * 24 + (i - 3) * 2,
                    x["first_both_markers_ms"] if ok else x["visible_ms"],
                    color=colors[rate],
                    marker="o" if ok else "x",
                    s=45,
                )
    axes[0].set(
        xlabel="Requested video rate (Kbps); link always600",
        ylabel="Source change to decoded marker match (ms)",
        title="All42 changes; crosses mean no update before nextsource",
    )
    axes[1].bar(
        [str(g["requested_kbps"]) for g in groups],
        [g["active_tail_drops"] for g in groups],
        color=[colors[g["requested_kbps"]] for g in groups],
    )
    axes[1].set(
        xlabel="Requested video rate (Kbps)",
        ylabel="Tail-dropped packets during visible stimuli",
        title="Startup and teardown excluded",
    )
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.savefig(root / "headroom-curves.png", dpi=170)
    fig.savefig(root / "headroom-curves.svg")
    plt.close(fig)
    body = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Encoder headroom repeats</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:40px auto;padding:20px}img{max-width:100%}td,th{padding:10px;border-bottom:1px solid #ddd}</style><h1>Encoder headroom under a fixed600 Kbps link</h1><p>Six actual Mac Screen Sharing captures completed in order600,450,300,300,450,600 Kbps. The link remained600 Kbps per direction, with250 ms queue capacity and10 ms propagation delay. All runs used a1280×720 virtual desktop and native4:2:0 decoding.</p><table><tr><th>Encoder request</th><th>Updates before next change</th><th>Conditional median</th><th>Conditional p95</th><th>Active tail drops</th></tr>'
    ]
    for g in groups:
        body.append(
            f"<tr><td>{g['requested_kbps']} Kbps</td><td>{g['successful']}/{g['changes']}</td><td>{g['conditional_median_ms']:.1f} ms</td><td>{g['conditional_p95_ms']:.1f} ms</td><td>{g['active_tail_drops']}</td></tr>"
        )
    body += [
        '</table><p><strong>Reducing the offer did not solve long delays.</strong> The300 Kbps condition still contains updates over1.6 seconds. Other conditions include changes not observed before the next source, about3 seconds later. Conditional p95 must therefore not be interpreted as the latency of all updates.</p><img src="headroom-curves.png" alt="All response observations and active-window drops"><p>During the visible stimuli, delivery errors were absent; teardown errors are retained separately in the raw logs. Tail drops fell with lower offers, but the small repeated experiment does not prove a causal explanation for each slow response.</p><p>These are Mac-loopback process-link measurements. They do not establish physical WAN or RTX4090 receiver performance. The two markers do not measure text sharpness. Further packet/frame recovery instrumentation is needed before changing default bitrate.</p><p><a href="HEADROOM-REPEATS.json">Complete statistics, timing records and limitations</a></p></html>'
    ]
    (root / "HEADROOM-REPORT.html").write_text("\n".join(body))
    for g in groups:
        print(g)


if __name__ == "__main__":
    main()
