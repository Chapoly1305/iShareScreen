"""Compare rendered phases, timed OCR, and local versus network latency."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.super_resolution.mac_readability.quality import save, sha, summarize
from benchmarks.super_resolution.mac_readability.stream_protocol import HEADER
from benchmarks.super_resolution.mac_readability.stream_report import stats, traffic


def read(path):
    return json.loads(path.read_text())


def audit(root, name):
    folder = root / "stream-runs" / name
    sender, receiver = [
        read(folder / part / "RESULTS.json") for part in ["sender", "receiver"]
    ]
    assert sender["complete"] and receiver["complete"] and sender["ssh_returncode"] == 0
    assert not receiver["errors"] and len(receiver["source_events"]) == 32
    assert receiver["received_application_bytes"] == sender[
        "paced_application_bytes"
    ] + 9 * (HEADER.size + 8)
    phases, points = [], []
    for source in receiver["source_events"]:
        gen = source["generation"]
        events = [
            (x["present_ns"], "base")
            for x in receiver["records"]
            if x["generation"] == gen and "present_ns" in x
        ]
        events += [
            (x["present_ns"], "sr")
            for x in receiver["sr_events"]
            if x["generation"] == gen
        ]
        events += [
            (x["present_ns"], "detail")
            for x in receiver["supplements"]
            if x["generation"] == gen and "present_ns" in x
        ]
        events += [
            (x["present_ns"], "detail")
            for x in receiver.get("mix_refresh_events", [])
            if x["generation"] == gen
        ]
        sequence = []
        for timestamp, phase in sorted(events):
            if not sequence or sequence[-1]["phase"] != phase:
                sequence.append(dict(phase=phase, time_ns=timestamp))
        phases.append(
            dict(
                state_id=source["state_id"],
                generation=gen,
                transitions=sequence,
                sr_to_base=sum(
                    a["phase"] == "sr" and b["phase"] == "base"
                    for a, b in zip(sequence, sequence[1:])
                ),
            )
        )
        acks = [x for x in sender["acknowledgments"] if x.get("generation") == gen]
        point = dict(state_id=source["state_id"], generation=gen)
        for kind in ["first", "sr", "detail"]:
            event = next((x for x in acks if x["event"] == kind), None)
            point[kind] = (
                None
                if event is None
                else dict(
                    ack_upper_ms=(event["sender_received_ns"] - event["source_ns"])
                    / 1e6,
                    local_ms=event.get(
                        "local_ms",
                        event.get(
                            "receive_to_readback_ms",
                            event.get("input_received_to_sr_ms"),
                        ),
                    ),
                )
            )
        first_record = next(
            x
            for x in receiver["records"]
            if x["generation"] == gen and "present_ns" in x
        )
        sr = next((x for x in receiver["sr_events"] if x["generation"] == gen), None)
        point["first_packet_receipt_to_first_sr_ms"] = (
            None
            if sr is None
            else (sr["present_ns"] - first_record["received_ns"]) / 1e6
        )
        points.append(point)
    temporal = [
        x
        for x in receiver["temporal_samples"]
        if x["age_s"] <= 2.8 and "mean_abs_rgb_delta" in x
    ]
    same_phase = [
        x
        for x in temporal
        if x["phase"] == x["previous_phase"] and x["phase"] != "blank"
    ]
    result = dict(
        name=name,
        policy=receiver["policy"],
        source_weight=receiver.get("source_weight", 1.0),
        complete=True,
        decoded_frames=receiver["decoded_frames"],
        phases=phases,
        points=points,
        sr_to_base_reversions=sum(x["sr_to_base"] for x in phases),
        sources_with_reversions=sum(x["sr_to_base"] > 0 for x in phases),
        sampled_same_phase_delta=stats([x["mean_abs_rgb_delta"] for x in same_phase]),
        sampled_all_phase_delta=stats([x["mean_abs_rgb_delta"] for x in temporal]),
        temporal_scope="100ms grid-stride4 RGB samples restricted to first2.8s after receiver source announcement; not a human flicker score",
        all_sr_input_receipt_to_readback_ms=stats(
            [x["input_received_to_sr_ms"] for x in receiver["sr_events"]]
        ),
        first_packet_receipt_to_first_sr_ms=stats(
            [
                x["first_packet_receipt_to_first_sr_ms"]
                for x in points
                if x["first_packet_receipt_to_first_sr_ms"] is not None
            ]
        ),
        summary={
            kind: dict(
                completed=sum(x[kind] is not None for x in points),
                total=32,
                ack_upper_ms=stats(
                    [x[kind]["ack_upper_ms"] for x in points if x[kind] is not None]
                ),
                local_ms=stats(
                    [x[kind]["local_ms"] for x in points if x[kind] is not None]
                ),
            )
            for kind in ["first", "sr", "detail"]
        },
        traffic=traffic(sender, read(folder / "sender/NETTOP.json")),
        evidence={
            part: sha(folder / part / "RESULTS.json") for part in ["sender", "receiver"]
        },
    )
    if receiver.get("refresh_source_mix"):
        refreshes = receiver["mix_refresh_events"]
        result["refresh_job_input_to_readback_ms"] = stats(
            [x["input_received_to_sr_ms"] for x in refreshes]
        )
        result["refreshed_states"] = len({x["state_id"] for x in refreshes})
        result["refresh_count"] = len(refreshes)
        last = []
        for source in receiver["source_events"]:
            candidates = [
                x
                for x in sender["acknowledgments"]
                if x.get("generation") == source["generation"]
                and x["event"] in ["detail", "mix_refresh"]
            ]
            event = candidates[-1]
            last.append(
                dict(
                    state_id=source["state_id"],
                    ack_upper_ms=(event["sender_received_ns"] - event["source_ns"])
                    / 1e6,
                )
            )
        result["final_refinement_ack_upper_ms"] = stats(
            [x["ack_upper_ms"] for x in last]
        )
        result["final_refinement_points"] = last
    if (folder / "timed-quality/PROTOCOL.json").exists():
        assert read(folder / "timed-quality/PROTOCOL.json")["complete"]
        result["timed_ocr"] = read(folder / "timed-quality/SUMMARY.json")
        result["timed_phases"] = {
            str(age): dict(
                Counter(
                    x["phase"]
                    for x in receiver["timed_images"]
                    if x["age_target_s"] == age
                )
            )
            for age in [0.5, 1, 2, 2.8]
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--include-refresh", action="store_true")
    args = parser.parse_args()
    root = args.root
    names = ["pixel", "source", "holdmix", "bypass"]
    if args.include_refresh:
        names.insert(-1, "refresh")
    runs = {name: audit(root, f"stream-stability-{name}-01") for name in names}
    bypass = {x["state_id"]: x for x in runs["bypass"]["points"]}
    for name in names[:-1]:
        run = runs[name]
        run["added_first_local_ms"] = stats(
            [
                x["first"]["local_ms"] - bypass[x["state_id"]]["first"]["local_ms"]
                for x in run["points"]
            ]
        )
    assert runs["holdmix"]["sr_to_base_reversions"] == 0
    if args.include_refresh:
        assert runs["refresh"]["sr_to_base_reversions"] == 0
    receiver = read(
        root / "stream-runs/stream-stability-holdmix-01/receiver/RESULTS.json"
    )
    assert all(
        x["input_revision"] == x["latest_codec_revision"] for x in receiver["sr_events"]
    )
    # Blank/late displays remain in the primary timed metric. This paired subset
    # is only a diagnostic of how early SR behaves once both runs have pixels.
    available = []
    for name in ["pixel", "source"]:
        data = read(
            root / f"stream-runs/stream-stability-{name}-01/receiver/RESULTS.json"
        )
        available.append(
            {
                x["state_id"]
                for x in data["timed_images"]
                if x["age_target_s"] == 0.5 and x["phase"] != "blank"
            }
        )
    common = available[0] & available[1]
    early = {}
    for name in ["pixel", "source"]:
        rows = read(
            root / f"stream-runs/stream-stability-{name}-01/timed-quality/ROWS.json"
        )
        selected = [
            x for x in rows if x["source_age_s"] == 0.5 and x["state_id"] in common
        ]
        early[name] = summarize(selected)[0]
    output = dict(
        complete=True,
        script_sha256=sha(Path(__file__)),
        runs=runs,
        matched_nonblank_half_second=dict(states=sorted(common), results=early),
        decision="Keep hold: retain accepted SR within unchanged source, but require latest codec revision at publication. Reject aggressive source policy after early OCR regression.",
        test_used=False,
        full_goal_met=False,
        limitations=[
            "Frozen recording replay, not live capture or Screen Sharing protocol",
            "One sequential run per condition; network and workload variation are not controlled repetitions",
            "Holdmix changes both display policy and final blend; do not attribute final quality changes to policy alone",
            "Software HEVC, CUDA AMP SR, WGPU synchronized readback; excludes physical panel",
            "Job-input local latency excludes waits for an admissible input and supplement transmission",
            "Timed OCR ages start at receiver announcement, not user action or frozen codec PTS",
            "600Kbps application/framing allowance; no strict physical wire/VPN cap",
        ],
    )
    save(root / "STREAM-STABILITY-RESULTS.json", output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for name in names[:-1]:
        rows = sorted(runs[name]["timed_ocr"], key=lambda x: x["source_age_s"])
        axes[0].plot(
            [x["source_age_s"] for x in rows],
            [x["macro_cer"] * 100 for x in rows],
            ".-",
            label=name,
        )
    axes[0].set(
        xlabel="Receiver announcement age (s)",
        ylabel="Full16 document macro CER (%)",
        title="Actual visible pixels, including blank displays",
    )
    axes[0].legend()
    axes[1].bar(names[:-1], [runs[n]["sr_to_base_reversions"] for n in names[:-1]])
    axes[1].set(
        ylabel="SR to base reversions", title="Within unchanged source generations"
    )
    fig.savefig(root / "stream-stability-curves.png", dpi=150)
    plt.close(fig)
    rows = ""
    for name in names[:-1]:
        run = runs[name]
        timed = sorted(run["timed_ocr"], key=lambda x: x["source_age_s"])
        rows += (
            f"<tr><td>{name}</td><td>{run['sr_to_base_reversions']}</td>"
            + "".join(f"<td>{x['macro_cer'] * 100:.3f}%</td>" for x in timed)
            + f"<td>{run['added_first_local_ms']['p95']:.2f}</td></tr>"
        )
    chosen = runs["holdmix"]
    s = chosen["summary"]
    t = chosen["traffic"]
    refresh_note = ""
    if args.include_refresh:
        v = runs["refresh"]
        refresh_note = f"<p>Continued fusion run: {v['refreshed_states']}/32 states received later source-backed updates; {v['refresh_count']} refreshes. Added first local p95 {v['added_first_local_ms']['p95']:.2f}ms; accepted refresh input→readback p95 {v['refresh_job_input_to_readback_ms']['p95']:.1f}ms. Final source-backed update→ACK p95 {v['final_refinement_ack_upper_ms']['p95']:.1f}ms. Final means the last update observed before the next source, not a proven future convergence point.</p>"
    document = f"""<!doctype html><meta charset="utf-8"><title>Display stability and actual stream OCR</title>
<style>body{{font:17px system-ui;max-width:1100px;margin:40px auto;line-height:1.5}}table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:9px}}img{{max-width:100%}}</style>
<h1>Stable display with source-backed refinement</h1>
<p>All32 source states completed on the actual M1 Mac → RTX4090 replay. Hold preserves a valid SR image during codec fluctuations, while still rejecting an outdated codec revision when publishing a new SR result. A new source always invalidates previous content.</p>
<table><tr><th>Policy</th><th>SR→base reversions</th><th>0.5s CER</th><th>1s CER</th><th>2s CER</th><th>2.8s CER</th><th>Added first local p95 ms</th></tr>{rows}</table>
<p>Timed CER includes all16 document groups and blank images. It is distinct from the frozen hard15 steady-frame metric. In the matched26 nonblank states at0.5s, pixel CER was {early["pixel"]["macro_cer"] * 100:.2f}% and aggressive source policy CER was {early["source"]["macro_cer"] * 100:.2f}%. Therefore the aggressive policy was rejected.</p>
<p>Holdmix uses25% exact native420 source and75% current-source SR when available. Its final blend and display policy changed together; the final-quality difference cannot be attributed to policy alone. See the separate quality report for fallback and regression accounting.</p>
<p>Holdmix: added local first-display p95 {chosen["added_first_local_ms"]["p95"]:.2f}ms; accepted SR job-input→readback p95 {chosen["all_sr_input_receipt_to_readback_ms"]["p95"]:.1f}ms. First encoded-packet receipt→first accepted SR p95 was {chosen["first_packet_receipt_to_first_sr_ms"]["p95"]:.1f}ms, which includes waiting for an admissible input and is a different, more conservative interval.</p>
<p>Scheduled source change→ACK p95: first display {s["first"]["ack_upper_ms"]["p95"]:.1f}ms, first SR {s["sr"]["ack_upper_ms"]["p95"]:.1f}ms, source-backed refinement {s["detail"]["ack_upper_ms"]["p95"]:.1f}ms. Complete supplement receipt→readback p95 {s["detail"]["local_ms"]["p95"]:.1f}ms. None of these supports a200ms end-to-end refinement claim.</p>
<p>OS SSH outgoing mean {t["os_mean_out_kbps"]:.1f}Kbps; largest approximate1s sample {t["os_approximate_1s_out_kbps"]["maximum"]:.1f}Kbps. This is application-paced frozen replay, not a strict physical-wire cap, live desktop capture, NVDEC, or panel latency measurement. One sequential run per condition cannot establish reproducibility across network states.</p>
{refresh_note}
<img src="stream-stability-curves.png"><p><a href="STREAM-STABILITY-RESULTS.json">Raw audit, per-state timings and counters</a></p>"""
    (root / "STREAM-STABILITY-REPORT.html").write_text(document)
    print(
        json.dumps(
            {
                k: dict(reversions=v["sr_to_base_reversions"], summary=v["summary"])
                for k, v in runs.items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
