"""Audit real two-host replay; keep local processing and network timing separate."""

import argparse
import csv
import io
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from benchmarks.super_resolution.mac_readability.quality import save, sha, markers
from benchmarks.super_resolution.mac_readability.stream_protocol import HEADER


def read(p):
    return json.loads(p.read_text())


def stats(values):
    return (
        dict(
            n=len(values),
            p50=float(np.median(values)),
            p95=float(np.percentile(values, 95)),
            maximum=float(np.max(values)),
        )
        if values
        else dict(n=0)
    )


def traffic(sender, nettop):
    categories = {
        kind: sum(x["application_bytes"] for x in sender["sent"] if x["kind"] == kind)
        for kind in sorted({x["kind"] for x in sender["sent"]})
    }
    samples = []
    for record in nettop:
        if record.get("returncode") != 0:
            continue
        values = []
        for row in csv.reader(io.StringIO(record["stdout"])):
            if row and row[0].startswith("ssh."):
                values.append(dict(bytes_in=int(row[1]), bytes_out=int(row[2])))
        if len(values) == 2:
            samples.append(
                dict(
                    start_ns=record["start_ns"],
                    end_ns=record["end_ns"],
                    start=values[0],
                    end=values[1],
                    approximate_1s_out_kbps=(
                        values[1]["bytes_out"] - values[0]["bytes_out"]
                    )
                    * 8
                    / 1000,
                )
            )
    assert samples
    t0 = samples[0]["start_ns"]
    t1 = samples[-1]["end_ns"]
    outgoing = samples[-1]["end"]["bytes_out"] - samples[0]["start"]["bytes_out"]
    incoming = samples[-1]["end"]["bytes_in"] - samples[0]["start"]["bytes_in"]
    # Audit the observed fragment schedule, without assuming exact timer wakes.
    cumulative = 0.0
    minimum = float("inf")
    excess = 0.0
    start = sender["replay_start_ns"]
    for event in sender["sent"]:
        elapsed = (event["write_end_ns"] - start) / 1e9
        minimum = min(minimum, cumulative - 75000 * elapsed)
        cumulative += event["budget_bytes"]
        excess = max(excess, cumulative - 75000 * elapsed - minimum)
    return dict(
        paced_application_bytes_by_kind=categories,
        paced_application_bytes=sender["paced_application_bytes"],
        paced_budget_bytes=sender["paced_budget_bytes"],
        feedback_application_bytes=sender["ack_application_bytes"],
        observed_pacing_envelope_extra_bytes=excess,
        os_counter_out_bytes=outgoing,
        os_counter_in_bytes=incoming,
        os_observation_duration_s=(t1 - t0) / 1e9,
        os_mean_out_kbps=outgoing * 8 / ((t1 - t0) / 1e9) / 1000,
        os_approximate_1s_out_kbps=stats(
            [x["approximate_1s_out_kbps"] for x in samples]
        ),
        samples=samples,
        limitations=[
            "OS process-specific SSH counters, not packet-level IP/VPN accounting",
            "One-second nettop intervals are approximate and include startup/drain",
            "Forward application pacing includes80byte/fragment reserve and bounded late-wakeup catch-up; not a strict physicalwire cap",
        ],
    )


def audit_run(root, name, frozen_inputs, native):
    folder = root / "stream-runs" / name
    s = read(folder / "sender/RESULTS.json")
    d = read(folder / "receiver/RESULTS.json")
    assert (
        s["complete"] and d["complete"] and s["ssh_returncode"] == 0 and not d["errors"]
    )
    assert len(s["states"]) == 32 and len(d["source_events"]) == 32
    assert d["received_application_bytes"] == s["paced_application_bytes"] + 9 * (
        HEADER.size + 8
    )
    before = min(
        s["clock_before"], key=lambda x: x["sender_received_ns"] - x["sender_ns"]
    )
    after = min(
        s["clock_after"], key=lambda x: x["sender_received_ns"] - x["sender_ns"]
    )

    def offset(probe):
        return (
            (probe["recv_ns"] - probe["sender_ns"])
            + (probe["send_ns"] - probe["sender_received_ns"])
        ) / 2

    offsets = [offset(before), offset(after)]
    points = []
    checks = []
    for state in s["states"]:
        sid = state["state_id"]
        gen = state["generation"]
        acks = [x for x in s["acknowledgments"] if x.get("generation") == gen]
        events = {
            kind: next((x for x in acks if x["event"] == kind), None)
            for kind in ["first", "sr", "detail"]
        }
        value = dict(state_id=sid, group=state["group"], generation=gen)
        for kind, event in events.items():
            if event is None:
                value[kind] = None
                continue
            position = np.clip(
                (event["source_ns"] - before["sender_ns"])
                / (after["sender_ns"] - before["sender_ns"]),
                0,
                1,
            )
            estimate = offsets[0] * (1 - position) + offsets[1] * position
            value[kind] = dict(
                ack_upper_ms=(event["sender_received_ns"] - event["source_ns"]) / 1e6,
                estimated_source_to_readback_ms=(
                    event["present_ns"] - estimate - event["source_ns"]
                )
                / 1e6,
                local_ms=event.get(
                    "local_ms",
                    event.get(
                        "receive_to_readback_ms", event.get("stable_decoded_to_sr_ms")
                    ),
                ),
            )
        assert value["first"] is not None
        first = np.asarray(
            Image.open(folder / "receiver" / f"s{sid:02d}-first.png").convert("RGB")
        )
        observed = markers(first)
        expected = np.asarray(
            Image.open(frozen_inputs[sid]["decoded_path"]).convert("RGB")
        )
        actual = np.asarray(
            Image.open(folder / "receiver" / f"s{sid:02d}-steady-decoded.png").convert(
                "RGB"
            )
        )
        delta = np.abs(expected.astype(np.int16) - actual.astype(np.int16))
        assert delta.max() <= 1
        check = dict(
            state_id=sid,
            first_markers=observed,
            first_current=observed == [sid, sid],
            steady_decode_max_delta=int(delta.max()),
        )
        if value["detail"] is not None:
            output = np.asarray(
                Image.open(folder / "receiver" / f"s{sid:02d}-detail.png").convert(
                    "RGB"
                )
            )
            reference = np.asarray(Image.open(native[sid]["path"]).convert("RGB"))
            diff = np.abs(output.astype(np.int16) - reference.astype(np.int16))
            assert diff.max() <= 1
            check["detail_max_delta"] = int(diff.max())
            supplement = next(x for x in d["supplements"] if x["generation"] == gen)
            assert supplement["raw_sha256"] == state["raw_sha256"]
        checks.append(check)
        points.append(value)
    result = dict(
        name=name,
        mode=s["mode"],
        decoded_frames=d["decoded_frames"],
        points=points,
        checks=checks,
        all_first_frames_current=all(x["first_current"] for x in checks),
        stale_sr_rejected=d["stale_sr_results"],
        stale_video_frames_skipped=d["stale_video_frames"],
        traffic=traffic(s, read(folder / "sender/NETTOP.json")),
        clock=dict(
            before_rtt_ms=(before["sender_received_ns"] - before["sender_ns"]) / 1e6,
            after_rtt_ms=(after["sender_received_ns"] - after["sender_ns"]) / 1e6,
            offset_drift_ms=(offsets[1] - offsets[0]) / 1e6,
            estimate_note="Best-RTT before/after calibration interpolated; ACK upper bounds use only the Mac clock",
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
        all_accepted_sr_local_ms=stats(
            [x["stable_decoded_to_sr_ms"] for x in d["sr_events"]]
        ),
        all_supplement_local_ms=stats(
            [
                x["receive_to_readback_ms"]
                for x in d["supplements"]
                if "receive_to_readback_ms" in x
            ]
        ),
        evidence={
            part: sha(folder / part / "RESULTS.json") for part in ["sender", "receiver"]
        },
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    r = a.root
    inputs = {
        x["state_id"]: x
        for x in read(r / "dense-inference-inputs/INPUTS.json")["images"]
    }
    native = {
        x["state_id"]: x
        for x in read(r / "precodec-images-v1/IMAGES.json")["images"]
        if x["config"] == "1080p-420-no-codec"
    }
    runs = [
        audit_run(r, name, inputs, native)
        for name in ["stream-full-bypass-01", "stream-full-enhanced-01"]
    ]
    b = {x["state_id"]: x for x in runs[0]["points"]}
    e = {x["state_id"]: x for x in runs[1]["points"]}
    differences = [
        dict(
            state_id=sid,
            group=e[sid]["group"],
            added_first_ms=e[sid]["first"]["local_ms"] - b[sid]["first"]["local_ms"],
        )
        for sid in sorted(b)
    ]
    groups = sorted({x["group"] for x in differences})
    paired = np.array(
        [[x["added_first_ms"] for x in differences if x["group"] == g] for g in groups]
    )
    draws = np.random.default_rng(20261103).integers(
        0, len(groups), (10000, len(groups))
    )
    boot = np.percentile(paired[draws].reshape(10000, -1), 95, axis=1)
    result = dict(
        complete=True,
        scope=__doc__,
        scripts_sha256=sha(Path(__file__)),
        runs=runs,
        paired_first_differences=differences,
        added_first_local_ms=stats([x["added_first_ms"] for x in differences]),
        added_first_p95_group_bootstrap_ci95=np.quantile(boot, [0.025, 0.975]).tolist(),
        test_used=False,
        full_goal_met=False,
        production_selected=False,
        limitations=[
            "Actual two-host transport of frozen recordings, not live desktop capture or Screen Sharing protocol",
            "Software HEVC decode; GPU inference and offscreen synchronized readback; no panel scanout",
            "One sequential bypass/enhanced campaign; grouped bootstrap does not cover time-varying network or power conditions",
            "Counters cover this SSH process; no strict600Kbps physical/VPN-wire guarantee",
            "25% quality gate remains unachieved; no human preference claim; temporal pixel flicker not yet scored",
        ],
    )
    raw = read(r / "stream-runs/stream-full-enhanced-01/receiver/RESULTS.json")
    phase_rows = []
    for source in raw["source_events"]:
        gen = source["generation"]
        events = [
            (x["present_ns"], "base")
            for x in raw["records"]
            if x["generation"] == gen and "present_ns" in x
        ]
        events += [
            (x["present_ns"], "sr") for x in raw["sr_events"] if x["generation"] == gen
        ]
        events += [
            (x["present_ns"], "detail")
            for x in raw["supplements"]
            if x["generation"] == gen and "present_ns" in x
        ]
        transitions = []
        for timestamp, phase in sorted(events):
            if not transitions or phase != transitions[-1]["phase"]:
                transitions.append(dict(time_ns=timestamp, phase=phase))
        count = sum(
            x["phase"] == "sr" and y["phase"] == "base"
            for x, y in zip(transitions, transitions[1:])
        )
        phase_rows.append(
            dict(
                state_id=source["state_id"],
                generation=gen,
                transitions=transitions,
                sr_to_base_reversions=count,
            )
        )
    phase_audit = dict(
        complete=True,
        scope="Observed GPU-readback stage transitions, not a perceptual flicker score",
        total_sr_to_base_reversions=sum(x["sr_to_base_reversions"] for x in phase_rows),
        sources_with_reversions=sum(x["sr_to_base_reversions"] > 0 for x in phase_rows),
        rows=phase_rows,
        test_used=False,
    )
    save(r / "STREAM-PHASE-AUDIT.json", phase_audit)
    result["phase_audit"] = dict(
        total_sr_to_base_reversions=phase_audit["total_sr_to_base_reversions"],
        sources_with_reversions=phase_audit["sources_with_reversions"],
        evidence_sha256=sha(r / "STREAM-PHASE-AUDIT.json"),
    )
    save(r / "STREAM-RESULTS.json", result)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), layout="constrained")
    for kind, label in [
        ("first", "First decoded display"),
        ("sr", "First current SR"),
        ("detail", "Exact source refinement"),
    ]:
        axes[0].plot(
            list(e),
            [e[sid][kind]["ack_upper_ms"] if e[sid][kind] else np.nan for sid in e],
            marker=".",
            label=label,
        )
    axes[0].set(
        xlabel="Frozen source state",
        ylabel="Scheduled update to ACK (ms)",
        title="Actual Mac-to4090 replay; same-clock upper bounds",
    )
    axes[0].legend(fontsize=8)
    axes[1].bar(list(e), [x["added_first_ms"] for x in differences])
    axes[1].axhline(20, color="red", ls="--")
    axes[1].set(
        xlabel="Frozen source state",
        ylabel="Added first local display time (ms)",
        title="Enhanced minus bypass, matched sources",
    )
    fig.savefig(r / "stream-curves.png", dpi=150)
    plt.close(fig)
    enhanced = runs[1]
    summary = enhanced["summary"]
    rows = ""
    for kind, label in [
        ("first", "First decoded display"),
        ("sr", "First accepted SR"),
        ("detail", "Exact source refinement"),
    ]:
        x = summary[kind]
        rows += f"<tr><td>{label}</td><td>{x['completed']}/32</td><td>{x['ack_upper_ms'].get('p50', float('nan')):.1f}</td><td>{x['ack_upper_ms'].get('p95', float('nan')):.1f}</td><td>{x['local_ms'].get('p95', float('nan')):.1f}</td></tr>"
    t = enhanced["traffic"]
    document = f"""<!doctype html><meta charset="utf-8"><title>Actual Mac-to4090 replay</title>
<style>body{{font:17px system-ui;max-width:1100px;margin:40px auto;line-height:1.55}}td,th{{padding:10px;border:1px solid #ddd}}table{{border-collapse:collapse}}img{{max-width:100%}}.limit{{background:#fff4d8;padding:16px}}</style>
<h1>Actual Mac-to-RTX4090 replay: base video, asynchronous SR and source refinement</h1>
<p>All 16 frozen development documents (32 source states) were sent from the M1 Mac to the verified RTX 4090 host over the existing authenticated SSH connection. The receiver decoded the actual Mac HEVC packets, ran the fixed best EDSR checkpoint asynchronously, and rendered received source refinements. Both modes carried video and supplements; bypass ignored enhancement to provide a matched traffic control.</p>
<p class="limit">This is replay of frozen Mac recordings, not new live desktop capture. Forward traffic was paced using one 600 Kbps application/framing budget plus an 80-byte fragment allowance. OS SSH counters were recorded separately. The experiment does not establish a strict physical TCP/VPN cap, NVDEC performance, or panel latency.</p>
<table><tr><th>Enhanced path</th><th>Completed</th><th>Update→ACK p50 (ms)</th><th>Update→ACK p95 (ms)</th><th>Local stage p95 (ms)</th></tr>{rows}</table>
<p>Update→ACK uses only the Mac clock and includes the receiver display event plus return-path acknowledgment, making it an upper bound on scheduled replay update to GPU readback. Local first-display timing starts at receipt of the complete encoded frame. Local SR timing starts at receipt of the last distinct decoded input accepted by that SR job. Local source-refinement timing starts when its complete supplement is received.</p>
<p>Paired added first-display p95: <b>{result["added_first_local_ms"]["p95"]:.2f} ms</b>; descriptive grouped-bootstrap interval {result["added_first_p95_group_bootstrap_ci95"][0]:.2f}–{result["added_first_p95_group_bootstrap_ci95"][1]:.2f} ms. All accepted SR jobs had local p95 {enhanced["all_accepted_sr_local_ms"].get("p95", float("nan")):.1f} ms. Rejected outdated SR results: {enhanced["stale_sr_rejected"]}; those outputs were not displayed.</p>
<p>All first displayed state markers current: {enhanced["all_first_frames_current"]}. All steady decoded inputs match their frozen controls within one RGB level; completed source refinements independently match native 4:2:0 references within one level and have the exact expected raw-source hash.</p>
<p>OS-reported SSH outgoing mean: {t["os_mean_out_kbps"]:.1f} Kbps over its observation interval; maximum approximate one-second sample: {t["os_approximate_1s_out_kbps"]["maximum"]:.1f} Kbps. These process counters are not packet-level IP/VPN measurements. Forward application bytes: {t["paced_application_bytes"]:,}; feedback application bytes: {t["feedback_application_bytes"]:,}.</p>
<p>The first smoke test exposed macOS sleep coalescing: resetting every send deadline after a late wakeup unintentionally reduced throughput. Absolute deadlines with bounded catch-up restored useful capacity; the original smoke evidence remains preserved.</p>
<p>A remaining display issue is now measured: {phase_audit["total_sr_to_base_reversions"]} transitions from SR back to the base image occurred within {phase_audit["sources_with_reversions"]} unchanged source generations, before the exact supplement arrived. This counts actual rendered stages, not subjective flicker. The receiver currently invalidates SR on codec-pixel changes even when the announced source is unchanged. Source-generation ownership and temporal image differences need a separate controlled fix. The quality goal and independent confirmation also remain open.</p>
<img src="stream-curves.png"><p><a href="STREAM-RESULTS.json">Per-state results, counters and evidence hashes</a> · <a href="COMPONENT-REPORT.html">Source quality and codec evidence</a></p>"""
    (r / "STREAM-REPORT.html").write_text(document)
    print("addedfirst", result["added_first_local_ms"])
    print("enhanced", enhanced["summary"])
    print("allSR", enhanced["all_accepted_sr_local_ms"])
    print("OS rate", t["os_mean_out_kbps"], t["os_approximate_1s_out_kbps"]["maximum"])


if __name__ == "__main__":
    main()
