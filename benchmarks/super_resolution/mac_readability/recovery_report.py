"""Relate source responses to decoder feed/error counters without claiming causality."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.super_resolution.mac_readability.analyze_link import analyze
from benchmarks.super_resolution.mac_readability.quality import save, sha


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    rows = []
    runs = []
    stalls = []
    for path in sorted(root.glob("capture-recovery-*-600k")):
        run = analyze(path)
        status = read(path / "STATUS.json")
        stim = read(path / "stimulus/manifest.json")
        link = read(path / "LINK.json")
        assert status["state"] == "completed" and status["canvas"] == [1280, 720]
        assert (
            link["config"]["rate_bps_per_direction"] == 600000
            and status["offered"]["kbps"] == 600
        )
        samples = status["samples"]
        assert max(x["time_ns"] for x in samples) >= stim["finished_ns"]
        runs.append(run)
        # Look beyond the first second of each source transition: a long stall
        # may begin later or span two source states. Keep all >=500ms gaps.
        segment = None

        def finish_segment(first, last):
            if first is None or (last["time_ns"] - first["time_ns"]) < 500000000:
                return
            feeds = (
                last["recovery"]["decoder_feed_total"]
                - first["recovery"]["decoder_feed_total"]
            )
            errors = (
                last["recovery"]["decoder_error_total"]
                - first["recovery"]["decoder_error_total"]
            )
            active = [
                x
                for x in samples
                if max(first["time_ns"], stim["states"][0]["draw_start_ns"])
                <= x["time_ns"]
                <= min(last["time_ns"], stim["finished_ns"])
            ]
            active_gap = (
                (active[-1]["time_ns"] - active[0]["time_ns"]) / 1e6
                if len(active) > 1
                else 0.0
            )
            active_feeds = (
                active[-1]["recovery"]["decoder_feed_total"]
                - active[0]["recovery"]["decoder_feed_total"]
                if len(active) > 1
                else 0
            )
            active_errors = (
                active[-1]["recovery"]["decoder_error_total"]
                - active[0]["recovery"]["decoder_error_total"]
                if len(active) > 1
                else 0
            )
            stalls.append(
                dict(
                    capture=path.name,
                    start_ns=first["time_ns"],
                    end_ns=last["time_ns"],
                    observed_gap_ms=(last["time_ns"] - first["time_ns"]) / 1e6,
                    decoder_feed_delta=feeds,
                    decoder_error_delta=errors,
                    active_stimulus_gap_ms=active_gap,
                    active_stimulus_feed_delta=active_feeds,
                    active_stimulus_error_delta=active_errors,
                    active_stimulus_fed_erroring_stall=bool(
                        active_gap >= 500 and active_feeds > 30 and active_errors > 30
                    ),
                    video_packets_received_delta=last["received_packets"]
                    - first["received_packets"],
                    fed_and_erroring_without_new_frames=bool(
                        feeds > 30 and errors > 30
                    ),
                    end_recovery=last["recovery"],
                )
            )

        for before_sample, after_sample in zip(samples, samples[1:]):
            if (
                after_sample["decoder_good_counts"]
                == before_sample["decoder_good_counts"]
            ):
                if segment is None:
                    segment = before_sample
            else:
                finish_segment(segment, before_sample)
                segment = None
        finish_segment(segment, samples[-1])
        for response in run["responses"]:
            if response["warmup"]:
                continue
            source = next(
                x for x in stim["states"] if x["state_id"] == response["state_id"]
            )
            start = source["draw_start_ns"]
            end = start + 1000000000
            before = max(
                (x for x in samples if x["time_ns"] <= start),
                key=lambda x: x["time_ns"],
            )
            after = max(
                (x for x in samples if x["time_ns"] <= end), key=lambda x: x["time_ns"]
            )
            window = [x for x in samples if start <= x["time_ns"] <= end]
            assert window
            rb = before["recovery"]
            ra = after["recovery"]
            good = [
                b - a
                for a, b in zip(
                    before["decoder_good_counts"], after["decoder_good_counts"]
                )
            ]
            assert all(v >= 0 for v in good), (
                "Decoder counter reset requires separate treatment"
            )
            feed = ra["decoder_feed_total"] - rb["decoder_feed_total"]
            errors = ra["decoder_error_total"] - rb["decoder_error_total"]
            incoming = link["directions"]["incoming"]["records"]
            offered = [
                x
                for x in incoming
                if start / 1e9 <= x["arrival"] < end / 1e9 and x["stream"] == "udp-5901"
            ]
            delivered = [
                x
                for x in incoming
                if start / 1e9 <= x["time"] < end / 1e9
                and x["stream"] == "udp-5901"
                and x["status"] == "delivered"
            ]
            row = dict(
                capture=path.name,
                **response,
                source_draw_ms=(source["draw_return_ns"] - start) / 1e6,
                sample_interval_ms=(after["time_ns"] - before["time_ns"]) / 1e6,
                sample_start_offset_ms=(before["time_ns"] - start) / 1e6,
                sample_end_offset_ms=(after["time_ns"] - end) / 1e6,
                video_packets_received_delta=after["received_packets"]
                - before["received_packets"],
                detected_sequence_gaps_delta=after["lost_packets"]
                - before["lost_packets"],
                decoder_feed_delta=feed,
                decoder_error_delta=errors,
                good_frames_delta=good,
                fed_and_erroring_without_new_frames=bool(
                    feed > 30 and errors > 30 and sum(good) == 0
                ),
                max_video_queue=max(x["video_queue_depth"] for x in window),
                video_queue_drops_delta=after["video_queue_dropped"]
                - before["video_queue_dropped"],
                offered_video_bytes_first_second=sum(
                    x["payload_bytes"] for x in offered
                ),
                delivered_video_bytes_first_second=sum(
                    x["payload_bytes"] for x in delivered
                ),
                tail_drops_first_second=sum(
                    x["status"] == "tail_drop" for x in offered
                ),
                end_recovery=ra,
            )
            rows.append(row)
    assert len(runs) == 3 and len(rows) == 21
    ok = [x["first_both_markers_ms"] for x in rows if x["observed_before_next_source"]]
    stalled = [x for x in rows if x["fed_and_erroring_without_new_frames"]]
    slow = [
        x
        for x in rows
        if not x["observed_before_next_source"] or x["first_both_markers_ms"] >= 600
    ]
    summary = dict(
        changes=len(rows),
        observed_before_next_source=len(ok),
        missing_or_late=len(rows) - len(ok),
        conditional_median_ms=float(np.median(ok)),
        conditional_p95_ms=float(np.percentile(ok, 95)),
        diagnostic_slow_or_missing=len(slow),
        fed_erroring_no_frames_first_second=len(stalled),
        stalled_without_new_sequence_gap=sum(
            x["detected_sequence_gaps_delta"] == 0 for x in stalled
        ),
        entire_capture_no_output_intervals=len(stalls),
        entire_capture_fed_erroring_stalls=sum(
            x["fed_and_erroring_without_new_frames"] for x in stalls
        ),
        active_stimulus_fed_erroring_stalls=sum(
            x["active_stimulus_fed_erroring_stall"] for x in stalls
        ),
    )
    output = dict(
        schema="mac_recovery_diagnostics_v1",
        complete=True,
        script_sha256=sha(Path(__file__)),
        scope="ActualMac loopback ScreenSharing virtual720,600Kbps request and600Kbps process-local link; Mac native420 decoder; not physicalWAN/4090 receiver",
        summary=summary,
        runs=runs,
        responses=rows,
        no_output_intervals=stalls,
        conclusion="Repeated feeds plus explicit decoder errors and no new frames locate some delay inside the decode/recovery path, not an accumulating receiver queue. This does not identify the exact cause of the missing references or establish a fix.",
        limitations=[
            "First-second counter differences bracket events with approximately100ms samples and are not atomic.",
            "Sequence gaps can reflect earlier losses; absence of a new gap does not mean an intact reference chain.",
            "draw_return is CPU submission, not an observed source panel/compositor present.",
            "Markers measure decoded state visibility, not text quality or atomic whole-frame freshness.",
            "A600ms diagnostic threshold identifies long responses; it is not the200ms refinement acceptance threshold.",
            "Decoder latency EMA can remain low during a stall because no successful decode updates it.",
        ],
        test_used=False,
    )
    save(root / "RECOVERY-DIAGNOSTICS.json", output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout="constrained")
    for i, x in enumerate(rows):
        value = (
            x["first_both_markers_ms"]
            if x["observed_before_next_source"]
            else x["visible_ms"]
        )
        color = "#d06030" if x["fed_and_erroring_without_new_frames"] else "#286b9c"
        axes[0].scatter(
            i + 1,
            value,
            color=color,
            marker="o" if x["observed_before_next_source"] else "x",
        )
        axes[1].scatter(
            x["decoder_error_delta"],
            value,
            color=color,
            marker="o" if x["observed_before_next_source"] else "x",
        )
    axes[0].set(
        xlabel="Source change across three runs",
        ylabel="Decoded marker response (ms)",
        title="All21 changes; crosses are missing updates",
    )
    axes[1].set(
        xlabel="Decode errors in first-second sample interval",
        ylabel="Decoded marker response (ms)",
        title="Orange: fed and erroring, no new frames",
    )
    fig.savefig(root / "recovery-diagnostics.png", dpi=170)
    plt.close(fig)
    table = "".join(
        f"<tr><td>{x['capture']}</td><td>{x['state_id']}</td><td>{x['first_both_markers_ms'] if x['observed_before_next_source'] else 'Missing'}</td><td>{x['decoder_feed_delta']}</td><td>{x['decoder_error_delta']}</td><td>{x['good_frames_delta']}</td><td>{x['max_video_queue']}</td></tr>"
        for x in slow
    )
    (root / "RECOVERY-DIAGNOSTICS.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Screen Sharing recovery diagnosis</title><style>body{font:16px system-ui;max-width:1200px;margin:40px auto}td,th{padding:9px;border-bottom:1px solid #ddd}img{max-width:100%}</style><h1>Screen Sharing decode/recovery diagnosis</h1><p>Three repeated600Kbps requests under the same600Kbps process-local link. Native Mac420 decoding,1280×720 virtual desktop. All21 post-warmup source changes retained.</p><pre>'
        + json.dumps(summary, indent=2)
        + '</pre><img src="recovery-diagnostics.png"><p>Orange points identify windows with more than30 feeds and30 decoder errors but zero new frames. Such windows demonstrate failed decoding while input continues. They do not establish which upstream loss/reference event caused the failure. A low decode-latency EMA during a stall is not evidence of fast response.</p><h2>Responses over600ms or missing</h2><table><tr><th>Run</th><th>State</th><th>Response ms</th><th>Feeds</th><th>Errors</th><th>Good frames per tile</th><th>Max receiver queue</th></tr>'
        + table
        + '</table><p>Counter intervals are approximately1second and carry explicit sample offsets in the JSON. This is Mac-loopback diagnostic evidence, not physicalWAN or RTX4090 end-to-end performance. Private captures and authentication material remain on the Mac.</p><a href="RECOVERY-DIAGNOSTICS.json">All observations and source hashes</a>'
    )
    print(summary)
    for x in slow:
        print(
            x["capture"],
            x["state_id"],
            x["first_both_markers_ms"],
            x["decoder_feed_delta"],
            x["decoder_error_delta"],
            x["good_frames_delta"],
        )


if __name__ == "__main__":
    main()
