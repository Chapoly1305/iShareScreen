"""Summarize the process-local link without discarding failed/missing updates."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rolling_peak(records, seconds):
    rows = sorted(
        (r["time"], r["charged_bytes"]) for r in records if r["status"] == "delivered"
    )
    left = total = peak = 0
    for right, (now, size) in enumerate(rows):
        total += size
        while left < right and rows[left][0] <= now - seconds:
            total -= rows[left][1]
            left += 1
        peak = max(peak, total)
    return peak * 8 / seconds


def percentiles(values):
    return (
        {f"p{p}": float(np.percentile(values, p)) for p in [50, 95, 100]}
        if values
        else None
    )


def analyze(capture):
    status = json.loads((capture / "STATUS.json").read_text())
    path = capture / "LINK.json"
    result = dict(
        capture=capture.name,
        status_sha256=digest(capture / "STATUS.json"),
        state=status["state"],
        error_type=status.get("error_type"),
        error=status.get("error"),
        actual_canvas=status.get("canvas"),
        snapshots=len(status.get("snapshots", [])),
        final_counters=status.get("final_counters"),
        directions={},
        responses=[],
        limitations=[
            "Mac-local socket-boundary emulation, not a physical WAN or RTX4090 receiver",
            "TCP application data plus modeled headers; no real ACK/retransmit, Wi-Fi or VPN overhead",
            "Marker agreement is a decoded content-response signal, not a guarantee of text sharpness or an atomic canvas",
            "Missing scene updates are retained; conditional latency excludes them and cannot establish success alone",
        ],
    )
    if path.exists():
        link = json.loads(path.read_text())
        result["link_sha256"] = digest(path)
        result["link_config"] = link["config"]
        for direction, value in link["directions"].items():
            rows = value["records"]
            delivered = [r for r in rows if r["status"] == "delivered"]
            start = min((r["service_start"] for r in delivered), default=None)
            end = max((r["time"] for r in delivered), default=None)
            streams = {}
            for stream in sorted({r["stream"] for r in rows}):
                subset = [r for r in rows if r["stream"] == stream]
                streams[stream] = dict(
                    status_counts=dict(Counter(r["status"] for r in subset)),
                    offered_payload_bytes=sum(r["payload_bytes"] for r in subset),
                    delivered_payload_bytes=sum(
                        r["payload_bytes"] for r in subset if r["status"] == "delivered"
                    ),
                )
            serial = sorted(delivered, key=lambda r: r["service_start"])
            serialization_pass = all(
                r["service_end"] - r["service_start"] + 1e-6
                >= r["charged_bytes"] * 8 / link["config"]["rate_bps_per_direction"]
                for r in serial
            )
            nonoverlap = all(
                a["service_end"] <= b["service_start"]
                for a, b in zip(serial, serial[1:])
            )
            result["directions"][direction] = dict(
                status_counts=dict(Counter(r["status"] for r in rows)),
                streams=streams,
                delivered_charged_bytes=value["delivered_charged_bytes"],
                delivered_payload_bytes=value["delivered_payload_bytes"],
                queued_bytes=value["queued_bytes"],
                service_span_s=end - start if start is not None else None,
                service_span_charged_bps=value["delivered_charged_bytes"]
                * 8
                / (end - start)
                if start is not None and end > start
                else None,
                serialization_durations_pass=serialization_pass,
                service_nonoverlap_pass=nonoverlap,
                delivered_queue_to_arrival_ms=percentiles(
                    [(r["time"] - r["arrival"]) * 1000 for r in delivered]
                ),
                rolling_peak_charged_bps={
                    str(w): rolling_peak(rows, w) for w in [0.1, 1.0, 10.0]
                },
                rolling_discretization_allowance_bps={
                    str(w): max((r["charged_bytes"] for r in delivered), default=0)
                    * 8
                    / w
                    for w in [0.1, 1.0, 10.0]
                },
            )
        result["socket_receive_overflows"] = link["socket_receive_overflows"]
    stimulus_path = capture / "stimulus/manifest.json"
    if stimulus_path.exists():
        stimulus = json.loads(stimulus_path.read_text())
        result["stimulus_sha256"] = digest(stimulus_path)
        states = stimulus["states"]
        events = status.get("marker_events", [])
        for index, state in enumerate(states):
            start = state["draw_start_ns"]
            end = (
                states[index + 1]["draw_start_ns"]
                if index + 1 < len(states)
                else stimulus["finished_ns"]
            )
            matching = [
                r
                for r in events
                if r["observed_ns"] >= start and r["markers"] == [state["state_id"]] * 2
            ]
            first = min(matching, key=lambda r: r["observed_ns"]) if matching else None
            partial = [
                r
                for r in events
                if start <= r["observed_ns"] < end and state["state_id"] in r["markers"]
            ]
            result["responses"].append(
                dict(
                    state_id=state["state_id"],
                    group=state["group"],
                    warmup=index == 0,
                    visible_ms=(end - start) / 1e6,
                    first_partial_ms=(min(r["observed_ns"] for r in partial) - start)
                    / 1e6
                    if partial
                    else None,
                    first_both_markers_ms=(first["observed_ns"] - start) / 1e6
                    if first
                    else None,
                    observed_before_next_source=bool(
                        first and first["observed_ns"] < end
                    ),
                )
            )
        measured = [r for r in result["responses"] if not r["warmup"]]
        successes = [r for r in measured if r["observed_before_next_source"]]
        result["warm_response_summary"] = dict(
            changes=len(measured),
            current_updates=len(successes),
            missing_or_late=len(measured) - len(successes),
            conditional_ms=percentiles([r["first_both_markers_ms"] for r in successes]),
        )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, action="append", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = dict(
        schema="mac_link_analysis_v1",
        script_sha256=digest(Path(__file__)),
        runs=[analyze(path) for path in args.capture],
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in result["runs"]:
        print(row["capture"], row["state"], row.get("warm_response_summary"))


if __name__ == "__main__":
    main()
