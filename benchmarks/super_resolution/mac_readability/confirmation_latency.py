"""Audit uncensored local completion after the last distinct decoded input."""

import argparse
import json
from pathlib import Path

from benchmarks.super_resolution.mac_readability.quality import save, sha
from benchmarks.super_resolution.mac_readability.stream_report import stats


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root
    folder = root / "stream-runs/confirm-idle-tail-01"
    sender = read(folder / "sender/RESULTS.json")
    receiver = read(folder / "receiver/RESULTS.json")
    primary = read(root / "stream-runs/confirm-refresh-01/sender/RESULTS.json")
    prior_receiver = read(root / "stream-runs/confirm-refresh-01/receiver/RESULTS.json")
    assert (
        sender["complete"]
        and receiver["complete"]
        and sender["ssh_returncode"] == 0
        and not receiver["errors"]
    )
    assert sender["idle_tail_s"] == 1 and len(receiver["source_events"]) == 32
    for key in ["source_sha256", "codec_manifest_sha256", "controls_sha256"]:
        assert sender[key] == primary[key]
    for key in [
        "script_sha256",
        "checkpoint_sha256",
        "display_state_sha256",
        "mode",
        "policy",
        "source_weight",
        "refresh_source_mix",
    ]:
        assert receiver[key] == prior_receiver[key], key
    assert receiver["decoded_frames"] == prior_receiver["decoded_frames"] == 2876
    for a, b in zip(sender["states"], primary["states"]):
        for key in ["state_id", "generation", "group", "raw_sha256"]:
            assert a[key] == b[key]
        assert abs(a["ready_s"] - b["ready_s"] - (a["generation"] - 1)) < 1e-6
    sources = receiver["source_events"]
    rows = []
    for j, source in enumerate(sources):
        gen = source["generation"]
        records = [
            x for x in receiver["records"] if x["generation"] == gen and x["changed"]
        ]
        latest = records[-1]
        revision = latest["codec_revision"]
        updates = sorted(
            [
                x
                for x in receiver["sr_events"] + receiver["mix_refresh_events"]
                if x["generation"] == gen and x["input_revision"] == revision
            ],
            key=lambda x: x["present_ns"],
        )
        boundary = (
            sources[j + 1]["recv_ns"]
            if j + 1 < len(sources)
            else next(
                x["send_ns"] for x in sender["acknowledgments"] if x["event"] == "done"
            )
        )
        available = (boundary - latest["received_ns"]) / 1e6
        assert available >= 200, (source["state_id"], available)
        value = dict(
            state_id=source["state_id"],
            generation=gen,
            last_input_revision=revision,
            last_input_pts_s=latest["pts_us"] / 1e6,
            stable_window_ms=available,
            completed=bool(updates),
            input_to_readback_ms=None,
        )
        if updates:
            update = updates[0]
            assert (
                update["latest_codec_revision"] == revision
                and update["state_id"] == source["state_id"]
            )
            value["input_to_readback_ms"] = (
                update["present_ns"] - latest["received_ns"]
            ) / 1e6
            assert update["present_ns"] < boundary
        rows.append(value)
    completed = [x["input_to_readback_ms"] for x in rows if x["completed"]]
    summary = stats(completed)
    all_complete = len(completed) == 32
    result = dict(
        complete=True,
        scope=__doc__,
        rows=rows,
        summary_ms=summary,
        all32_completed=all_complete,
        local_stable_target_met=all_complete and summary["p95"] <= 200,
        interval="Receipt of finaldistinctdecodedinput in sourcegeneration to SR GPUreadback; packetqueue/decode/inference/renderincluded; excludespriornetworkwaiting andphysicalpanel",
        changed="Oneextraidle secondaftereachoriginal3secondpacketsequence; unchangedpackets/source/receiver/model/pixelblend",
        quality_selection=False,
        test_used=False,
        evidence={
            part: sha(folder / part / "RESULTS.json") for part in ["sender", "receiver"]
        },
    )
    save(root / "IDLE-TAIL-RESULTS.json", result)
    print(
        "All32completed",
        all_complete,
        "localstablep95",
        summary.get("p95"),
        "target",
        result["local_stable_target_met"],
    )


if __name__ == "__main__":
    main()
