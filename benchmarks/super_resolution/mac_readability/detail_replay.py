"""Validate Mac-generated tiles and separate byte-budget quality from link replay.

Quality images overlay source tiles on a frozen steady EDSR output. They are
NOT images observed at a wall-clock deadline. Separately replay actual Mac
VideoToolbox packet sizes/PTS plus the same detail packets through a modeled
shared 600 Kbps serializer, base-video priority, no initial token credit.
"""

import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time

import av
import numpy as np
from PIL import Image

from benchmarks.super_resolution.mac_readability.detail_transport import (
    Receiver,
    wire_bytes,
)
from benchmarks.super_resolution.mac_readability.prepare_precodec import planar_nv12
from benchmarks.super_resolution.mac_readability.quality import sha, save


def fragments(size):
    return [min(1200, size - i) + 44 for i in range(0, size, 1200)]


def shared_link(run, images, rate=600000):
    base = []
    for index, p in enumerate(run["probe"]["packets"]):
        ready = float(p.get("dts_time", p["pts_time"]))
        chunks = fragments(int(p["size"]))
        for i, size in enumerate(chunks):
            base.append(
                dict(
                    kind="base",
                    ready=max(0, ready),
                    wire=size,
                    frame=index,
                    last=i == len(chunks) - 1,
                )
            )
    base = deque(sorted(base, key=lambda v: (v["ready"], v["frame"])))
    detail = []
    for state_index, row in enumerate(images):
        ready = state_index * 3 + 0.1 + row["encode_ms"] / 1000
        detail.append(
            dict(
                kind="detail",
                ready=ready,
                wire=wire_bytes(32),
                state_index=state_index,
                tile=-1,
                last=True,
            )
        )
        for index, tile in enumerate(row["tiles"]):
            chunks = fragments(tile["payload_bytes"])
            assert sum(chunks) == tile["wire_bytes"]
            for j, size in enumerate(chunks):
                detail.append(
                    dict(
                        kind="detail",
                        ready=ready,
                        wire=size,
                        state_index=state_index,
                        tile=index,
                        last=j == len(chunks) - 1,
                    )
                )
    detail = deque(detail)
    now = 0.0
    events = []
    cancelled = 0
    frame_delays = []
    while base or detail:
        while detail and now >= 3 * (detail[0]["state_index"] + 1):
            cancelled += detail.popleft()["wire"]
        if now >= 6:
            break
        if base and base[0]["ready"] <= now:
            item = base.popleft()
        elif detail and detail[0]["ready"] <= now:
            item = detail.popleft()
        else:
            now = min(
                base[0]["ready"] if base else float("inf"),
                detail[0]["ready"] if detail else float("inf"),
            )
            continue
        start = now
        now += item["wire"] * 8 / rate
        accepted = item["kind"] == "base" or now < 3 * (item["state_index"] + 1)
        events.append(dict(item, start_s=start, finish_s=now, accepted=accepted))
        if item["kind"] == "base" and item["last"]:
            frame_delays.append((now - item["ready"]) * 1000)
    detail_states = []
    for state_index, row in enumerate(images):
        completions = [
            e
            for e in events
            if e["kind"] == "detail"
            and e["state_index"] == state_index
            and e["last"]
            and e["tile"] >= 0
            and e["accepted"]
        ]
        done = {e["tile"]: e["finish_s"] - 3 * state_index for e in completions}
        snapshots = []
        for age in [0.2, 0.5, 1.0, 2.8]:
            ids = [i for i, t in done.items() if t <= age]
            snapshots.append(
                dict(
                    age_s=age,
                    tiles=len(ids),
                    pixels=sum(
                        row["tiles"][i]["width"] * row["tiles"][i]["height"]
                        for i in ids
                    ),
                )
            )
        detail_states.append(
            dict(
                state_id=row["state_id"],
                completed_tiles=len(done),
                total_tiles=len(row["tiles"]),
                complete_age_s=max(done.values())
                if len(done) == len(row["tiles"])
                else None,
                snapshots=snapshots,
            )
        )
    total = sum(e["wire"] for e in events)
    assert total * 8 <= now * rate + 1e-5
    return dict(
        group=run["group"],
        rate_bps=rate,
        events=events,
        states=detail_states,
        base_frame_delay_p95_ms=float(np.percentile(frame_delays, 95)),
        cancelled_queued_detail_wire_bytes=cancelled,
        wire_bytes=dict(
            base=sum(e["wire"] for e in events if e["kind"] == "base"),
            detail=sum(e["wire"] for e in events if e["kind"] == "detail"),
        ),
        scope="Deterministic unbounded-queue serializer; no loss/jitter/retransmission/decoding/display, not a live measurement. Frame PTS used as packet availability, excludes encoder scheduling. Codec configuration/control overhead not included.",
    )


def rgb420(planar):
    frame = av.VideoFrame.from_ndarray(planar, format="yuv420p")
    frame.colorspace = 1
    frame.color_range = 1
    return frame.to_ndarray(format="rgb24")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    a = p.parse_args()
    root = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    source = root / "detail-tiles-v1"
    out = root / "detail-replay-v1"
    out.mkdir(exist_ok=False)
    data = json.loads((source / "RESULTS.json").read_text())
    assert data["complete"] and len(data["images"]) == 32
    assert data["codec_script_sha256"] == sha(
        Path(__file__).with_name("detail_transport.py")
    )
    precodec = json.loads((root / "precodec-v1/MANIFEST.json").read_text())
    assert data["controls_sha256"] == sha(root / "precodec-v1/MANIFEST.json")
    selected = json.loads((root / "TRAINING-V2-SELECTED-IMAGES.json").read_text())
    assert data["source_sha256"] == selected["source_sha256"]
    codec_path = (
        a.project
        / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/codec-v1/MANIFEST.json"
    )
    codec = json.loads(codec_path.read_text())
    images = []
    measurements = []
    for row in data["images"]:
        reference = next(
            v
            for v in precodec["images"]
            if v["state_id"] == row["state_id"] and v["height"] == 1080
        )
        path = root / "precodec-v1" / reference["path"]
        assert sha(path) == row["raw_sha256"]
        raw = path.read_bytes()
        target = rgb420(planar_nv12(raw, 1920, 1080))
        base = next(v for v in selected["images"] if v["state_id"] == row["state_id"])
        base_path = Path(base["path"])
        assert sha(base_path) == base["sha256"]
        original = np.asarray(Image.open(base_path).convert("RGB"))
        receiver = Receiver(1920, 1080)
        receiver.begin(row["state_id"], bytes.fromhex(row["identity"]))
        cumulative = wire_bytes(32)
        tile_index = 0
        budgets = []
        # 15KB is already all 600Kbps for 200ms, before base video or any waits.
        for budget in data["parameters"]["quality_budgets_wire_bytes"] + [
            row["wire_bytes"]
        ]:
            begin = time.perf_counter()
            while (
                tile_index < len(row["tiles"])
                and cumulative + row["tiles"][tile_index]["wire_bytes"] <= budget
            ):
                tile = row["tiles"][tile_index]
                path = source / tile["path"]
                assert sha(path) == tile["sha256"]
                assert receiver.accept(path.read_bytes())
                cumulative += tile["wire_bytes"]
                tile_index += 1
            decoded = rgb420(receiver.planar())
            assert np.array_equal(decoded[receiver.mask], target[receiver.mask]), (
                "Partial tile color conversion mismatch"
            )
            output = original.copy()
            output[receiver.mask] = decoded[receiver.mask]
            elapsed = (time.perf_counter() - begin) * 1000
            budgets.append(
                dict(
                    budget=budget,
                    used_wire_bytes=cumulative,
                    tiles=tile_index,
                    pixels=int(receiver.mask.sum()),
                    replay_decode_composite_ms=elapsed,
                    host="H100 CPU, includes hash/read and PNG input excluded; not client timing",
                )
            )
            if budget == row["wire_bytes"]:
                assert receiver.mask.all() and np.array_equal(output, target)
                continue  # Full native420 already has an independently frozen OCR score.
            method = f"source-detail-{budget}bytes"
            name = f"s{row['state_id']:02d}-{method}.png"
            dest = out / name
            Image.fromarray(output).save(dest)
            images.append(
                dict(
                    id=name[:-4],
                    path=str(dest.resolve()),
                    sha256=sha(dest),
                    state_id=row["state_id"],
                    group=row["group"],
                    method=method,
                    config="720p-600k-plus-detail-budget",
                    source_age_s=None,
                )
            )
        measurements.append(
            dict(state_id=row["state_id"], budgets=budgets, exact_all_420_planes=True)
        )
    save(out / "IMAGES.json", dict(source_sha256=data["source_sha256"], images=images))
    links = []
    for run in codec["runs"]:
        if run["height"] != 720:
            continue
        frames = [
            next(v for v in data["images"] if v["state_id"] == sid)
            for sid in run["state_ids"]
        ]
        links.append(shared_link(run, frames))
    record = dict(
        complete=True,
        scope=__doc__,
        sender_sha256=sha(source / "RESULTS.json"),
        selected_base_sha256=sha(root / "TRAINING-V2-SELECTED-IMAGES.json"),
        codec_sha256=sha(codec_path),
        script_sha256=sha(Path(__file__)),
        source_sha256=data["source_sha256"],
        test_used=False,
        measurements=measurements,
        link_replays=links,
        quality_images=len(images),
        limits=[
            "No live network, new display timing or human preference claim",
            "Quality images combine frozen steady SR with a byte budget, not frame-time-matched dynamic playback",
            "Shared simulation includes the actual encoded base packet sizes; controls, codec setup and ACKs excluded",
            "Encoder-only identical-tile skip is measured; receiver cache restoration is not implemented",
        ],
    )
    save(out / "RESULTS.json", record)
    print(
        "Complete:",
        len(images),
        "quality images;",
        len(links),
        "shared-link simulations",
        flush=True,
    )


if __name__ == "__main__":
    main()
