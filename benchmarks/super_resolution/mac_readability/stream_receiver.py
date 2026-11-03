"""Actual RTX4090 replay receiver: HEVC decode, asynchronous SR, exact supplement.

Uses software HEVC decoding and GPU inference/drawing/readback. No NVDEC or
physical-panel claim. Frozen recordings are replayed; this is not a live Mac
desktop capture. stdout is reserved for timestamped protocol acknowledgments.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import queue
import socket
import struct
import subprocess
import sys
import threading
import time

import av
import cv2
import numpy as np
from PIL import Image
import torch
import wgpu

from isharescreen.enhancement import models
from isharescreen.frontend.desktop.gpu import Renderer
from benchmarks.super_resolution.mac_readability.component_receiver import (
    ComponentReceiver,
)
from benchmarks.super_resolution.mac_readability.component_display_probe import (
    rgb_nv12,
    sha,
    save,
)
from benchmarks.super_resolution.mac_readability.stream_protocol import Reader, KINDS
from benchmarks.super_resolution.mac_readability.stream_display_state import (
    DisplayState,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["bypass", "enhanced"], required=True)
    p.add_argument("--policy", choices=["pixel", "hold", "source"], default="pixel")
    p.add_argument("--source-weight", type=float, choices=[0.25, 1.0], default=1.0)
    p.add_argument("--refresh-source-mix", action="store_true")
    a = p.parse_args()
    if a.refresh_source_mix:
        assert a.source_weight < 1 and a.mode == "enhanced" and a.policy == "hold"
    a.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    assert "GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a" in gpu
    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    assert not any(
        int(x) != os.getpid() for x in occupied.splitlines() if x.strip().isdigit()
    ), "GPU occupied"
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = models.Upscaler("edsr", device="cuda:0", precision="amp-fp16")
    checkpoint = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102/selected-v2/checkpoint-001000.pt"
    )
    assert (
        sha(checkpoint)
        == "01d99513d182f669e1e1e92c88e0e17a9a053b6e2d046a8733119047e61db4cd"
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.nn.Module.load_state_dict(model.model, state["state_dict"], strict=True)
    model.model.eval().requires_grad_(False)

    def infer(rgb):
        native, timing = model(rgb)
        return cv2.resize(native, (1920, 1080), interpolation=cv2.INTER_AREA), timing

    inference = ThreadPoolExecutor(max_workers=1)
    inference.submit(infer, np.zeros((720, 1280, 3), np.uint8)).result()
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    assert "4090" in str(adapter.info)
    device = adapter.request_device_sync()
    renderer = Renderer(device, "rgba8unorm", 960, 540)
    target = device.create_texture(
        size=(1920, 1080, 1),
        format="rgba8unorm",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )

    def show(rgb):
        renderer.upload_enhanced(rgb)
        renderer.draw(target.create_view(), 1920, 1080)
        return (
            np.frombuffer(
                device.queue.read_texture(
                    {"texture": target}, {"bytes_per_row": 7680}, (1920, 1080, 1)
                ),
                np.uint8,
            )
            .reshape(1080, 1920, 4)
            .copy()
        )

    blank = np.zeros((1080, 1920, 3), np.uint8)
    for _ in range(4):
        show(blank)
    gate = ComponentReceiver()
    reader = Reader()
    incoming = queue.Queue()
    writer = ThreadPoolExecutor(max_workers=1)
    ack_bytes = 0
    result = dict(
        complete=False,
        scope=__doc__,
        mode=a.mode,
        policy=a.policy,
        source_weight=a.source_weight,
        host=socket.gethostname(),
        gpu=gpu,
        adapter=dict(adapter.info),
        checkpoint_sha256=sha(checkpoint),
        av_version=av.__version__,
        script_sha256=sha(Path(__file__)),
        records=[],
        source_events=[],
        supplements=[],
        sr_events=[],
        mix_refresh_events=[],
        refresh_source_mix=a.refresh_source_mix,
        errors=[],
        stale_video_frames=0,
        stale_sr_results=0,
        decoded_frames=0,
        sr_rejections={},
        temporal_samples=[],
        timed_images=[],
        sample_interval_ms=100,
        timed_snapshot_ages_s=[0.5, 1.0, 2.0, 2.8],
        display_state_sha256=sha(Path(__file__).with_name("stream_display_state.py")),
        wire_scope="Input/output application bytes exact; outer SSH/TCP/VPN bytes are a separate sender OS counter",
        test_used=False,
    )

    def emit(event, **fields):
        nonlocal ack_bytes
        record = dict(event=event, **fields, send_ns=time.monotonic_ns())
        line = json.dumps(record, separators=(",", ":")) + "\n"
        ack_bytes += len(line.encode())
        sys.stdout.write(line)
        sys.stdout.flush()

    def receive():
        try:
            while True:
                item = reader.read(sys.stdin.buffer)
                if item:
                    kind, generation, identity, payload = item
                    if kind == KINDS["source"]:
                        reader.discard_old_detail(generation)
                    incoming.put(
                        (time.monotonic_ns(), kind, generation, identity, payload)
                    )
                    if kind == KINDS["bye"]:
                        return
        except BaseException as exc:
            incoming.put(("error", repr(exc)))

    threading.Thread(target=receive, daemon=True).start()
    emit("ready", mode=a.mode, host=socket.gethostname(), gpu=gpu)
    decoders = {}
    generation = 0
    sid = None
    revision = 0
    latest = None
    latest_hash = None
    detail = None
    source_ns = None
    first_seen = set()
    sr_seen = set()
    pending = None
    submitted = None
    last_change_ns = None
    ownership = DisplayState(a.policy)
    visible = blank
    generation_rx = None
    sample_at = None
    previous_sample = None
    sample_phase = None
    snapshots = set()
    cached_sr = None
    cached_sr_record = None
    finish_at = None
    done = False
    last_save = time.monotonic()
    try:
        while True:
            try:
                item = incoming.get(timeout=0.001)
            except queue.Empty:
                item = None
            if item and item[0] == "error":
                raise RuntimeError(item[1])
            if item:
                rx, kind, gen, identity, payload = item
                if kind == KINDS["bye"]:
                    break
                if kind == KINDS["ping"]:
                    emit(
                        "pong",
                        id=identity,
                        sender_ns=struct.unpack(">Q", payload)[0],
                        recv_ns=rx,
                    )
                elif kind == KINDS["config"]:
                    data = json.loads(payload)
                    decoder = av.CodecContext.create("hevc", "r")
                    decoder.thread_count = 1
                    decoder.extradata = base64.b64decode(data["extradata"])
                    decoder.open()
                    decoders[data["clip"]] = decoder
                elif kind == KINDS["source"]:
                    data = json.loads(payload)
                    assert gen > generation
                    generation = gen
                    sid = data["state_id"]
                    source_ns = data["source_ns"]
                    revision += 1
                    latest = latest_hash = detail = submitted = None
                    last_change_ns = None
                    cached_sr = cached_sr_record = None
                    ownership.begin(gen)
                    gate.begin(gen, bytes.fromhex(data["raw_sha256"]))
                    show(blank)
                    visible = blank
                    generation_rx = rx
                    sample_at = rx + 100000000
                    previous_sample = None
                    sample_phase = None
                    snapshots = set()
                    result["source_events"].append(
                        dict(
                            generation=gen,
                            state_id=sid,
                            source_ns=source_ns,
                            recv_ns=rx,
                            handled_ns=time.monotonic_ns(),
                        )
                    )
                    emit(
                        "source",
                        generation=gen,
                        state_id=sid,
                        source_ns=source_ns,
                        recv_ns=rx,
                    )
                elif kind == KINDS["video"]:
                    decoder = decoders[(gen - 1) // 2]
                    begin = time.monotonic_ns()
                    pts = struct.unpack(">q", payload[:8])[0]
                    packet = av.Packet(payload[8:])
                    packet.pts = packet.dts = pts
                    packet.time_base = Fraction(1, 1000000)
                    frames = decoder.decode(packet)
                    for frame in frames:
                        result["decoded_frames"] += 1
                        assert (
                            frame.width == 1280
                            and frame.height == 720
                            and frame.format.name == "yuv420p"
                        )
                        if gen != generation:
                            result["stale_video_frames"] += 1
                            continue
                        rgb = frame.to_ndarray(format="rgb24")
                        pixel_hash = hashlib.sha256(rgb.tobytes()).hexdigest()
                        changed = pixel_hash != latest_hash
                        if changed:
                            revision += 1
                            latest = rgb
                            latest_hash = pixel_hash
                            last_change_ns = rx
                        record = dict(
                            generation=gen,
                            state_id=sid,
                            pts_us=pts,
                            received_ns=rx,
                            decode_started_ns=begin,
                            decoded_ns=time.monotonic_ns(),
                            pixel_sha256=pixel_hash,
                            codec_revision=revision,
                            changed=changed,
                        )
                        if (
                            detail is None
                            and (changed or gen not in first_seen)
                            and ownership.base(gen)
                        ):
                            native = cv2.resize(
                                rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC
                            )
                            pixels = show(native)
                            record["present_ns"] = time.monotonic_ns()
                            visible = pixels[:, :, :3]
                            record["receive_to_base_readback_ms"] = (
                                record["present_ns"] - rx
                            ) / 1e6
                            if gen not in first_seen:
                                first_seen.add(gen)
                                writer.submit(
                                    Image.fromarray(pixels[:, :, :3]).save,
                                    a.output / f"s{sid:02d}-first.png",
                                )
                                emit(
                                    "first",
                                    generation=gen,
                                    state_id=sid,
                                    source_ns=source_ns,
                                    recv_ns=rx,
                                    present_ns=record["present_ns"],
                                    local_ms=record["receive_to_base_readback_ms"],
                                )
                        if pts % 3000000 == 2800000:
                            writer.submit(
                                Image.fromarray(rgb).save,
                                a.output / f"s{sid:02d}-steady-decoded.png",
                            )
                        result["records"].append(record)
                elif kind == KINDS["detail"]:
                    if gen != generation:
                        result["supplements"].append(dict(generation=gen, stale=True))
                        continue
                    if a.mode == "bypass":
                        emit(
                            "detail_ignored",
                            generation=gen,
                            state_id=sid,
                            source_ns=source_ns,
                            recv_ns=rx,
                        )
                    else:
                        start = time.monotonic_ns()
                        raw = gate.accept(gen, payload)
                        assert raw is not None
                        assert ownership.detail(gen)
                        detail = rgb_nv12(raw)
                        effective_weight = 1.0
                        composed = detail
                        if a.source_weight < 1 and cached_sr is not None:
                            assert cached_sr_record["generation"] == gen
                            effective_weight = a.source_weight
                            composed = cv2.addWeighted(
                                detail,
                                effective_weight,
                                cached_sr,
                                1 - effective_weight,
                                0,
                            )
                        pixels = show(composed)
                        present = time.monotonic_ns()
                        visible = pixels[:, :, :3]
                        delta = np.abs(
                            pixels[:, :, :3].astype(np.int16)
                            - composed.astype(np.int16)
                        )
                        assert delta.max() <= 1
                        record = dict(
                            generation=gen,
                            state_id=sid,
                            source_ns=source_ns,
                            recv_ns=rx,
                            present_ns=present,
                            processing_ms=(present - start) / 1e6,
                            receive_to_readback_ms=(present - rx) / 1e6,
                            render_max_delta=int(delta.max()),
                            raw_sha256=hashlib.sha256(raw).hexdigest(),
                            source_weight_requested=a.source_weight,
                            source_weight_actual=effective_weight,
                            sr_base_record=cached_sr_record
                            if effective_weight < 1
                            else None,
                            source_rgb_sha256=hashlib.sha256(
                                detail.tobytes()
                            ).hexdigest(),
                            sr_base_rgb_sha256=hashlib.sha256(
                                cached_sr.tobytes()
                            ).hexdigest()
                            if effective_weight < 1
                            else None,
                        )
                        if effective_weight < 1:
                            writer.submit(
                                Image.fromarray(cached_sr).save,
                                a.output / f"s{sid:02d}-refinement-sr-base.png",
                            )
                            writer.submit(
                                Image.fromarray(detail).save,
                                a.output / f"s{sid:02d}-refinement-source.png",
                            )
                        result["supplements"].append(record)
                        writer.submit(
                            Image.fromarray(pixels[:, :, :3]).save,
                            a.output / f"s{sid:02d}-detail.png",
                        )
                        emit("detail", **record)
                elif kind == KINDS["finish"]:
                    finish_at = time.monotonic() + 1
                else:
                    raise ValueError(f"Unexpected kind{kind}")
            if pending is not None and pending.done():
                owner_gen, owner_revision, owner_sid, origin, change_ns, future = (
                    pending.result_meta
                )
                sr, timing = pending.result()
                pending = None
                source_backed = a.refresh_source_mix and detail is not None
                rejection = ownership.sr_rejection(
                    owner_gen, owner_revision, revision, source_backed
                )
                if rejection is None:
                    assert ownership.sr(
                        owner_gen, owner_revision, revision, source_backed
                    )
                    composed = (
                        cv2.addWeighted(
                            detail, a.source_weight, sr, 1 - a.source_weight, 0
                        )
                        if source_backed
                        else sr
                    )
                    pixels = show(composed)
                    present = time.monotonic_ns()
                    visible = pixels[:, :, :3]
                    record = dict(
                        generation=generation,
                        state_id=sid,
                        revision=revision,
                        present_ns=present,
                        source_ns=source_ns,
                        input_revision=owner_revision,
                        latest_codec_revision=revision,
                        input_received_to_sr_ms=(present - change_ns) / 1e6,
                        stable_decoded_to_sr_ms=(present - change_ns) / 1e6,
                        model_timing=timing,
                        timing_note="Job input receipt to SR readback; source policy may accept an earlier codec revision of the same announced source",
                    )
                    cached_sr = sr
                    cached_sr_record = record.copy()
                    if source_backed:
                        ordinal = len(result["mix_refresh_events"])
                        stem = f"s{sid:02d}-mix-refresh{ordinal:03d}"
                        source_record = next(
                            x
                            for x in reversed(result["supplements"])
                            if x["generation"] == generation
                        )
                        record.update(
                            source_weight_actual=a.source_weight,
                            raw_sha256=source_record["raw_sha256"],
                            source_rgb_sha256=hashlib.sha256(
                                detail.tobytes()
                            ).hexdigest(),
                            sr_base_rgb_sha256=hashlib.sha256(sr.tobytes()).hexdigest(),
                            path=stem + ".png",
                            source_path=stem + "-source.png",
                            sr_base_path=stem + "-sr.png",
                            render_max_delta=int(
                                np.abs(
                                    visible.astype(np.int16) - composed.astype(np.int16)
                                ).max()
                            ),
                        )
                        assert record["render_max_delta"] <= 1
                        for image, name in [
                            (visible, record["path"]),
                            (detail, record["source_path"]),
                            (sr, record["sr_base_path"]),
                        ]:
                            writer.submit(Image.fromarray(image).save, a.output / name)
                        result["mix_refresh_events"].append(record)
                        emit("mix_refresh", **record)
                    else:
                        result["sr_events"].append(record)
                    if not source_backed and generation not in sr_seen:
                        sr_seen.add(generation)
                        writer.submit(
                            Image.fromarray(pixels[:, :, :3]).save,
                            a.output / f"s{sid:02d}-sr.png",
                        )
                        emit("sr", **record)
                else:
                    result["stale_sr_results"] += 1
                    result["sr_rejections"][rejection] = (
                        result["sr_rejections"].get(rejection, 0) + 1
                    )
            if (
                not done
                and a.mode == "enhanced"
                and pending is None
                and latest is not None
                and (detail is None or a.refresh_source_mix)
                and submitted != (generation, revision)
            ):
                submitted = (generation, revision)
                pending = inference.submit(infer, latest)
                pending.result_meta = (
                    generation,
                    revision,
                    sid,
                    source_ns,
                    last_change_ns,
                    pending,
                )
            now = time.monotonic_ns()
            if generation_rx is not None and not done:
                age = (now - generation_rx) / 1e9
                if now >= sample_at:
                    sample = visible[::4, ::4, :].copy()
                    row = dict(
                        generation=generation,
                        state_id=sid,
                        age_s=age,
                        sampled_ns=now,
                        phase=ownership.phase,
                    )
                    if previous_sample is not None:
                        delta = np.abs(
                            sample.astype(np.int16) - previous_sample.astype(np.int16)
                        )
                        row.update(
                            previous_phase=sample_phase,
                            mean_abs_rgb_delta=float(delta.mean()),
                            changed_fraction=float(np.any(delta, axis=2).mean()),
                        )
                    result["temporal_samples"].append(row)
                    previous_sample = sample
                    sample_phase = ownership.phase
                    sample_at = now + 100000000
                for target_age in result["timed_snapshot_ages_s"]:
                    if target_age not in snapshots and age >= target_age:
                        name = f"s{sid:02d}-age{round(target_age * 1000):04d}.png"
                        copy = visible.copy()
                        writer.submit(Image.fromarray(copy).save, a.output / name)
                        result["timed_images"].append(
                            dict(
                                generation=generation,
                                state_id=sid,
                                age_target_s=target_age,
                                age_actual_s=age,
                                phase=ownership.phase,
                                path=name,
                                pixel_sha256=hashlib.sha256(copy.tobytes()).hexdigest(),
                                shape=list(copy.shape),
                            )
                        )
                        snapshots.add(target_age)
            if (
                finish_at is not None
                and time.monotonic() >= finish_at
                and pending is None
                and not done
            ):
                writer.shutdown(wait=True)
                result.update(
                    complete=True,
                    received_application_bytes=reader.bytes,
                    ack_application_bytes=ack_bytes,
                )
                save(a.output / "RESULTS.json", result)
                emit(
                    "done",
                    received_bytes=reader.bytes,
                    decoded_frames=result["decoded_frames"],
                    output=str(a.output),
                )
                done = True
            if time.monotonic() - last_save > 2 and not done:
                save(
                    a.output / "PROGRESS.json",
                    dict(
                        decoded=result["decoded_frames"],
                        source=sid,
                        supplements=len(result["supplements"]),
                        sr=len(result["sr_events"]),
                        queue=incoming.qsize(),
                    ),
                )
                last_save = time.monotonic()
    except BaseException as exc:
        result["errors"].append(repr(exc))
        save(a.output / "FAILED.json", result)
        raise
    finally:
        inference.shutdown(wait=True)
        if not done:
            writer.shutdown(wait=True)


if __name__ == "__main__":
    main()
