"""RTX4090 local supplement decode/upload/GPU-readback timing.

Base SR images are precomputed. This isolates supplemental processing; it is
NOT full base decode/inference, source-to-display or physical-panel latency.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import av
import cv2
import numpy as np
from PIL import Image
import wgpu

from isharescreen.frontend.desktop.gpu import Renderer
from benchmarks.super_resolution.mac_readability.component_receiver import (
    ComponentReceiver,
)
from benchmarks.super_resolution.mac_readability.detail_transport import planes_nv12


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(path)


def rgb_nv12(raw):
    planar = np.concatenate([p.ravel() for p in planes_nv12(raw, 1920, 1080)]).reshape(
        1620, 1920
    )
    frame = av.VideoFrame.from_ndarray(planar, format="yuv420p")
    frame.colorspace = 1
    frame.color_range = 1
    return frame.to_ndarray(format="rgb24")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    a = p.parse_args()
    cv2.setNumThreads(1)
    root = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    out = root / "component-display-v2"
    out.mkdir(exist_ok=False)
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
        int(s) != os.getpid() for s in occupied.splitlines() if s.strip().isdigit()
    ), "GPU occupied"
    source = root / "component-mac-v1"
    sender = json.loads((source / "RESULTS.json").read_text())
    assert sender["complete"]
    base = json.loads((root / "TRAINING-V2-SELECTED-IMAGES.json").read_text())
    items = []
    for row in sender["images"]:
        path = source / row["path"]
        assert sha(path) == row["sha256"]
        packet = path.read_bytes()
        b = next(x for x in base["images"] if x["state_id"] == row["state_id"])
        path = (
            root
            / "training-v2/models/steady_l1/validation-001000"
            / Path(b["path"]).name
        )
        assert sha(path) == b["sha256"]
        items.append((row, packet, np.asarray(Image.open(path).convert("RGB"))))
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

    gate = ComponentReceiver()
    records = []
    result = dict(
        complete=False,
        scope=__doc__,
        host=socket.gethostname(),
        gpu=gpu,
        adapter=dict(adapter.info),
        sender_sha256=sha(source / "RESULTS.json"),
        scripts={
            name: sha(Path(__file__).with_name(name))
            for name in [
                "component_display_probe.py",
                "component_receiver.py",
                "component_codec.py",
            ]
        },
        renderer_sha256=sha(
            Path(
                __import__(
                    "isharescreen.frontend.desktop.gpu", fromlist=["Renderer"]
                ).__file__
            )
        ),
        records=records,
        warmup=4,
        test_used=False,
        control_order="Counterbalanced control-first/treatment-first; each starts from the same previous source pixels",
    )
    save(out / "RESULTS.json", result)
    previous_rgb = np.zeros((1080, 1920, 3), np.uint8)
    for index, (row, packet, base_rgb) in enumerate(items[:4] + items):
        old = index
        for stage in ["control", "fresh"] if index % 2 == 0 else ["fresh", "control"]:
            show(previous_rgb)
            start = time.perf_counter()
            if stage == "fresh":
                gate.begin(index + 1, bytes.fromhex(row["raw_sha256"]))
                assert gate.latest is None
                first = show(base_rgb)
                first_ms = (time.perf_counter() - start) * 1000
            else:
                expected_base = show(base_rgb)
                baseline_ms = (time.perf_counter() - start) * 1000
        assert np.array_equal(first, expected_base)
        assert gate.accept(old, packet) is None
        start = time.perf_counter()
        raw = gate.accept(index + 1, packet)
        decode_ms = (time.perf_counter() - start) * 1000
        assert raw is not None and hashlib.sha256(raw).hexdigest() == row["raw_sha256"]
        convert = time.perf_counter()
        rgb = rgb_nv12(raw)
        convert_ms = (time.perf_counter() - convert) * 1000
        draw = time.perf_counter()
        refined = show(rgb)
        draw_ms = (time.perf_counter() - draw) * 1000
        total_ms = (time.perf_counter() - start) * 1000
        delta = np.abs(refined[:, :, :3].astype(np.int16) - rgb.astype(np.int16))
        assert delta.max() <= 1
        previous_rgb = rgb
        if index >= 4:
            records.append(
                dict(
                    state_id=row["state_id"],
                    control_first=index % 2 == 0,
                    first_draw_ms=first_ms,
                    baseline_draw_ms=baseline_ms,
                    added_first_ms=first_ms - baseline_ms,
                    decode_ms=decode_ms,
                    convert_ms=convert_ms,
                    upload_draw_readback_ms=draw_ms,
                    supplement_available_to_readback_ms=total_ms,
                    first_base_exact=True,
                    stale_rejected=True,
                    render_max_delta=int(delta.max()),
                )
            )
            save(out / "RESULTS.json", result)
    result.update(
        complete=True,
        stats={
            key: dict(
                p50=float(np.median([x[key] for x in records])),
                p95=float(np.percentile([x[key] for x in records], 95)),
            )
            for key in [
                "first_draw_ms",
                "added_first_ms",
                "decode_ms",
                "convert_ms",
                "upload_draw_readback_ms",
                "supplement_available_to_readback_ms",
            ]
        },
    )
    save(out / "RESULTS.json", result)
    print(result["stats"], flush=True)


if __name__ == "__main__":
    main()
