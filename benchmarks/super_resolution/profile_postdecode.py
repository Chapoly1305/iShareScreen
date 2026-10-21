"""Time decoded YUV tiles through enhancement and completed Metal rendering.

This is a serial processing-budget probe, not network-to-screen latency or a
measurement of the viewer's display pacing. Decode and fixture preparation are
outside the timed interval. GPU completion is included, not just upload enqueue.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from benchmarks.super_resolution.training_state import atomic_json
from benchmarks.super_resolution.video_latency import sample_indices
from isharescreen.enhancement import variants
from isharescreen.enhancement.live import CanvasSnapshot, compose
from isharescreen.enhancement.models import sha256
from isharescreen.proxy.media.decode_common import _av_frame_to_tile
from isharescreen.proxy.media.tiles import TileFrame


def fixtures(path, samples, num_tiles):
    """Retain actual decoded 4:4:4 planes, splitting them into padded tile slots."""
    import av

    # Count without retaining full clips: long RGB/YUV recordings can otherwise
    # consume several GiB before a small sample set is selected.
    with av.open(str(path)) as container:
        decoded_count = sum(1 for _ in container.decode(video=0))
    chosen = sample_indices(decoded_count, samples)
    selected = set(chosen)
    decoded = []
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index not in selected:
                continue
            if (frame.width, frame.height) != (960, 540):
                raise ValueError("Use fixed 960x540 video, without input resizing")
            if frame.format.name not in ("yuv444p", "yuvj444p"):
                raise ValueError("Use decoded planar YUV 4:4:4 video for this probe")
            if frame.colorspace != 1 or frame.color_range != 2:
                raise ValueError("Use full-range BT.709 video matching the live shader")
            tile, error = _av_frame_to_tile(frame, [None], set())
            if tile is None or error:
                raise ValueError("Invalid decoded fixture")
            decoded.append((index, tile))
    snapshots, identities = [], []
    slot = math.ceil(math.ceil(540 / num_tiles) / 16) * 16
    for index, tile in decoded:
        tiles = []
        for i in range(num_tiles):
            start, end = i * slot, min((i + 1) * slot, 540)
            if start >= end:
                raise ValueError("Too many tile slots for the canvas")
            planes = []
            for value, stride in ((tile.y, tile.y_stride), (tile.u, tile.uv_stride),
                                  (tile.v, tile.uv_stride)):
                source = np.frombuffer(value, np.uint8).reshape(540, stride)
                padded = np.zeros((slot, stride), np.uint8)
                padded[:end-start] = source[start:end]
                planes.append(padded.tobytes())
            tiles.append((TileFrame(*planes, 960, slot, tile.y_stride,
                                    tile.uv_stride, 960, slot), slot))
        snapshot = CanvasSnapshot(960, 540, tuple(tiles))
        snapshots.append(snapshot)
        identities.append(dict(frame=index, rgb_sha256=hashlib.sha256(compose(snapshot).tobytes()).hexdigest()))
    return snapshots, dict(path=str(path), sha256=sha256(path), decoded_frames=decoded_count,
                          num_tiles=num_tiles, slot_height=slot, frames=identities)


def summarize(rows, budget):
    result = {key: dict(p50_ms=float(np.median([r[key] for r in rows])),
                       p95_ms=float(np.percentile([r[key] for r in rows], 95)),
                       max_ms=float(max(r[key] for r in rows))) for key in rows[0]}
    result["budget"] = dict(threshold_ms=budget,
        p95_pass=result["postdecode_ms"]["p95_ms"] < budget,
        every_sample_pass=all(row["postdecode_ms"] < budget for row in rows),
        over_budget=sum(row["postdecode_ms"] >= budget for row in rows), samples=len(rows))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, nargs="+", required=True)
    parser.add_argument("--models", nargs="+", default=["quicksrnet-torch", "plainusr", "rt4ksr", "efrlfn"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument("--budget-ms", type=float, default=33)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or min(args.samples, args.warmup, args.tiles) < 1:
        parser.error("Use positive counts and a fresh report path")
    import cv2
    import torch
    import wgpu
    from isharescreen.frontend.desktop.gpu import Renderer

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    sources = [Path(__file__), Path("src/isharescreen/enhancement/live.py"),
        Path("src/isharescreen/enhancement/variants.py"),
        Path("src/isharescreen/enhancement/execution.py"),
        Path("src/isharescreen/enhancement/candidates/__init__.py"),
        Path("src/isharescreen/enhancement/mps_output.py"),
        Path("src/isharescreen/enhancement/models.py"),
        Path("src/isharescreen/frontend/desktop/gpu.py")]
    report = dict(schema="sr_postdecode_profile_v1", started_at=time.time(),
        platform=platform.platform(), torch=torch.__version__, wgpu=wgpu.__version__,
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        source_sha256={str(p): sha256(p) for p in sources},
        gpu_only=os.environ.get("ISS_SR_GPU_ONLY"),
        mps_cpu_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        scope="Serial decoded-tile composition, actual adapter, RGB texture upload, draw and GPU queue completion. Excludes decode, network, worker queueing, live GPU contention, display pacing and scanout. Not viewer FPS or input-to-display latency.",
        fixture_scope="Actual decoded full-range BT.709 YUV444 frames split into four CTU-padded slots; not a synchronized multi-stream capture.",
        completion_scope="An ordered 1x1 target-texture readback fences the draw; its diagnostic overhead is included. Full-image pixel verification is outside timing.",
        warmup=args.warmup, records=[])
    prepared = [fixtures(path, args.samples, args.tiles) for path in args.videos]
    gpu = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = gpu.request_device_sync()
    report["renderer_adapter"] = dict(gpu.info)
    renderer = Renderer(device, "rgba8unorm", 960, 540)
    target = device.create_texture(size=(1920, 1080, 1), format="rgba8unorm",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC)
    view = target.create_view()
    for key in args.models:
        adapter = variants.load(key, device=args.device)
        for snapshots, identity in prepared:
            rows = []
            for i in range(args.warmup + len(snapshots)):
                snapshot = snapshots[i % len(snapshots) if i < args.warmup else i-args.warmup]
                started = time.perf_counter()
                rgb = compose(snapshot)
                assembled = time.perf_counter()
                output, stages = adapter(rgb)
                inferred = time.perf_counter()
                renderer.upload_enhanced(output)
                renderer.draw(view, 1920, 1080)
                # An ordered copy from this render target waits for its draw.
                # Use one pixel, avoiding a full-frame diagnostic download.
                device.queue.read_texture({"texture": target},
                    {"bytes_per_row":256}, (1,1,1))
                completed = time.perf_counter()
                if i >= args.warmup:
                    rows.append(dict(assembly_ms=(assembled-started)*1000,
                        adapter_ms=(inferred-assembled)*1000,
                        render_completed_ms=(completed-inferred)*1000,
                        postdecode_ms=(completed-started)*1000, **stages))
            pixels = np.frombuffer(device.queue.read_texture({"texture": target},
                {"bytes_per_row":1920*4}, (1920,1080,1)), np.uint8).reshape(1080,1920,4)
            delta = np.abs(pixels[...,:3].astype(np.int16)-output.astype(np.int16))
            if delta.max() > 1:
                raise ValueError("Rendered pixels differ from the model output")
            row = dict(model=key, provenance=adapter.provenance, video=identity,
                renderer_max_rgb_delta=int(delta.max()), samples=rows,
                summary=summarize(rows, args.budget_ms))
            report["records"].append(row)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(args.output, report)
            print(json.dumps(dict(model=key, video=identity["path"],
                postdecode=row["summary"]["postdecode_ms"],
                budget=row["summary"]["budget"])), flush=True)
        del adapter
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    report.update(complete=True, finished_at=time.time())
    atomic_json(args.output, report)


if __name__ == "__main__":
    main()
