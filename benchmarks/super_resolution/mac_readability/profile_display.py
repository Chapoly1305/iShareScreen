"""Serial before/after RTX4090 display readback with an actual trained EDSR.

Measures tile upload, live scheduling, inference, GPU drawing and synchronized
offscreen readback. No network, hardware decode, panel scanout or user-input
latency claim. The readback synchronization itself is included in timings.
"""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
import wgpu

from isharescreen.enhancement import live, models
from isharescreen.frontend.desktop.gpu import Renderer
from isharescreen.proxy.media.tiles import TileFrame


def exclusive():
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    other = [
        int(s)
        for s in rows.splitlines()
        if s.strip().isdigit() and int(s) != os.getpid()
    ]
    if other:
        raise RuntimeError(f"Other GPU compute processes: {other}")


def tiles(rgb):
    a = rgb.astype(np.float32)
    y = 0.2126 * a[:, :, 0] + 0.7152 * a[:, :, 1] + 0.0722 * a[:, :, 2]
    u = (a[:, :, 2] - y) / 1.8556 + 127.5
    v = (a[:, :, 0] - y) / 1.5748 + 127.5
    pack = lambda p: np.clip(np.rint(p), 0, 255).astype(np.uint8)
    yp, up, vp = (
        pack(y),
        pack(cv2.resize(u, (640, 360), interpolation=cv2.INTER_AREA)),
        pack(cv2.resize(v, (640, 360), interpolation=cv2.INTER_AREA)),
    )
    return [
        TileFrame(
            yp[i * 180 : (i + 1) * 180].tobytes(),
            up[i * 90 : (i + 1) * 90].tobytes(),
            vp[i * 90 : (i + 1) * 90].tobytes(),
            1280,
            180,
            1280,
            640,
            640,
            90,
        )
        for i in range(4)
    ]


def stats(values):
    return dict(
        n=len(values),
        p50_ms=float(np.median(values)),
        p95_ms=float(np.percentile(values, 95)),
        samples_ms=values,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    args = p.parse_args()
    output = args.root / "DISPLAY-4090.json"
    if output.exists():
        raise FileExistsError(output)
    device_info = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    assert "GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a" in device_info
    exclusive()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    manifest = json.loads((args.inputs / "PROFILE-INPUTS.json").read_text())
    frames = []
    for row in manifest["rows"]:
        path = args.inputs / row["path"]
        assert models.sha256(path) == row["sha256"]
        frames.append(tiles(np.asarray(Image.open(path).convert("RGB"))))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    done = json.loads(args.checkpoint.with_name("COMPLETED.json").read_text())
    assert done["best_sha256"] == models.sha256(args.checkpoint)
    adapter = models.Upscaler("edsr", device="cuda:0", precision="amp-fp16")
    torch.nn.Module.load_state_dict(adapter.model, ckpt["state_dict"], strict=True)
    adapter.model.eval().requires_grad_(False)
    spec = importlib.util.spec_from_file_location(
        "isharescreen.enhancement.live_before", args.root / "evidence/live_before.py"
    )
    before = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = before
    spec.loader.exec_module(before)
    gpu = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    device = gpu.request_device_sync()
    info = dict(gpu.info)
    if "4090" not in str(info):
        raise RuntimeError(f"Wrong render adapter: {info}")
    target = device.create_texture(
        size=(1920, 1080, 1),
        format="rgba8unorm",
        usage=wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
    )

    def read(renderer):
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

    result = dict(
        schema="mac_readability_live_display_gpu_v2",
        host=socket.gethostname(),
        gpu=device_info,
        render_adapter=info,
        checkpoint_sha256=models.sha256(args.checkpoint),
        checkpoint_epoch=ckpt["epoch"],
        precision="amp-fp16",
        scope=__doc__,
        started=time.time(),
        script_sha256=models.sha256(Path(__file__)),
        before_sha256=models.sha256(args.root / "evidence/live_before.py"),
        after_sha256=models.sha256(Path(live.__file__)),
        warmup_changes=4,
        measured_changes=24,
        records=[],
        complete=False,
    )
    outputs = {}
    for label, module in [("before", before), ("after", live)]:
        exclusive()
        renderer = Renderer(device, "rgba8unorm", 1280, 720)
        native = Renderer(device, "rgba8unorm", 1280, 720)
        worker = module.EnhancementWorker(
            device="cuda:0", factory=lambda *a, **k: adapter
        )
        display = module.EnhancementDisplay(worker, renderer, 1280, 720, 4)
        worker.select("edsr")
        rows = []

        def update(source):
            for i, tile in enumerate(source):
                renderer.upload_tile(i, tile, 180)
                display.update_tile(i, tile, 180)
            display.step()

        def settle(processed):
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                display.step()
                s = worker.snapshot()
                if s["state"] == "error":
                    raise RuntimeError(s["error"])
                if s["processed"] > processed and s["displayed"] == "edsr":
                    return read(renderer)
                time.sleep(0.0005)
            raise RuntimeError("Refinement timeout")

        try:
            for index in range(28):
                a = frames[(2 * index) % len(frames)]
                b = frames[(2 * index + 1) % len(frames)]
                old_count = worker.snapshot()["processed"]
                update(a)
                settle(old_count)
                start = time.perf_counter()
                for i, tile in enumerate(b):
                    native.upload_tile(i, tile, 180)
                expected = read(native)
                native_ms = (time.perf_counter() - start) * 1000
                old_count = worker.snapshot()["processed"]
                start = time.perf_counter()
                update(b)
                immediate = read(renderer)
                first_ms = (time.perf_counter() - start) * 1000
                immediate_enhanced = renderer._enhanced
                enhanced = settle(old_count)
                refined_ms = (time.perf_counter() - start) * 1000
                if index >= 4:
                    rows.append(
                        dict(
                            index=index - 4,
                            native_ms=native_ms,
                            first_draw_ms=first_ms,
                            added_first_draw_ms=first_ms - native_ms,
                            stable_to_refined_ms=refined_ms,
                            immediate_native_exact=bool(
                                np.array_equal(immediate, expected)
                            ),
                            immediate_used_enhanced=immediate_enhanced,
                            worker=worker.snapshot()["timing_ms"],
                        )
                    )
                    if index == 4:
                        outputs[label] = enhanced
                        Image.fromarray(immediate[:, :, :3]).save(
                            args.root / f"{label}-first-update.png"
                        )
                        Image.fromarray(enhanced[:, :, :3]).save(
                            args.root / f"{label}-settled.png"
                        )
            result["records"].append(
                dict(
                    variant=label,
                    rows=rows,
                    first_draw=stats([r["first_draw_ms"] for r in rows]),
                    added_first_draw=stats([r["added_first_draw_ms"] for r in rows]),
                    stable_to_refined=stats([r["stable_to_refined_ms"] for r in rows]),
                    fresh_native_frames=sum(r["immediate_native_exact"] for r in rows),
                    immediate_enhanced_frames=sum(
                        r["immediate_used_enhanced"] for r in rows
                    ),
                )
            )
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                label,
                "fresh",
                result["records"][-1]["fresh_native_frames"],
                "/24",
                flush=True,
            )
        finally:
            worker.close()
    delta = np.abs(
        outputs["before"].astype(np.int16) - outputs["after"].astype(np.int16)
    )
    result.update(
        complete=True,
        finished=time.time(),
        one_settled_frame_max_delta=int(delta.max()),
        one_settled_frame_mean_delta=float(delta.mean()),
    )
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
