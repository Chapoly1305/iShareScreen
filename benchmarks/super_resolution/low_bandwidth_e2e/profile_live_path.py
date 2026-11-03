"""Replay encoded frames through the live worker at 30 fps on one Linux GPU."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import socket
import time

import cv2
import numpy as np
import torch

from isharescreen.enhancement.live import EnhancementWorker
from isharescreen.enhancement.models import sha256


def run_video(path: Path, width: int, height: int, bps: int) -> dict:
    torch.cuda.reset_peak_memory_stats(0)
    worker = EnhancementWorker(device="cuda:0")
    worker.select("quicksrnet-torch")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video: {path}")
    decode_ms, worker_ms, forward_ms, fhd_ms, age_ms = [], [], [], [], []
    displayed = submitted = 0
    first = time.monotonic()
    frame_limit = 90
    try:
        while submitted < frame_limit:
            due = first + submitted / 30
            if time.monotonic() < due:
                time.sleep(min(0.005, due - time.monotonic()))
                continue
            t0 = time.perf_counter()
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            decode_ms.append((time.perf_counter() - t0) * 1000)
            if rgb.shape != (height, width, 3):
                raise ValueError("Decoded geometry differs from labeled input")
            worker.submit(rgb)
            submitted += 1
            ready = worker.poll()
            if ready is not None:
                t1 = time.perf_counter()
                pixels = (
                    ready.rgb
                    if height == 540
                    else cv2.resize(
                        ready.rgb, (1920, 1080), interpolation=cv2.INTER_LANCZOS4
                    )
                )
                if pixels.shape != (1080, 1920, 3):
                    raise ValueError("FHD presentation geometry differs")
                fhd_ms.append((time.perf_counter() - t1) * 1000)
                worker.displayed(ready.generation, ready.model)
                displayed += 1
                worker_ms.append(ready.timing["assembly_and_processing_ms"])
                forward_ms.append(ready.timing["inference_ms"])
                age_ms.append(ready.timing["age_ms"])
        deadline = time.monotonic() + 2
        while (
            time.monotonic() < deadline
            and worker.snapshot()["processed"]
            < submitted - worker.snapshot()["dropped"]
        ):
            ready = worker.poll()
            if ready is not None:
                t1 = time.perf_counter()
                pixels = (
                    ready.rgb
                    if height == 540
                    else cv2.resize(
                        ready.rgb, (1920, 1080), interpolation=cv2.INTER_LANCZOS4
                    )
                )
                fhd_ms.append((time.perf_counter() - t1) * 1000)
                worker.displayed(ready.generation, ready.model)
                displayed += 1
                worker_ms.append(ready.timing["assembly_and_processing_ms"])
                forward_ms.append(ready.timing["inference_ms"])
                age_ms.append(ready.timing["age_ms"])
            time.sleep(0.002)
        stats = worker.snapshot()
    finally:
        capture.release()
        worker.close()
    if not decode_ms or stats["state"] == "error":
        raise ValueError(f"Live worker failed: {stats}")

    def summary(values: list[float]) -> dict:
        return dict(
            n=len(values),
            median_ms=float(np.median(values)) if values else None,
            p95_ms=float(np.percentile(values, 95)) if values else None,
            maximum_ms=float(max(values)) if values else None,
        )

    return dict(
        input_geometry=[width, height],
        native_output_geometry=[width * 2, height * 2],
        display_geometry=[1920, 1080],
        requested_video_bps=bps,
        encoded_sha256=sha256(path),
        source=str(path),
        submitted=submitted,
        processed=stats["processed"],
        dropped=stats["dropped"],
        presented=displayed,
        presentation_overwrites=stats["processed"] - displayed,
        source_fps=30,
        achieved_presented_fps=displayed / max(time.monotonic() - first, 1e-9),
        decode_and_rgb=summary(decode_ms),
        model_forward=summary(forward_ms),
        worker_assembly_and_model=summary(worker_ms),
        output_to_fhd=summary(fhd_ms),
        queue_age=summary(age_ms),
        max_allocated_bytes=torch.cuda.max_memory_allocated(0),
        max_reserved_bytes=torch.cuda.max_memory_reserved(0),
        model_provenance=stats["provenance"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Profile already exists")
    if not torch.cuda.is_available():
        raise ValueError("CUDA GPU required")
    torch.cuda.init()
    cells = []
    start = time.time()
    for width, height in ((960, 540), (1280, 720), (1920, 1080)):
        for bps in (600000, 1000000):
            path = (
                args.input_root
                / args.case_id
                / f"{width}x{height}-{bps // 1000}k"
                / "encoded.mkv"
            )
            cells.append(run_video(path, width, height, bps))
            print(
                "PROFILE",
                height,
                bps,
                cells[-1]["submitted"],
                cells[-1]["dropped"],
                flush=True,
            )
    result = dict(
        schema="isharescreen_low_bandwidth_live_replay_profile_v1",
        host=socket.gethostname(),
        platform=platform.platform(),
        gpu=torch.cuda.get_device_name(0),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch=str(torch.__version__),
        cuda_runtime=torch.version.cuda,
        precision="fp32",
        source_case=args.case_id,
        cells=cells,
        elapsed_gpu_job_s=time.time() - start,
        finished=time.time(),
        scope="Offline libx265 file decode -> RGB -> actual EnhancementWorker bounded queue and official QuickSRNet -> FHD CPU presentation pixels. No texture upload, Mac renderer, network transport or real remote desktop.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
