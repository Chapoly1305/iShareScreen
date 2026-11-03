"""Render preselected side-by-side HEVC/QuickSRNet videos for inspection only."""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import torch

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


_loader = importlib.util.spec_from_file_location(
    "campaign_evaluate_gpu", Path(__file__).with_name("evaluate_gpu.py")
)
assert _loader is not None and _loader.loader is not None
_module = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(_module)
get_model = _module.get_model


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def render(
    input_path: Path,
    output_path: Path,
    model: torch.nn.Module,
    crop_box: list[int],
    height: int,
) -> dict:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot decode encoded video: {input_path}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        "3840x1080",
        "-r",
        "30",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    prev_base = prev_model = None
    temporal_base, temporal_model, forward_ms = [], [], []
    x, y, w, h = crop_box
    frames = 0
    try:
        while frames < 90:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if rgb.shape[:2] != (height, {540: 960, 720: 1280, 1080: 1920}[height]):
                raise ValueError("Video frame geometry differs from frozen cell")
            baseline = (
                rgb
                if height == 1080
                else cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            )
            tensor = (
                torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None]
                .to("cuda:0")
                .float()
                .div_(255)
            )
            torch.cuda.synchronize(0)
            t0 = time.perf_counter()
            with torch.inference_mode():
                result = model(tensor)
            torch.cuda.synchronize(0)
            if (
                result.shape != (1, 3, height * 2, rgb.shape[1] * 2)
                or not torch.isfinite(result).all()
            ):
                raise ValueError("Presentation model returned invalid pixels")
            forward_ms.append((time.perf_counter() - t0) * 1000)
            enhanced = (
                result[0]
                .float()
                .mul(255)
                .round()
                .clamp(0, 255)
                .byte()
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            enhanced = (
                enhanced
                if height == 540
                else cv2.resize(
                    enhanced, (1920, 1080), interpolation=cv2.INTER_LANCZOS4
                )
            )
            if prev_base is not None:
                temporal_base.append(
                    float(
                        np.abs(
                            baseline[y : y + h, x : x + w].astype(np.int16) - prev_base
                        ).mean()
                    )
                )
                temporal_model.append(
                    float(
                        np.abs(
                            enhanced[y : y + h, x : x + w].astype(np.int16) - prev_model
                        ).mean()
                    )
                )
            prev_base = baseline[y : y + h, x : x + w].copy()
            prev_model = enhanced[y : y + h, x : x + w].copy()
            both = np.concatenate((baseline, enhanced), axis=1)
            assert process.stdin is not None
            process.stdin.write(both.tobytes())
            frames += 1
        assert process.stdin is not None
        process.stdin.close()
        error = process.stderr.read().decode() if process.stderr is not None else ""
        if process.wait() != 0:
            raise RuntimeError("Presentation encoder failed: " + error[-1000:])
    finally:
        capture.release()
        if process.poll() is None:
            process.kill()
            process.wait()
    if frames != 90 or len(temporal_model) != 89:
        raise ValueError("Comparison video lacks the frozen 90 frames")
    return dict(
        frames=frames,
        fps=30,
        input_sha256=sha256(input_path),
        video_sha256=sha256(output_path),
        video_bytes=output_path.stat().st_size,
        video_codec="H.264 CRF20 only for human presentation",
        left="HEVC decoded input at fixed FHD display via bicubic/native",
        right="Validation-selected x2 model at fixed FHD display",
        mean_text_crop_temporal_abs_rgb_base=float(np.mean(temporal_base)),
        mean_text_crop_temporal_abs_rgb_model=float(np.mean(temporal_model)),
        p95_text_crop_temporal_abs_rgb_base=float(np.percentile(temporal_base, 95)),
        p95_text_crop_temporal_abs_rgb_model=float(np.percentile(temporal_model, 95)),
        median_forward_ms=float(np.median(forward_ms)),
        scope="Temporal RGB difference is auxiliary and sensitive to enhancement contrast; inspect video and OCR separately.",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    selection = read(root / "MEDIA_SELECTION.json")
    final = read(root / "FINAL_SELECTION.json")
    protocol_path = root / "EVAL_PROTOCOL.json"
    if selection["eval_protocol_sha256"] != sha256(protocol_path) or selection[
        "selection_sha256"
    ] != sha256(root / "FINAL_SELECTION.json"):
        raise ValueError("Media selection changed from frozen protocol")
    protocol = read(protocol_path)
    case_by_id = {c["id"]: c for c in protocol["cases"] if c["split"] == "test"}
    weights = read(args.weights.resolve())
    if weights["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Model binding differs from frozen test")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "TASK.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.time()
        rows = []
        loaded = {}
        torch.set_num_threads(2)
        for case_id in selection["case_ids"]:
            case = case_by_id[case_id]
            for cell in selection["comparison_cells"]:
                method = cell["method"]
                if method not in final["test_methods"]:
                    raise ValueError("Media method was not test selected")
                if method not in loaded:
                    model, identity = get_model(
                        method, weights["models"], args.home.resolve()
                    )
                    if (
                        identity["checkpoint_sha256"]
                        != final["test_checkpoint_sha256"][method]
                    ):
                        raise ValueError(
                            "Media model differs from validation selection"
                        )
                    loaded[method] = model.to("cuda:0").eval().requires_grad_(False)
                model = loaded[method]
                height = cell["base_height"]
                bps = cell["requested_bps"]
                width = {540: 960, 720: 1280, 1080: 1920}[height]
                source = (
                    root
                    / "evaluation-inputs"
                    / case_id
                    / f"{width}x{height}-{bps // 1000}k"
                    / "encoded.mkv"
                )
                receipt = read(source.parent / "COMPLETED.json")
                if (
                    receipt["binding"]["protocol_sha256"] != sha256(protocol_path)
                    or receipt["binding"]["case"] != case_id
                    or receipt["binding"]["requested_bps"] != bps
                    or receipt["video_sha256"] != sha256(source)
                ):
                    raise ValueError(
                        "Presentation video changed from the frozen HEVC cell"
                    )
                filename = f"{case_id}-{height}p-{bps // 1000}k-{method}.mp4"
                target = args.output / filename
                if target.exists():
                    raise FileExistsError(
                        "Comparison video already exists; inspect before rerun"
                    )
                info = render(source, target, model, case["line"]["xywh"], height)
                rows.append(
                    dict(
                        case=case_id,
                        source_group=case["source_group"],
                        source_kind=case["source_kind"],
                        method=method,
                        base_height=height,
                        requested_bps=bps,
                        video=filename,
                        checkpoint_sha256=final["test_checkpoint_sha256"][method],
                        **info,
                    )
                )
                atomic_json(
                    args.output / "progress.json",
                    dict(
                        rows=rows,
                        media_selection_sha256=sha256(root / "MEDIA_SELECTION.json"),
                    ),
                )
                print(
                    "MEDIA VIDEO",
                    len(rows),
                    "/",
                    len(selection["case_ids"]) * len(selection["comparison_cells"]),
                    flush=True,
                )
        atomic_json(
            args.output / "results.json",
            dict(
                schema="isharescreen_low_bandwidth_comparison_media_v1",
                media_selection_sha256=sha256(root / "MEDIA_SELECTION.json"),
                eval_protocol_sha256=sha256(protocol_path),
                rows=rows,
                elapsed_gpu_job_s=time.time() - start,
                finished=time.time(),
            ),
        )
        atomic_json(
            args.output / "COMPLETED.json",
            dict(
                results_sha256=sha256(args.output / "results.json"),
                videos=len(rows),
                finished=time.time(),
            ),
        )


if __name__ == "__main__":
    main()
