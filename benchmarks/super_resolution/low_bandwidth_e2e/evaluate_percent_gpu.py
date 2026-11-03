"""Run frozen percent holdout EDSR choices on one GPU and save paired FHD crops.

The remote worker receives only its versioned code, pinned weights, reference
images and decoded HEVC frames. OCR is scored later on H100 with one backend.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import platform
import socket
import time

import cv2
import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.metrics import metrics
from benchmarks.super_resolution.text_corpus.scratch import build
from benchmarks.super_resolution.text_corpus.official import load as load_official
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


class NormalizedEdsr(torch.nn.Module):
    """The pinned EDSR adapter uses RGB 0..255; the shared evaluator uses 0..1."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.model(tensor * 255.0) / 255.0


def get_model(name: str, weights: dict, home: Path) -> tuple[torch.nn.Module, dict]:
    from isharescreen.enhancement import models

    if name not in (
        "old_finetuned_edsr_x2",
        "generic_tuned_edsr_x2",
        "symbol_tuned_edsr_x2",
    ):
        raise ValueError(
            "This isolated percent study only evaluates pinned EDSR choices"
        )
    adapter = models.Upscaler("edsr", device="cpu", home=home, precision="fp32")
    if adapter.provenance["checkpoint_sha256"] != weights["official_edsr_x2"]["sha256"]:
        raise ValueError("EDSR official initializer differs")
    old = weights["old_finetuned_edsr_x2"]
    old_path = home / old["asset"]
    manifest = home / "fine-tuned/edsr-fold-0-epoch-1500-current/manifest.json"
    if sha256(old_path) != old["sha256"] or sha256(manifest) != old["manifest_sha256"]:
        raise ValueError("Historical EDSR bundle differs")
    spec = read(manifest)
    if (
        spec["schema"] != "isharescreen_live_weights_v1"
        or spec["model"] != "edsr"
        or spec["fold"] != 0
        or spec["epoch"] != 1500
        or spec["role"] != "current"
        or spec["weights_sha256"] != old["sha256"]
        or spec["official_checkpoint_sha256"] != weights["official_edsr_x2"]["sha256"]
    ):
        raise ValueError("Historical EDSR manifest differs")
    state = torch.load(old_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(v, torch.Tensor) and torch.isfinite(v).all().item()
            for v in state.values()
        )
    ):
        raise ValueError("Historical EDSR must be finite tensor state")
    torch.nn.Module.load_state_dict(adapter.model, state, strict=True)
    model = NormalizedEdsr(adapter.model)
    bound = weights[name]
    path = home / bound["asset"]
    if sha256(path) != bound["sha256"]:
        raise ValueError("Selected EDSR checkpoint differs")
    if name in ("generic_tuned_edsr_x2", "symbol_tuned_edsr_x2"):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload["protocol_binding"] != bound["training_protocol_binding"]
            or payload["step"] != bound["selected_step"]
        ):
            raise ValueError("Symbol-trained EDSR protocol/checkpoint differs")
        torch.nn.Module.load_state_dict(model, payload["state_dict"], strict=True)
    identity = dict(
        checkpoint_sha256=bound["sha256"],
        parameters=adapter.provenance["parameters"],
        source_revision="8dba5581a7502b92de9641eb431130d6c8ca5d7f",
        precision="fp32",
        variant=bound["training"],
    )
    return model, identity


def run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    protocol_path = root / "EVAL_PROTOCOL.json"
    protocol = read(protocol_path)
    weights = read(args.weights.resolve())
    if weights["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Weights/evaluation protocol binding differs")
    if args.split == "test":
        if args.selection is None or not args.selection.exists():
            raise ValueError(
                "Frozen validation selection is required before test inference"
            )
        selected = read(args.selection)
        if selected["eval_protocol_sha256"] != sha256(protocol_path):
            raise ValueError("Validation selection refers to another test protocol")
        if args.method not in selected["test_methods"]:
            raise ValueError(
                "Method was not in the frozen validation-selected test roster"
            )
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / "TASK.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(2)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        if not torch.cuda.is_available():
            raise ValueError("GPU worker has no CUDA device")
        device = torch.device("cuda:0")
        model, identity = get_model(args.method, weights["models"], args.home.resolve())
        model.to(device).eval().requires_grad_(False)
        torch.cuda.reset_peak_memory_stats(device)
        started = time.time()
        manifest_path = out / "progress.json"
        binding = dict(
            protocol_sha256=sha256(protocol_path),
            split=args.split,
            method=args.method,
            checkpoint_sha256=identity["checkpoint_sha256"],
            runner_sha256=sha256(Path(__file__)),
            limit_cases=args.limit_cases,
        )
        if manifest_path.exists():
            existing = read(manifest_path)
            if existing["binding"] != binding:
                raise ValueError(
                    "Previous GPU evaluation has another protocol or checkpoint"
                )
            rows = existing["rows"]
            for row in rows:
                if sha256(out / row["crop"]) != row["crop_sha256"]:
                    raise ValueError("Saved crop changed before GPU evaluation resume")
                if (
                    row["display"]
                    and sha256(out / row["display"]["path"]) != row["display"]["sha256"]
                ):
                    raise ValueError(
                        "Saved display frame changed before GPU evaluation resume"
                    )
            done = {
                (r["case"], r["base_height"], r["requested_bps"], r["frame_index"])
                for r in rows
            }
        else:
            rows = []
            done = set()
        cases = [c for c in protocol["cases"] if c["split"] == args.split]
        if args.limit_cases is not None:
            cases = cases[: args.limit_cases]
        expected = len(cases) * 3 * 2 * 2
        display = set(protocol["fixed_display_ids"][args.split])
        for case in cases:
            reference_path = root / case["hr"]
            if sha256(reference_path) != case["hr_sha256"]:
                raise ValueError(f"Reference changed: {case['id']}")
            reference = np.array(Image.open(reference_path).convert("RGB"))
            for width, height in protocol["main_grid"]["base_geometry"]:
                for bps in protocol["main_grid"]["requested_video_bps"]:
                    cell = (
                        root
                        / "evaluation-inputs"
                        / case["id"]
                        / f"{width}x{height}-{bps // 1000}k"
                    )
                    receipt = read(cell / "COMPLETED.json")
                    if (
                        receipt["binding"]["protocol_sha256"]
                        != binding["protocol_sha256"]
                    ):
                        raise ValueError(
                            "Decoded HEVC frame belongs to another protocol"
                        )
                    for frame in receipt["sampled_frames"]:
                        key = (case["id"], height, bps, frame["index"])
                        if key in done:
                            continue
                        image_path = cell / frame["path"]
                        if sha256(image_path) != frame["sha256"]:
                            raise ValueError("Decoded HEVC pixels changed")
                        image = np.array(Image.open(image_path).convert("RGB"))
                        before = time.perf_counter()
                        tensor = (
                            torch.from_numpy(image.transpose(2, 0, 1).copy())[None]
                            .to(device)
                            .float()
                            .div_(255)
                        )
                        torch.cuda.synchronize(device)
                        begin = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                        begin.record()
                        with torch.inference_mode():
                            output = model(tensor)
                        end.record()
                        torch.cuda.synchronize(device)
                        forward_ms = begin.elapsed_time(end)
                        if (
                            output.shape != (1, 3, height * 2, width * 2)
                            or not torch.isfinite(output).all()
                        ):
                            raise ValueError("Invalid x2 model output")
                        pixels = (
                            output[0]
                            .float()
                            .mul(255)
                            .round()
                            .clamp(0, 255)
                            .byte()
                            .permute(1, 2, 0)
                            .cpu()
                            .numpy()
                        )
                        display_pixels = (
                            pixels
                            if height == 540
                            else cv2.resize(
                                pixels, (1920, 1080), interpolation=cv2.INTER_LANCZOS4
                            )
                        )
                        torch.cuda.synchronize(device)
                        full_path_ms = (time.perf_counter() - before) * 1000
                        x, y, w, h = case["line"]["xywh"]
                        folder = (
                            out / "crops" / case["id"] / f"{height}p-{bps // 1000}k"
                        )
                        folder.mkdir(parents=True, exist_ok=True)
                        crop = folder / f"frame-{frame['index']:03d}.png"
                        Image.fromarray(display_pixels[y : y + h, x : x + w]).save(crop)
                        full = None
                        if case["id"] in display:
                            folder = (
                                out
                                / "display"
                                / case["id"]
                                / f"{height}p-{bps // 1000}k"
                            )
                            folder.mkdir(parents=True, exist_ok=True)
                            image_file = folder / f"frame-{frame['index']:03d}.png"
                            Image.fromarray(display_pixels).save(image_file)
                            full = dict(
                                path=str(image_file.relative_to(out)),
                                sha256=sha256(image_file),
                            )
                        quality = metrics(reference, display_pixels)
                        rows.append(
                            dict(
                                case=case["id"],
                                source_group=case["source_group"],
                                source_kind=case["source_kind"],
                                content_category=case["content_category"],
                                base_height=height,
                                requested_bps=bps,
                                measured_video_bps=receipt[
                                    "measured_video_bps_after_first_second"
                                ],
                                encoder_eligible=receipt["encoder_eligible"],
                                frame_index=frame["index"],
                                keyframe=frame["keyframe"],
                                native_output_geometry=[width * 2, height * 2],
                                display_geometry=[1920, 1080],
                                forward_ms=forward_ms,
                                full_processing_ms=full_path_ms,
                                psnr=quality["psnr"],
                                ssim=quality["ssim"],
                                crop=str(crop.relative_to(out)),
                                crop_sha256=sha256(crop),
                                display=full,
                            )
                        )
                        done.add(key)
                    atomic_json(
                        manifest_path,
                        dict(
                            binding=binding,
                            identity=identity,
                            rows=rows,
                            progress=len(rows),
                            expected=expected,
                            updated=time.time(),
                        ),
                    )
                    print(
                        "GPU EVAL",
                        args.method,
                        args.split,
                        len(rows),
                        "/",
                        expected,
                        flush=True,
                    )
        if len(rows) != expected or len(done) != expected:
            raise ValueError("Incomplete or duplicate GPU evaluation rows")
        final = dict(
            binding=binding,
            identity=identity,
            rows=rows,
            subset_smoke=args.limit_cases is not None,
            host=socket.gethostname(),
            platform=platform.platform(),
            gpu=torch.cuda.get_device_name(device),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            torch=str(torch.__version__),
            cuda_runtime=torch.version.cuda,
            precision="fp32",
            tf32=False,
            max_allocated_bytes=torch.cuda.max_memory_allocated(device),
            max_reserved_bytes=torch.cuda.max_memory_reserved(device),
            elapsed_gpu_job_s=time.time() - started,
            completed=time.time(),
            scope="Warm/cold mixed per-frame model inference, source-bound paired pixels and full postdecode processing through FHD resize/crop; no decode, display upload, live queue or Mac timing.",
        )
        atomic_json(out / "results.json", final)
        atomic_json(
            out / "COMPLETED.json",
            dict(
                results_sha256=sha256(out / "results.json"),
                rows=len(rows),
                checkpoint_sha256=identity["checkpoint_sha256"],
                finished=time.time(),
            ),
        )
        print("GPU EVAL COMPLETE", args.method, args.split, len(rows), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--limit-cases", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
