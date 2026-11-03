"""Isolate clipping/quantization order in frozen EDSR 720-to-1080 replay.

Same network output is used for all three display transforms. Legacy output
must match the frozen images. New alternatives are frozen before OCR scoring.
No model training, source-label input, or test data are used.
"""

import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.mac_readability import train_ablations as base
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement import models


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    a = p.parse_args()
    root = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    out = root / "resize-order-v1"
    out.mkdir(exist_ok=False)
    gpu = base.verify_device()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = root / "dense-inference-inputs/INPUTS.json"
    data = base.read(inputs)
    assert len(data["images"]) == 32
    legacy_official = base.read(root / "dense-models-v1/RESULTS.json")
    legacy_trained = base.read(root / "TRAINING-V2-SELECTED-IMAGES.json")
    protocol = dict(
        scope=__doc__,
        started=time.time(),
        gpu=gpu,
        precision="amp-fp16",
        script_sha256=base.sha(Path(__file__)),
        input_sha256=base.sha(inputs),
        variants=[
            "legacy_uint8_then_area",
            "clamp_float_then_area",
            "area_float_then_clamp",
        ],
        models=["edsr-official", "edsr-steady-v2-step1000"],
        selection="Frozen hard15 CER; compare clipping andquantization separately; no finaltest",
        source_sha256=data["source_sha256"],
        test_used=False,
        complete=False,
        images=[],
        parity=[],
        diagnostics=[],
    )
    atomic_json(out / "RESULTS.json", protocol)
    for trained in [False, True]:
        label = "edsr-steady-v2-step1000" if trained else "edsr-official"
        adapter = models.Upscaler("edsr", device="cuda:0", precision="amp-fp16")
        if trained:
            path = root / "selected-v2/checkpoint-001000.pt"
            assert (
                base.sha(path)
                == "01d99513d182f669e1e1e92c88e0e17a9a053b6e2d046a8733119047e61db4cd"
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            torch.nn.Module.load_state_dict(
                adapter.model, payload["state_dict"], strict=True
            )
        model = adapter.model.eval().requires_grad_(False)
        for row in data["images"]:
            path = inputs.parent / row["input_path"]
            assert base.sha(path) == row["decoded_sha256"]
            rgb = np.asarray(Image.open(path).convert("RGB"))
            x = (
                torch.from_numpy(rgb.copy())
                .permute(2, 0, 1)
                .unsqueeze(0)
                .float()
                .cuda()
            )
            torch.cuda.synchronize()
            begin = time.perf_counter()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                native = model(x)
            native = native.float()
            assert torch.isfinite(native).all()
            torch.cuda.synchronize()
            inference_ms = (time.perf_counter() - begin) * 1000
            original = (
                native[0]
                .round()
                .clamp(0, 255)
                .to(torch.uint8)
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            variants = {
                "legacy_uint8_then_area": cv2.resize(
                    original, (1920, 1080), interpolation=cv2.INTER_AREA
                )
            }
            for name, value in [
                ("clamp_float_then_area", native.clamp(0, 255)),
                ("area_float_then_clamp", native),
            ]:
                start = time.perf_counter()
                pixels = (
                    base.area_4_to_3(value)[0]
                    .round()
                    .clamp(0, 255)
                    .to(torch.uint8)
                    .permute(1, 2, 0)
                    .contiguous()
                    .cpu()
                    .numpy()
                )
                variants[name] = pixels
                protocol["diagnostics"].append(
                    dict(
                        model=label,
                        state_id=row["state_id"],
                        variant=name,
                        transform_download_ms=(time.perf_counter() - start) * 1000,
                        inference_ms=inference_ms,
                    )
                )
            expected = next(
                v
                for v in (legacy_trained if trained else legacy_official)["images"]
                if v["state_id"] == row["state_id"]
                and (trained or v["method"] == "edsr-official")
            )
            expected_path = (
                root
                / "training-v2/models/steady_l1/validation-001000"
                / Path(expected["path"]).name
                if trained
                else root / "dense-models-v1" / expected["path"]
            )
            assert base.sha(expected_path) == expected["sha256"]
            delta = np.abs(
                variants["legacy_uint8_then_area"].astype("i2")
                - np.asarray(Image.open(expected_path)).astype("i2")
            )
            assert delta.max() <= 1
            protocol["parity"].append(
                dict(
                    model=label,
                    state_id=row["state_id"],
                    max_delta=int(delta.max()),
                    exact=bool(not delta.any()),
                    native_below_zero=float((native < 0).float().mean()),
                    native_above_255=float((native > 255).float().mean()),
                )
            )
            for name, pixels in variants.items():
                method = label + "-" + name
                name_on_disk = row["id"] + "-" + method + ".png"
                path = out / name_on_disk
                Image.fromarray(pixels).save(path)
                protocol["images"].append(
                    {
                        k: row[k]
                        for k in [
                            "group",
                            "state_id",
                            "config",
                            "source_age_s",
                            "height",
                            "codec_frame_count_valid",
                        ]
                    }
                    | dict(
                        id=row["id"] + "-" + method,
                        method=method,
                        path=name_on_disk,
                        sha256=base.sha(path),
                        input_sha256=row["decoded_sha256"],
                    )
                )
            atomic_json(out / "RESULTS.json", protocol)
        print(label, "32 frames, 3 transforms", flush=True)
        del adapter, model
        torch.cuda.empty_cache()
    protocol.update(complete=True, finished=time.time())
    atomic_json(out / "RESULTS.json", protocol)


if __name__ == "__main__":
    main()
