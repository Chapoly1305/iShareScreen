"""Official ITSRN replay on frozen Mac inputs; isolated development experiment.

Uses the unmodified author's architecture and checkpoint. Unlike the author's
permissive state-dict loader, all parameter names and shapes must match exactly.
The FP32 official path is the quality reference; AMP is a diagnostic only.
"""

import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2) + "\n")


def run(model, rgb, make_coord, precision, chunk=30000):
    begin = time.perf_counter()
    h, w = rgb.shape[:2]
    oh, ow = h * 3 // 2, w * 3 // 2
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div(255)
    x = ((x - 0.5) / 0.5).unsqueeze(0).cuda()
    coord = make_coord((oh, ow)).cuda().unsqueeze(0)
    scale = torch.ones_like(coord)
    scale[:, :, 0] /= oh
    scale[:, :, 1] /= ow
    torch.cuda.synchronize()
    prepared = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with (
        torch.inference_mode(),
        (
            torch.autocast("cuda", dtype=torch.float16)
            if precision == "amp-fp16"
            else nullcontext()
        ),
    ):
        model.gen_feat(x)
        predictions = [
            model.query_rgb(coord[:, i : i + chunk], scale[:, i : i + chunk])
            for i in range(0, oh * ow, chunk)
        ]
        pred = torch.cat(predictions, dim=1)[0]
    torch.cuda.synchronize()
    inferred = time.perf_counter()
    # Matches torchvision ToPILImage's truncation in the official demo.
    array = (pred.float() * 0.5 + 0.5).clamp(0, 1).view(oh, ow, 3).cpu().numpy()
    assert np.isfinite(array).all()
    pixels = (array * 255).astype(np.uint8)
    timing = dict(
        prepare_ms=(prepared - begin) * 1000,
        inference_ms=(inferred - prepared) * 1000,
        total_ms=(time.perf_counter() - begin) * 1000,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
    )
    del model.feat
    return pixels, timing


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--mode", choices=["smoke", "full"], required=True)
    a = p.parse_args()
    r = a.root.resolve()
    external = r / "external-models"
    repo = external / "ITSRN"
    checkpoint = external / "ITSRN_last.pth"
    output = r / ("itsrn-smoke-v1" if a.mode == "smoke" else "itsrn-official-v1")
    output.mkdir(exist_ok=False)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    assert "GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a" in gpu
    owners = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    assert not any(
        x.strip().isdigit() and int(x) != os.getpid() for x in owners.splitlines()
    )
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    assert revision == "5523d1ac2f02698eb502ed3ac779b9cd897fcadf"
    assert not subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--name-only"], text=True
    ).strip()
    assert (
        sha(checkpoint)
        == "2f3fadbebc0cbdc4583b4935929570655101bf34de2a77fd2fa6771478daf98b"
    )
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(repo / "code"))
    import models
    from utils import make_coord

    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = models.make(payload["model"])
    model.load_state_dict(payload["model"]["sd"], strict=True)
    model = model.cuda().eval().requires_grad_(False)
    inputs_path = r / "dense-inference-inputs/INPUTS.json"
    inputs = json.loads(inputs_path.read_text())
    assert len(inputs["images"]) == 32
    result = dict(
        schema="official_itsrn_mac_probe_v1",
        complete=False,
        mode=a.mode,
        started=time.time(),
        host=socket.gethostname(),
        gpu=gpu,
        torch=torch.__version__,
        author_repository="https://github.com/codyshen0000/ITSRN",
        revision=revision,
        checkpoint_sha256=sha(checkpoint),
        checkpoint_epoch=payload["epoch"],
        script_sha256=sha(Path(__file__)),
        input_manifest_sha256=sha(inputs_path),
        source_sha256=inputs["source_sha256"],
        model_parameters=sum(x.numel() for x in model.parameters()),
        quality_precision="fp32",
        query_batch_size=30000,
        output_resolution=[1920, 1080],
        quality_scope="Frozen real Mac 720p / 600 Kbps / 420 at 2.8s source age, direct 1.5x; no labels or HR input in inference",
        timing_scope="Sequential RTX4090 tensor pipeline; PNG IO excluded, includes query batching; not asynchronous display or network latency",
        test_used=False,
        images=[],
        checks=[],
        timings=[],
    )
    save(output / "RESULTS.json", result)
    rows = inputs["images"][:1] if a.mode == "smoke" else inputs["images"]
    if a.mode == "full":
        smoke = json.loads((r / "itsrn-smoke-v1/RESULTS.json").read_text())
        assert smoke["complete"] and smoke["checkpoint_sha256"] == sha(checkpoint)
    for row in rows:
        path = inputs_path.parent / row["input_path"]
        assert sha(path) == row["decoded_sha256"]
        rgb = np.asarray(Image.open(path).convert("RGB"))
        assert rgb.shape == (720, 1280, 3)
        if a.mode == "smoke":
            crop = rgb[120:184, 120:184]
            small, _ = run(model, crop, make_coord, "fp32", 1000)
            small2, _ = run(model, crop, make_coord, "fp32", 3000)
            diff = np.abs(small.astype("i2") - small2.astype("i2"))
            assert diff.max() <= 1 and diff.mean() < 0.01
            result["checks"].append(
                dict(
                    check="query_chunk_equivalence_64px",
                    max_pixel_delta=int(diff.max()),
                    mean_pixel_delta=float(diff.mean()),
                )
            )
        precisions = ["fp32", "amp-fp16"] if a.mode == "smoke" else ["fp32"]
        previous = None
        for precision in precisions:
            pixels, timing = run(model, rgb, make_coord, precision)
            name = row["id"] + "-itsrn-" + precision
            target = output / (name + ".png")
            Image.fromarray(pixels).save(target)
            result["images"].append(
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
                    id=name,
                    method="itsrn-official-" + precision,
                    path=target.name,
                    sha256=sha(target),
                    input_sha256=row["decoded_sha256"],
                    timing=timing,
                )
            )
            if previous is not None:
                diff = np.abs(previous.astype("i2") - pixels.astype("i2"))
                result["checks"].append(
                    dict(
                        check="amp_vs_fp32_full_frame",
                        max_pixel_delta=int(diff.max()),
                        mean_pixel_delta=float(diff.mean()),
                        fraction_pixels_different=float(np.mean(np.any(diff, axis=2))),
                    )
                )
            previous = pixels
            print(name, timing, flush=True)
            save(output / "RESULTS.json", result)
        if a.mode == "smoke":
            for i in range(5):
                _, timing = run(model, rgb, make_coord, "amp-fp16")
                result["timings"].append(
                    dict(precision="amp-fp16", repetition=i, **timing)
                )
                save(output / "RESULTS.json", result)
                print("warm timing", i, timing, flush=True)
    result.update(complete=True, finished=time.time())
    save(output / "RESULTS.json", result)


if __name__ == "__main__":
    main()
