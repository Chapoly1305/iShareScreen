"""Generate train-only ITSRN targets from frozen Mac steady 420 inputs.

This is an offline teacher, not an inference-path change. No development or test
images are accepted. Frozen original HR remains the truth; generated targets are
auxiliary supervision and never replace it or count as independent examples.
"""

import argparse
import fcntl
import importlib
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

from benchmarks.super_resolution.mac_readability.itsrn_probe import run, sha
from benchmarks.super_resolution.mac_readability.itsrn_cost import frame_cache
from benchmarks.super_resolution.training_state import atomic_json, fingerprint


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--smoke-only", action="store_true")
    a = p.parse_args()
    project = a.project.resolve()
    root = (
        project / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    source = project / "datasets/derived/mac-steady-720p-420-v1"
    out = root / "teacher-itsrn-steady-v1"
    out.mkdir(exist_ok=True)
    lock = (out / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = read(source / "DATA.json")
    frozen = read(source / "FROZEN.json")
    assert (
        data["complete"] and data["role"] == "train_only" and len(data["rows"]) == 512
    )
    assert (
        sha(source / "DATA.json")
        == frozen["data_sha256"]
        == "ce98d560bb23a7cddf9a84f1aaeca91f0b5401026a741c7746c5f97dd2fa2e6a"
    )
    assert read(source / "SMOKE-VALIDATION.json")["passed"]
    cost = read(root / "itsrn-cost-v1/RESULTS.json")
    assert cost["complete"] and all(x["exact"] for x in cost["comparisons"])
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
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    repo = root / "external-models/ITSRN"
    weight = root / "external-models/ITSRN_last.pth"
    assert (
        sha(weight)
        == "2f3fadbebc0cbdc4583b4935929570655101bf34de2a77fd2fa6771478daf98b"
    )
    assert (
        subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        == "5523d1ac2f02698eb502ed3ac779b9cd897fcadf"
    )
    assert not subprocess.check_output(
        ["git", "-C", str(repo), "diff", "--name-only"], text=True
    ).strip()
    sys.path.insert(0, str(repo / "code"))
    import models
    from utils import make_coord

    module = importlib.import_module("models.ITNSR")
    payload = torch.load(weight, map_location="cpu", weights_only=True)
    model = models.make(payload["model"])
    model.load_state_dict(payload["model"]["sd"], strict=True)
    model = model.cuda().eval().requires_grad_(False)
    binding = dict(
        source_sha256=sha(source / "DATA.json"),
        checkpoint_sha256=sha(weight),
        script_sha256=sha(Path(__file__)),
        probe_sha256=sha(Path(__file__).with_name("itsrn_probe.py")),
        cache_sha256=sha(Path(__file__).with_name("itsrn_cost.py")),
        cost_sha256=sha(root / "itsrn-cost-v1/RESULTS.json"),
        precision="fp32",
        query_batch=30000,
        output_shape=[1080, 1920, 3],
        teacher="Official ITSRN epoch283",
    )
    digest = fingerprint(binding)
    if (out / "PROTOCOL.json").exists():
        assert read(out / "PROTOCOL.json")["binding"] == binding
    else:
        atomic_json(
            out / "PROTOCOL.json",
            dict(binding=binding, scope=__doc__, created=time.time(), test_used=False),
        )
    atomic_json(
        out / "HOST.json",
        dict(
            host=socket.gethostname(),
            gpu=gpu,
            pid=os.getpid(),
            torch=str(torch.__version__),
        ),
    )
    status = out / "STATUS.json"

    def state(**kw):
        atomic_json(
            status, dict(pid=os.getpid(), updated=time.time(), binding=digest, **kw)
        )

    images = out / "images"
    images.mkdir(exist_ok=True)
    rows = data["rows"]
    assert len({r["id"] for r in rows}) == 512 and all(
        r["split"] == "train" for r in rows
    )
    # Explicitly cover both CodeSCAN and native Mac glyph training origins.
    smoke_rows = [
        next(r for r in rows if r["codec_dataset"] == name)
        for name in sorted({r["codec_dataset"] for r in rows})
    ]
    if not (out / "SMOKE-PASSED.json").exists():
        checks = []
        state(state="smoke")
        for row in smoke_rows:
            low = source / row["lr"]
            assert sha(low) == row["lr_sha256"]
            rgb = np.asarray(Image.open(low).convert("RGB"))
            expected, _ = run(model, rgb, make_coord, "fp32")
            with frame_cache(module, True):
                actual, timing = run(model, rgb, make_coord, "fp32")
            assert np.array_equal(expected, actual)
            checks.append(
                dict(
                    id=row["id"],
                    dataset=row["codec_dataset"],
                    exact_cached_output=True,
                    input_sha256=sha(low),
                    timing=timing,
                )
            )
        atomic_json(
            out / "SMOKE-PASSED.json", dict(binding=digest, passed=True, checks=checks)
        )
    assert read(out / "SMOKE-PASSED.json")["binding"] == digest
    if a.smoke_only:
        state(state="smoke_passed", completed=0, total=512)
        return
    manifest = out / "DATA.json"
    result = (
        read(manifest)
        if manifest.exists()
        else dict(
            schema="mac_itsrn_teacher_targets_v1",
            role="train_only_auxiliary_targets",
            binding=digest,
            complete=False,
            source_sha256=sha(source / "DATA.json"),
            rows=[],
            test_used=False,
        )
    )
    assert result["binding"] == digest
    done = {row["id"]: row for row in result["rows"]}
    assert set(done).issubset({row["id"] for row in rows})
    for row in result["rows"]:
        assert sha(out / row["teacher"]) == row["teacher_sha256"]
    try:
        for i, row in enumerate(rows):
            if row["id"] in done:
                continue
            state(
                state="generating",
                completed=len(result["rows"]),
                total=512,
                current_id=row["id"],
            )
            low = source / row["lr"]
            truth = source / row["hr"]
            assert sha(low) == row["lr_sha256"] and sha(truth) == row["hr_sha256"]
            rgb = np.asarray(Image.open(low).convert("RGB"))
            assert rgb.shape == (720, 1280, 3)
            with frame_cache(module, True):
                pixels, timing = run(model, rgb, make_coord, "fp32")
            assert pixels.shape == (1080, 1920, 3)
            target = images / (row["id"] + ".png")
            tmp = target.with_suffix(".tmp")
            Image.fromarray(pixels).save(tmp, format="PNG")
            os.replace(tmp, target)
            hr = np.asarray(Image.open(truth).convert("RGB")).astype(np.float32)
            mse = float(np.square(hr - pixels.astype(np.float32)).mean())
            result["rows"].append(
                dict(
                    id=row["id"],
                    group_id=row["group_id"],
                    split="train",
                    codec_dataset=row["codec_dataset"],
                    lr_sha256=row["lr_sha256"],
                    hr_sha256=row["hr_sha256"],
                    teacher=str(target.relative_to(out)),
                    teacher_sha256=sha(target),
                    teacher_pixel_sha256=__import__("hashlib")
                    .sha256(pixels.tobytes())
                    .hexdigest(),
                    full_frame_psnr_vs_original_hr=float(10 * np.log10(255**2 / mse)),
                    timing=timing,
                )
            )
            atomic_json(manifest, result)
            if len(result["rows"]) % 16 == 0:
                print("teacher", len(result["rows"]), "/512", flush=True)
        assert len(result["rows"]) == 512
        result.update(complete=True, finished=time.time())
        atomic_json(manifest, result)
        atomic_json(
            out / "FROZEN.json",
            dict(
                data_sha256=sha(manifest),
                binding=digest,
                count=512,
                scope="All original train IDs preserved; auxiliary teacher targets only, never validation/test or replacement ground truth",
            ),
        )
        state(state="completed", completed=512, total=512)
    except BaseException as e:
        state(
            state="failed",
            error_type=type(e).__name__,
            error=str(e),
            completed=len(result["rows"]),
            total=512,
        )
        raise


if __name__ == "__main__":
    main()
