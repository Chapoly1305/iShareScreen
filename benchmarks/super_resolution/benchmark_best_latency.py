"""Same-GPU CUDA-event forward latency of hash-pinned best checkpoints."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from benchmarks.super_resolution.legacy_validation_compare import adapter
from benchmarks.super_resolution.codescan_campaign.train import forward


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def guard():
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,memory.used",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if "GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a" not in gpu:
        raise RuntimeError("Wrong physical GPU")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    if any(
        int(pid) != os.getpid()
        for pid in processes.splitlines()
        if pid.strip().isdigit()
    ):
        raise RuntimeError("Another GPU compute process is active")
    return gpu


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    plan = json.loads((a.root / "PLAN.json").read_text())
    if (a.root / "RESULTS.json").exists():
        raise FileExistsError("Use a fresh benchmark directory")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    start_wall = time.time()
    result = dict(
        complete=False,
        plan=plan,
        hostname=platform.node(),
        gpu=guard(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        pid=os.getpid(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        started=start_wall,
        rows=[],
        scope="Batch1 CUDA-event forward, 720p RGB to native1440p; includes model normalization, excludes decode, transfer, final1080p resize and display. Repeated fixed input, not1000different images.",
        input="torch.linspace(0,1,1280*720*3), FP32 NCHW resident on GPU",
        tf32=False,
        cudnn_benchmark=True,
        percentile_method="numpy.percentile linear",
        quality_scope="PSNR copied from prior FP32 Mac validation; AMP quality is not re-evaluated here",
    )
    save(a.root / "RESULTS.json", result)
    try:
        for item in plan["models"]:
            key = item["model"]
            before = guard()
            path = a.root / "checkpoints" / f"{key}.pt"
            if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
                raise ValueError(f"Checkpoint hash mismatch: {key}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if (
                payload["epoch"] != item["epoch"]
                or abs(payload["mac_psnr"] - item["psnr"]) > 1e-9
            ):
                raise ValueError(f"Checkpoint selection mismatch: {key}")
            wrapper = adapter(key, "cuda:0")
            model = wrapper.model
            torch.nn.Module.load_state_dict(model, payload["state_dict"], strict=True)
            model.eval().requires_grad_(False)
            del payload
            x = torch.linspace(0, 1, 1280 * 720 * 3, device="cuda:0").reshape(
                1, 3, 720, 1280
            )
            samples = []
            telemetry = []
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                for _ in range(plan["warmup"]):
                    y = forward(model, key, x)
                torch.cuda.synchronize()
                if y.shape != (1, 3, 1440, 2560) or not torch.isfinite(y).all().item():
                    raise ValueError(f"Invalid warmup output: {key}")
                torch.cuda.reset_peak_memory_stats()
                for i in range(plan["samples"]):
                    begin.record()
                    y = forward(model, key, x)
                    end.record()
                    end.synchronize()
                    samples.append(begin.elapsed_time(end))
                    if (i + 1) % 100 == 0:
                        if not torch.isfinite(y).all().item():
                            raise ValueError(f"Nonfinite output: {key}")
                        telemetry.append(dict(sample=i + 1, gpu=guard()))
                        save(
                            a.root / "PROGRESS.json",
                            dict(
                                model=key,
                                samples=i + 1,
                                total=plan["samples"],
                                completed_models=len(result["rows"]),
                                elapsed_s=time.time() - start_wall,
                            ),
                        )
            row = dict(
                **item,
                samples=len(samples),
                p50_ms=float(np.percentile(samples, 50)),
                p95_ms=float(np.percentile(samples, 95)),
                p99_ms=float(np.percentile(samples, 99)),
                mean_ms=float(np.mean(samples)),
                min_ms=min(samples),
                max_ms=max(samples),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                before=before,
                telemetry=telemetry,
                samples_ms=samples,
            )
            result["rows"].append(row)
            save(a.root / "RESULTS.json", result)
            print(
                key,
                "p50",
                row["p50_ms"],
                "p95",
                row["p95_ms"],
                "p99",
                row["p99_ms"],
                flush=True,
            )
            del wrapper, model, x, y
            gc.collect()
            torch.cuda.empty_cache()
        result.update(
            complete=True, elapsed_s=time.time() - start_wall, finished=time.time()
        )
        save(a.root / "RESULTS.json", result)
        lines = [
            "# Same RTX 4090 AMP FP16 forward latency",
            "",
            "1000 measured calls/model after50warmups; batch1,720p input, native1440p output. No decode/transfer/display. Quality is prior FP32 validation.",
            "",
            "| Model | Best PSNR dB | Epoch | p50 ms | p95 ms | p99 ms |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in sorted(result["rows"], key=lambda r: r["psnr"], reverse=True):
            lines.append(
                f"| {row['model']} | {row['psnr']:.3f} | {row['epoch']} | {row['p50_ms']:.3f} | {row['p95_ms']:.3f} | {row['p99_ms']:.3f} |"
            )
        (a.root / "REPORT.md").write_text("\n".join(lines) + "\n")
    except BaseException as error:
        result.update(error=repr(error), failed=time.time())
        save(a.root / "RESULTS.json", result)
        raise


if __name__ == "__main__":
    main()
