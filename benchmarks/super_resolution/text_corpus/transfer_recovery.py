"""Explicit environment-only recovery; completed training is never repeated."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["start", "render"])
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "code") + ":" + str(root / "code/src"),
        CUDA_VISIBLE_DEVICES="GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a",
        OMP_NUM_THREADS="2",
    )
    python = root / "runtime-venv/bin/python"
    command = [
        str(python),
        "-u",
        "-m",
        "benchmarks.super_resolution.text_corpus.evaluate",
        "render",
        "--root",
        str(root / "data"),
        "--training",
        str(root / "runs/official-finetuned"),
        "--home",
        str(root / "assets"),
        "--device",
        "cuda:0",
        "--output",
        str(root / "data/evaluations/official-finetuned"),
    ]
    if a.action == "render":
        with (root / "logs/official-finetuned-render-recovered.log").open("xb") as log:
            result = subprocess.run(
                command,
                cwd=root / "code",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        write(
            root / "logs/official-finetuned-render-recovered.json",
            dict(exit_code=result.returncode, phase="render", finished=time.time()),
        )
        raise SystemExit(result.returncode)
    if (root / "runtime-recovery.json").exists():
        raise ValueError("Recovery already launched")
    assert (root / "runs/official-finetuned/COMPLETED.json").is_file()
    assert (
        not (root / "runs/scratch-control").exists()
        and not (root / "data/evaluations/official-finetuned").exists()
    )
    # Fail before launching if the complete frozen evaluation import graph fails.
    subprocess.run(
        [
            str(python),
            "-c",
            "from benchmarks.super_resolution.text_corpus.ocr_protocol import prepare_crop",
        ],
        cwd=root / "code",
        env=env,
        check=True,
    )
    commands = [
        (
            "official-render",
            [str(python), str(Path(__file__).resolve()), "render", "--root", str(root)],
        ),
        (
            "scratch-worker",
            [
                str(python),
                "-u",
                "-m",
                "benchmarks.super_resolution.text_corpus.transfer_campaign",
                "worker",
                "--root",
                str(root),
                "--name",
                "scratch-control",
                "--python",
                str(python),
            ],
        ),
    ]
    jobs = []
    for name, cmd in commands:
        with (root / "logs" / (name + "-recovery-worker.log")).open("xb") as log:
            child = subprocess.Popen(
                ["nice", "-n", "10"] + cmd,
                cwd=root / "code",
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        jobs.append(dict(name=name, pid=child.pid, command=cmd))
    write(
        root / "runtime-recovery.json",
        dict(
            reason="Missing optional skimage at evaluation import after completed official training; first scratch run had not started.",
            completed_official_training_preserved=True,
            original_failure_receipt="logs/official-finetuned-exit.json",
            official_recovered_render_receipt="logs/official-finetuned-render-recovered.json",
            python=str(python),
            inherited_site_packages="/home/chen/venvs/vllm/lib/python3.12/site-packages",
            added_packages={
                k: importlib.metadata.version(k)
                for k in ["scikit-image", "scipy", "lazy-loader", "imageio", "tifffile"]
            },
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            jobs=jobs,
            started=time.time(),
        ),
    )
    print(json.dumps(jobs), flush=True)


if __name__ == "__main__":
    main()
