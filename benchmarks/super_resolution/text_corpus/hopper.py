"""Frozen six-run scratch campaign on Hopper; no implicit retries/resumes."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement.models import sha256


def read(path):
    return json.loads(path.read_text())


def experiments():
    return [
        dict(
            name=f"{loss}-seed-{seed}",
            seed=seed,
            epochs=300,
            text_weight=text,
            edge_weight=edge,
        )
        for loss, text, edge in [("l1", 0.0, 0.0), ("text-edge", 2.0, 0.05)]
        for seed in (682, 683, 684)
    ]


def identities(root):
    files = list((root / "code").rglob("*.py")) + list((root / "code").rglob("*.sh"))
    files += list((root / "assets").rglob("*"))
    files += [
        root / "data" / name
        for name in [
            "dataset-views-v3.json",
            "quality-audit-v3.json",
            "hevc-v2/pairs.json",
            "evaluation-protocol/protocol.json",
            "evaluation-protocol/source-compatibility.json",
        ]
    ]
    return {
        str(p.relative_to(root)): sha256(p)
        for p in sorted(set(files))
        if p.is_file() and "__pycache__" not in p.parts
    }


def submit(args):
    root = args.root.resolve()
    out = root / "campaign.json"
    if out.exists():
        raise ValueError(
            "Campaign already prepared; inspect submission.json and Slurm, never resubmit implicitly"
        )
    if not (root / "TRANSFER_COMPLETE").is_file():
        raise ValueError("Input transfer has not completed")
    audit = read(root / "data/quality-audit-v3.json")
    if not audit["passed"]:
        raise ValueError("Dataset audit failed")
    plan = dict(
        schema="isharescreen_text_hopper_campaign_v1",
        runs=experiments(),
        frozen_files=identities(root),
        sampling="Same 80% source-text / 20% uniform crops in both losses; this is a loss ablation, not a sampler ablation.",
        validation=dict(
            seed=682,
            text_weight=2.0,
            edge_weight=0.05,
            selection="minimum common fixed validation proxy; never test OCR",
        ),
        comparison="Report all three seeds per loss; do not choose the best seed using test OCR. H100 100-epoch trial has a different budget.",
        gpu_concurrency=3,
        python=str(args.python),
        created=time.time(),
    )
    atomic_json(out, plan)
    (root / "slurm").mkdir()
    (root / "runs").mkdir(exist_ok=True)
    script = Path(__file__).with_name("slurm_text_corpus.sh")
    common = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        "--cpus-per-task=4",
        "--mem=32G",
        "--no-requeue",
        "--chdir=" + str(root / "code"),
        "--output=" + str(root / "slurm/%x-%A_%a.log"),
    ]
    export = "ALL,ISS_TEXT_ROOT=" + str(root) + ",ISS_TEXT_PYTHON=" + str(args.python)
    receipt = dict(campaign_sha256=sha256(out), created=time.time(), state="submitting")
    atomic_json(root / "submission.json", receipt)

    def job(command):
        raw = subprocess.check_output(command, text=True).strip()
        jobid = raw.split(";")[0]
        if not jobid.isdigit():
            raise ValueError("Unexpected sbatch response: " + raw)
        return jobid

    pack = job(
        common
        + [
            "--partition=normal",
            "--time=02:00:00",
            "--job-name=text-pack",
            "--export=" + export + ",ISS_TEXT_STAGE=pack",
            str(script),
        ]
    )
    receipt.update(pack_job=pack)
    atomic_json(root / "submission.json", receipt)
    array = job(
        common
        + [
            "--partition=contrib-gpuq",
            "--qos=gpu",
            "--gres=gpu:A100.80gb:1",
            "--time=12:00:00",
            "--job-name=text-scratch",
            "--array=0-5%3",
            "--dependency=afterok:" + pack,
            "--kill-on-invalid-dep=yes",
            "--export=" + export + ",ISS_TEXT_STAGE=train",
            str(script),
        ]
    )
    receipt.update(array_job=array, state="submitted", submitted=time.time())
    atomic_json(root / "submission.json", receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def worker(args):
    root = args.root.resolve()
    plan = read(root / "campaign.json")
    if identities(root) != plan["frozen_files"]:
        raise ValueError("Frozen campaign code, assets or input metadata changed")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "code") + ":" + str(root / "code/src")
    env["OMP_NUM_THREADS"] = "4"

    def run(module, arguments, log):
        with log.open("xb") as stream:
            subprocess.run(
                [
                    str(args.python),
                    "-u",
                    "-m",
                    "benchmarks.super_resolution.text_corpus." + module,
                    *arguments,
                ],
                cwd=root / "code",
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )

    host = dict(
        job_id=os.environ.get("SLURM_JOB_ID"),
        array_id=os.environ.get("SLURM_ARRAY_TASK_ID"),
        hostname=socket.gethostname(),
        pid=os.getpid(),
        campaign_binding=fingerprint(plan),
        started=time.time(),
    )
    if args.stage == "pack":
        atomic_json(root / "pack-job.json", host)
        run("pack", ["--root", str(root / "data")], root / "pack.log")
        packed = read(root / "data/packed-v1/manifest.json")
        if packed["source_pairs_sha256"] != sha256(root / "data/hevc-v2/pairs.json"):
            raise ValueError("Wrong packed pairs")
        atomic_json(
            root / "PACK_READY.json",
            dict(
                host,
                packed_sha256=sha256(root / "data/packed-v1/manifest.json"),
                finished=time.time(),
            ),
        )
        return
    ready = read(root / "PACK_READY.json")
    manifest = root / "data/packed-v1/manifest.json"
    if ready["packed_sha256"] != sha256(manifest):
        raise ValueError("Packed manifest differs from preparation job")
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    spec = plan["runs"][index]
    name = spec["name"]
    output = root / "runs" / name
    atomic_json(root / "slurm" / (name + "-host.json"), host)
    arguments = [
        "--packed",
        str(manifest),
        "--output",
        str(output),
        "--home",
        str(root / "assets"),
        "--device",
        "cuda:0",
        "--epochs",
        str(spec["epochs"]),
        "--seed",
        str(spec["seed"]),
        "--batch",
        "32",
        "--patch",
        "96",
        "--lr",
        "0.001",
        "--text-weight",
        str(spec["text_weight"]),
        "--edge-weight",
        str(spec["edge_weight"]),
        "--validation-seed",
        "682",
        "--validation-text-weight",
        "2",
        "--validation-edge-weight",
        "0.05",
    ]
    run("train", arguments, root / "slurm" / (name + "-train.log"))
    run(
        "evaluate",
        [
            "render",
            "--root",
            str(root / "data"),
            "--training",
            str(output),
            "--home",
            str(root / "assets"),
            "--device",
            "cuda:0",
            "--output",
            str(root / "data/evaluations" / name),
        ],
        root / "slurm" / (name + "-render.log"),
    )
    atomic_json(
        root / "slurm" / (name + "-COMPLETED.json"),
        dict(
            host,
            training_receipt_sha256=sha256(output / "COMPLETED.json"),
            render_receipt_sha256=sha256(
                root / "data/evaluations" / name / "rendered.json"
            ),
            finished=time.time(),
            scope="Training and frozen OCR crop rendering completed. OCR scoring runs separately with the frozen local Tesseract.",
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["submit", "worker"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--stage", choices=["pack", "train"])
    args = p.parse_args()
    dict(submit=submit, worker=worker)[args.action](args)


if __name__ == "__main__":
    main()
