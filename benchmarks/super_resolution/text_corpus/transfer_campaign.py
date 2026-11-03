"""Isolated official-weight transfer experiment with a matched scratch control."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path):
    return json.loads(path.read_text())


def prepare(root, previous, epochs):
    if (root / "campaign.json").exists():
        raise ValueError("Campaign already prepared")
    # The uploaded overlay is installed before copying unchanged dependencies.
    code = root / "code"
    code.mkdir(parents=True, exist_ok=True)
    for source in (previous / "code").rglob("*"):
        if not source.is_file() or "__pycache__" in source.parts:
            continue
        target = code / source.relative_to(previous / "code")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    shutil.copytree(previous / "assets", root / "assets")
    data = root / "data"
    data.mkdir()
    (data / "packed-v1").mkdir()
    (data / "evaluation-protocol").mkdir()
    for source in (previous / "data").iterdir():
        if source.name not in (
            "packed-v1",
            "evaluation-protocol",
            "evaluations",
            "evaluation-v1",
        ):
            (data / source.name).symlink_to(
                source.resolve(), target_is_directory=source.is_dir()
            )
    packed = read(previous / "data/packed-v1/manifest.json")
    shutil.copy2(
        previous / "data/packed-v1/manifest.json", data / "packed-v1/manifest.json"
    )
    for entry in packed["files"].values():
        (data / "packed-v1" / entry["path"]).symlink_to(
            (previous / "data/packed-v1" / entry["path"]).resolve()
        )
    allowed = {256000, 600000, 1000000, 2000000}
    pairs = {r["id"]: r for r in read(data / "hevc-v2/pairs.json")["pairs"]}
    for row in packed["rows"]:
        if row.get("requested_bps") != pairs[row["id"]]["requested_bps"]:
            raise ValueError("Packed rate identity differs")
    protocol = read(previous / "data/evaluation-protocol/protocol.json")
    protocol["parent_protocol_sha256"] = sha256(
        previous / "data/evaluation-protocol/protocol.json"
    )
    protocol["cases"] = [
        c for c in protocol["cases"] if pairs[c["view"]]["requested_bps"] in allowed
    ]
    protocol["counts"] = dict(Counter(c["split"] for c in protocol["cases"]))
    protocol["selection"] += (
        "; retained existing regions only at the new campaign allowed bitrate settings; no new OCR selection"
    )
    protocol["finished"] = time.time()
    for case in protocol["cases"]:
        destination = data / case["clean_crop"]
        destination.symlink_to((previous / "data" / case["clean_crop"]).resolve())
    (data / "evaluation-protocol/source-compatibility.json").symlink_to(
        (previous / "data/evaluation-protocol/source-compatibility.json").resolve()
    )
    atomic_json(data / "evaluation-protocol/protocol.json", protocol)
    plan = dict(
        schema="isharescreen_official_transfer_campaign_v1",
        created=time.time(),
        epochs=epochs,
        seed=682,
        allowed_bps=sorted(allowed),
        data_counts=dict(
            Counter(r["split"] for r in packed["rows"] if r["requested_bps"] in allowed)
        ),
        ocr_counts=protocol["counts"],
        packed_sha256=sha256(data / "packed-v1/manifest.json"),
        ocr_protocol_sha256=sha256(data / "evaluation-protocol/protocol.json"),
        comparison="Official direct baseline; official initialized training LR 1e-4; matched random initialized control LR 1e-3. Same views, crops, seed, epochs, objective and validation selection. Different LR recipes prevent attributing differences solely to initialization.",
        runs=[
            dict(name="official-finetuned", initialization="official", lr=0.0001),
            dict(name="scratch-control", initialization="scratch", lr=0.001),
        ],
        frozen_code={
            str(p.relative_to(root)): sha256(p)
            for p in code.rglob("*")
            if p.is_file() and "__pycache__" not in p.parts
        },
        scope="Single-seed exploratory offline text SR comparison; not live constrained-network evidence.",
    )
    atomic_json(root / "campaign.json", plan)
    (root / "runs").mkdir()
    (root / "logs").mkdir()
    print(
        json.dumps({k: v for k, v in plan.items() if k != "frozen_code"}, indent=2),
        flush=True,
    )


def worker(root, name, python):
    plan = read(root / "campaign.json")
    spec = next(r for r in plan["runs"] if r["name"] == name)
    for path, digest in plan["frozen_code"].items():
        if sha256(root / path) != digest:
            raise ValueError("Campaign code changed: " + path)
    output = root / "runs" / name
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "code") + ":" + str(root / "code/src"),
        OMP_NUM_THREADS="2",
    )
    train = [
        str(python),
        "-u",
        "-m",
        "benchmarks.super_resolution.text_corpus.train",
        "--packed",
        str(root / "data/packed-v1/manifest.json"),
        "--output",
        str(output),
        "--home",
        str(root / "assets"),
        "--epochs",
        str(plan["epochs"]),
        "--seed",
        "682",
        "--lr",
        str(spec["lr"]),
        "--initialization",
        spec["initialization"],
        "--exclude-bps",
        "128000",
        "--memory-fraction",
        "0.20",
        "--threads",
        "4",
    ]
    atomic_json(
        root / "logs" / (name + "-host.json"),
        dict(
            pid=os.getpid(),
            host=socket.gethostname(),
            cuda_visible_devices=env.get("CUDA_VISIBLE_DEVICES"),
            started=time.time(),
            command=train,
        ),
    )
    with (root / "logs" / (name + "-train.log")).open("xb") as log:
        result = subprocess.run(
            train, cwd=root / "code", env=env, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        atomic_json(
            root / "logs" / (name + "-exit.json"),
            dict(phase="training", exit_code=result.returncode, finished=time.time()),
        )
        raise SystemExit(result.returncode)
    command = [
        str(python),
        "-u",
        "-m",
        "benchmarks.super_resolution.text_corpus.evaluate",
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
    ]
    with (root / "logs" / (name + "-render.log")).open("xb") as log:
        result = subprocess.run(
            command, cwd=root / "code", env=env, stdout=log, stderr=subprocess.STDOUT
        )
    atomic_json(
        root / "logs" / (name + "-exit.json"),
        dict(phase="render", exit_code=result.returncode, finished=time.time()),
    )
    if result.returncode:
        raise SystemExit(result.returncode)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["prepare", "start", "worker"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--previous", type=Path)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--python", type=Path, default=Path(sys.executable))
    p.add_argument("--name")
    p.add_argument("--gpu-uuid", action="append")
    a = p.parse_args()
    if a.action == "prepare":
        prepare(a.root, a.previous, a.epochs)
    elif a.action == "worker":
        for name in a.name.split(","):
            worker(a.root, name, a.python)
    else:
        if (a.root / "LAUNCHED.json").exists():
            raise ValueError("Already launched")
        gpu_ids = a.gpu_uuid
        if not gpu_ids:
            raise ValueError("Specify the verified available GPU UUIDs")
        jobs = []
        runs = read(a.root / "campaign.json")["runs"]
        for index, gpu in enumerate(gpu_ids):
            name = ",".join(r["name"] for r in runs[index :: len(gpu_ids)])
            if not name:
                continue
            env = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES=gpu,
                PYTHONPATH=str(a.root / "code") + ":" + str(a.root / "code/src"),
            )
            with (a.root / "logs" / (name + "-worker.log")).open("xb") as log:
                child = subprocess.Popen(
                    [
                        "nice",
                        "-n",
                        "10",
                        str(a.python),
                        "-u",
                        "-m",
                        __package__ + ".transfer_campaign",
                        "worker",
                        "--root",
                        str(a.root),
                        "--name",
                        name,
                        "--python",
                        str(a.python),
                    ],
                    cwd=a.root / "code",
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            jobs.append(dict(name=name, pid=child.pid, gpu_uuid=gpu))
        atomic_json(a.root / "LAUNCHED.json", dict(jobs=jobs, started=time.time()))
        print(json.dumps(jobs), flush=True)


if __name__ == "__main__":
    main()
