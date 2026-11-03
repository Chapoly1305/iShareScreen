"""Two explicit H100 queues for the source-frozen Hopper campaign migration."""

import argparse
import os
from pathlib import Path
import socket
import subprocess
import time

from benchmarks.super_resolution.text_corpus import hopper
from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement.models import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    p.add_argument("--indices", required=True)
    args = p.parse_args()
    root = args.root
    plan = hopper.read(root / "campaign.json")
    runtime = root / "recovery-v1"
    metadata = root / "data/packed-v1/manifest.json"
    if hopper.identities(root) != plan["frozen_files"]:
        raise ValueError("Frozen code/assets/dataset identity differs")
    for index in map(int, args.indices.split(",")):
        spec = plan["runs"][index]
        name = spec["name"]
        output = root / "runs" / name
        host = dict(
            index=index,
            name=name,
            pid=os.getpid(),
            hostname=socket.gethostname(),
            gpu_uuid=os.environ["CUDA_VISIBLE_DEVICES"],
            started=time.time(),
            campaign_binding=fingerprint(plan),
        )
        atomic_json(runtime / (name + "-host.json"), host)
        env = dict(
            os.environ,
            PYTHONPATH=str(root / "code") + ":" + str(root / "code/src"),
            OMP_NUM_THREADS="4",
        )
        if index < 3:
            snapshot = hopper.read(runtime / "backups" / name / "SNAPSHOT.json")
            for file in [
                "protocol.json",
                "initial.pt",
                "best.pt",
                "last.pt",
                "progress.jsonl",
                "initial-validation.json",
            ]:
                if sha256(output / file) != snapshot["files"][file]:
                    raise ValueError("Run changed after validated snapshot: " + file)
            command = [
                str(args.python),
                "-u",
                str(runtime / "resume.py"),
                "--packed",
                str(metadata),
                "--output",
                str(output),
                "--home",
                str(root / "assets"),
                "--original-root",
                "/scratch/jchen73/isharescreen-text-corpus-v1.20261021",
                "--relocated-root",
                str(root),
            ]
        else:
            command = [
                str(args.python),
                "-u",
                "-m",
                "benchmarks.super_resolution.text_corpus.train",
                "--packed",
                str(metadata),
                "--output",
                str(output),
                "--home",
                str(root / "assets"),
                "--device",
                "cuda:0",
                "--epochs",
                "300",
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
        with (runtime / (name + "-train.log")).open("xb") as log:
            subprocess.run(
                command,
                cwd=root / "code",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        with (runtime / (name + "-render.log")).open("xb") as log:
            subprocess.run(
                [
                    str(args.python),
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
                ],
                cwd=root / "code",
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        atomic_json(
            root / "slurm" / (name + "-COMPLETED.json"),
            dict(
                host,
                training_receipt_sha256=sha256(output / "COMPLETED.json"),
                render_receipt_sha256=sha256(
                    root / "data/evaluations" / name / "rendered.json"
                ),
                runtime_amendment_sha256=sha256(root / "runtime-amendment.json"),
                finished=time.time(),
            ),
        )
        print("RUN COMPLETE", name, flush=True)


if __name__ == "__main__":
    main()
