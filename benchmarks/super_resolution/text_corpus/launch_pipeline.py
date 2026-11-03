"""Launch exactly one background training pipeline in an isolated task root."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    args = p.parse_args()
    root = args.root.resolve()
    if (root / "pipeline.pid").exists():
        raise ValueError(
            "A pipeline was already launched here; inspect its live process and logs, never retry implicitly"
        )
    for name in [
        "data/hevc-v2/pairs.json",
        "data/quality-audit-v3.json",
        "data/evaluation-protocol/source-compatibility.json",
    ]:
        if not (root / name).is_file():
            raise ValueError("Missing final training input: " + name)
    script = Path(__file__).with_name("remote_pipeline.sh")
    log = (root / "pipeline.log").open("wb")
    process = subprocess.Popen(
        [
            "flock",
            "--nonblock",
            str(root / "worker.lock"),
            "bash",
            str(script),
            str(root),
            str(args.python),
        ],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    (root / "pipeline.pid").write_text(str(process.pid) + "\n")
    (root / "launch.json").write_text(
        json.dumps(
            dict(
                pid=process.pid,
                started=time.time(),
                root=str(root),
                python=str(args.python),
                gpu_uuid="GPU-3bf9ecb9-5729-7ad7-1a71-197d4c00415e",
                stages=[
                    "pack",
                    "100-epoch scratch training",
                    "frozen OCR-region rendering",
                ],
            ),
            indent=2,
        )
    )
    print("PIPELINE_PID", process.pid, flush=True)


if __name__ == "__main__":
    main()
