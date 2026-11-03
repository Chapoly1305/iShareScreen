"""Explicit user exception for a known idle resident, never stop another process."""

import os
import subprocess


def main():
    uuid = "GPU-3bf9ecb9-5729-7ad7-1a71-197d4c00415e"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == uuid
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    for line in rows.splitlines():
        gpu, pid = [s.strip() for s in line.split(",")]
        if gpu != uuid:
            continue
        owner = subprocess.check_output(
            ["ps", "-p", pid, "-o", "user="], text=True
        ).strip()
        assert pid == "2822608" and owner == "xli62", (
            f"Unrecognized GPU0 process {pid} {owner}"
        )
    load = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            uuid,
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    assert int(load.strip()) <= 5, "GPU0 shows active computation; wait"
    print(
        "GPU0 user-authorized idle-resident exception checked; no process stopped",
        flush=True,
    )


if __name__ == "__main__":
    main()
