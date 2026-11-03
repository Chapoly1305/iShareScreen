"""Finish explicitly authorized transfer and start alex training unattended."""

from pathlib import Path
import subprocess
import time
import json
from benchmarks.super_resolution.training_state import atomic_json


def main():
    project = Path("/home/jchen73/iShareScreen")
    root = (
        project
        / "benchmarks/super_resolution/experiments/codescan-720p-600k-420-v1-rtx4090"
    )
    # Avoid overlapping the initial in-flight rsync; this PID and command were inspected.
    while True:
        p = Path("/proc/2745834/cmdline")
        if not p.exists():
            break
        command = p.read_bytes()
        if b"rsync" not in command or b"codescan-720p-600k-420" not in command:
            break
        atomic_json(
            root / "ACTIVATION.json",
            dict(stage="waiting_initial_transfer", updated=time.time()),
        )
        time.sleep(15)
    remote = "rtx4090:/home/alex/iShareScreen/"
    atomic_json(
        root / "ACTIVATION.json",
        dict(stage="final_transfer_verification", updated=time.time()),
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            "--partial",
            "--exclude=raw/",
            str(
                project
                / "benchmarks/super_resolution/experiments/codescan-720p-600k-420-v1/data"
            )
            + "/",
            remote
            + "benchmarks/super_resolution/experiments/codescan-720p-600k-420-v1-rtx4090/data/",
        ],
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            str(root / "CAMPAIGN.json"),
            str(root / "README.md"),
            remote
            + "benchmarks/super_resolution/experiments/codescan-720p-600k-420-v1-rtx4090/",
        ],
        check=True,
    )
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "rtx4090",
            "systemctl --user enable --now isharescreen-codescan-alex.service",
        ],
        check=True,
    )
    atomic_json(
        root / "ACTIVATION.json", dict(stage="service_started", updated=time.time())
    )


if __name__ == "__main__":
    main()
