"""Pause each original worker briefly and preserve a consistent epoch boundary."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

import torch

from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement.models import sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    if (root / "recovery-v1/SNAPSHOTS_READY.json").exists():
        raise ValueError("Snapshots already captured; never pause the old PIDs again")
    workers = [
        ("l1-seed-682", "gpu031", 764991),
        ("l1-seed-683", "gpu029", 280928),
        ("l1-seed-684", "gpu007", 3614064),
    ]
    paused = []
    try:
        for name, node, pid in workers:
            output = root / "runs" / name
            backup = root / "recovery-v1/backups" / name
            for attempt in range(5):
                control = f'test "$(stat -c %U /proc/{pid})" = jchen73 && tr "\\0" " " < /proc/{pid}/cmdline | grep -F -- {str(output)!r} >/dev/null && kill -STOP {pid}'
                subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", node, control], check=True
                )
                paused.append((node, pid))
                time.sleep(0.3)
                candidate = backup.with_name(name + f"-attempt-{attempt}")
                shutil.copytree(output, candidate)
                policy = json.loads((candidate / "protocol.json").read_text())
                binding = fingerprint(policy)
                last = torch.load(
                    candidate / "last.pt", map_location="cpu", weights_only=True
                )
                best = torch.load(
                    candidate / "best.pt", map_location="cpu", weights_only=True
                )
                records = [
                    json.loads(line)
                    for line in (candidate / "progress.jsonl").read_text().splitlines()
                    if line.strip()
                ]
                if (
                    records[-1]["epoch"] == last["epoch"]
                    and best["epoch"] <= last["epoch"]
                    and last["step"] == last["epoch"] * 253
                    and last["scheduler"]["last_epoch"] == last["epoch"]
                    and last["protocol_binding"] == best["protocol_binding"] == binding
                ):
                    candidate.rename(backup)
                    files = {
                        str(file.relative_to(backup)): sha256(file)
                        for file in backup.rglob("*")
                        if file.is_file()
                    }
                    atomic_json(
                        backup / "SNAPSHOT.json",
                        dict(
                            name=name,
                            node=node,
                            pid=pid,
                            epoch=last["epoch"],
                            step=last["step"],
                            files=files,
                            paused=True,
                            created=time.time(),
                            policy_binding=binding,
                        ),
                    )
                    print(
                        "SNAPSHOT VERIFIED AND PAUSED",
                        name,
                        last["epoch"],
                        last["step"],
                        flush=True,
                    )
                    break
                subprocess.run(
                    ["ssh", "-o", "BatchMode=yes", node, f"kill -CONT {pid}"],
                    check=True,
                )
                paused.remove((node, pid))
                time.sleep(2)
            else:
                raise ValueError("Could not capture a consistent epoch boundary")
        atomic_json(
            root / "recovery-v1/SNAPSHOTS_READY.json",
            dict(workers=workers, paused=paused, created=time.time()),
        )
    except BaseException:
        for node, pid in paused:
            subprocess.run(["ssh", "-o", "BatchMode=yes", node, f"kill -CONT {pid}"])
        raise


if __name__ == "__main__":
    main()
