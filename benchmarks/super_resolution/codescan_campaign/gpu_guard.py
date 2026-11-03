"""Refuse to launch on an H100 device already running a compute process."""

import os
import subprocess
import sys


def main():
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    rows = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
        text=True,
    )
    for line in rows.splitlines():
        gpu, pid = [s.strip() for s in line.split(",")]
        if gpu == uuid:
            owner = subprocess.check_output(
                ["ps", "-o", "user=", "-p", pid], text=True
            ).strip()
            import getpass
            from pathlib import Path

            command = Path("/proc") / pid / "cmdline"
            text = command.read_bytes().replace(b"\0", b" ").decode()
            allowed = os.environ.get("ISS_CONCURRENT_CAMPAIGN")
            if (
                allowed
                and owner == getpass.getuser()
                and "codescan_campaign.runner" in text
                and allowed in text
            ):
                print("Sharing GPU with authorized campaign PID", pid, flush=True)
                continue
            raise SystemExit(f"Refusing occupied GPU {uuid}: PID {pid}, owner {owner}")
    print("Verified GPU free:", uuid, flush=True)


if __name__ == "__main__":
    main()
