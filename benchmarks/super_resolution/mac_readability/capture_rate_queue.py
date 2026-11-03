"""Bounded actual Screen Sharing rate-headroom repeat experiment on Mac GUI."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    root = Path("/Users/alex/iShareScreen-experiments/text-readability-v2-20261102")
    out = root / "rate-headroom-v1"
    out.mkdir(exist_ok=False)
    result = dict(
        scope="ActualMac ScreenSharing virtual720,600Kbps process-local link; Mac decoder, not physicalWAN or4090 receiver",
        sequence=[600, 450, 300, 300, 450, 600],
        repeats=2,
        seconds=30,
        queue_ms=250,
        delay_ms=10,
        started=time.time(),
        complete=False,
        pid=os.getpid(),
        runs=[],
    )

    def save():
        temp = out / "STATUS.tmp"
        temp.write_text(json.dumps(result, indent=2) + "\n")
        temp.replace(out / "STATUS.json")

    save()
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-di", "-w", str(os.getpid())])
    try:
        for i, rate in enumerate(result["sequence"]):
            name = f"capture-headroom-{i + 1:02d}-{rate}k"
            path = root / name
            command = [
                sys.executable,
                str(root / "tools/probe.py"),
                "--output",
                str(path),
                "--seconds",
                "30",
                "--local-bind",
                "192.168.1.180",
                "--restrict-format",
                "--curtain",
                "--stimulus",
                "--snapshot-interval",
                "1",
                "--kbps",
                str(rate),
                "--link-kbps",
                "600",
                "--link-queue-ms",
                "250",
                "--link-delay-ms",
                "10",
            ]
            with (out / (name + ".log")).open("w") as log:
                process = subprocess.Popen(
                    command, stdout=log, stderr=subprocess.STDOUT
                )
                row = dict(
                    name=name,
                    requested_kbps=rate,
                    pid=process.pid,
                    started=time.time(),
                    state="running",
                    command=command,
                )
                result["runs"].append(row)
                save()
                try:
                    row["exit_code"] = process.wait(timeout=65)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    row["timeout"] = True
                row["finished"] = time.time()
                row["state"] = "finished"
                if (path / "STATUS.json").exists():
                    captured = json.loads((path / "STATUS.json").read_text())
                    row["capture_state"] = captured.get("state")
                    row["capture_error"] = captured.get("error")
                save()
            if (
                row.get("capture_error")
                and "locked" in str(row["capture_error"]).lower()
            ):
                break
            time.sleep(3)
        result.update(complete=len(result["runs"]) == 6, finished=time.time())
        save()
    finally:
        awake.terminate()
        awake.wait(timeout=5)


if __name__ == "__main__":
    main()
