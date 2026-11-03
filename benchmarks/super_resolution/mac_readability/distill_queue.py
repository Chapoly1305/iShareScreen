"""Bounded local supervisor: wait for one live teacher PID, then run v4 training.

Never starts, restarts or terminates the teacher. A failed/missing prerequisite
stops this supervisor. Training performs its own GPU-occupancy check and smoke.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.super_resolution.training_state import atomic_json
from benchmarks.super_resolution.mac_readability.itsrn_probe import sha


def read(p):
    return json.loads(p.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--teacher-pid", type=int, required=True)
    a = p.parse_args()
    root = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    folder = root / "distill-queue-v1"
    folder.mkdir(exist_ok=True)
    lock = (folder / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (root / "training-v4").mkdir(exist_ok=True)

    def state(**kw):
        atomic_json(
            folder / "STATUS.json",
            dict(pid=os.getpid(), teacher_pid=a.teacher_pid, updated=time.time(), **kw),
        )

    proc = Path("/proc") / str(a.teacher_pid)
    token = None
    if proc.exists():
        command = (proc / "cmdline").read_bytes()
        assert b"benchmarks.super_resolution.mac_readability.itsrn_teacher" in command
        token = (proc / "stat").read_text().split()[21]
    deadline = time.monotonic() + 3600
    teacher = root / "teacher-itsrn-steady-v1"
    try:
        while True:
            status = read(teacher / "STATUS.json")
            assert status["pid"] == a.teacher_pid
            try:
                alive = (
                    token is not None
                    and (proc / "stat").read_text().split()[21] == token
                )
            except FileNotFoundError:
                alive = False
            if status["state"] == "completed" and not alive:
                data = read(teacher / "DATA.json")
                frozen = read(teacher / "FROZEN.json")
                assert data["complete"] and len(data["rows"]) == frozen["count"] == 512
                assert sha(teacher / "DATA.json") == frozen["data_sha256"]
                break
            if status["state"] == "failed" or not alive:
                raise RuntimeError(f"Teacher did not complete successfully: {status}")
            if time.monotonic() > deadline:
                raise TimeoutError("Teacher wait exceeded1hour; no process modified")
            state(
                state="waiting_for_teacher",
                completed=status.get("completed"),
                total=512,
            )
            time.sleep(10)
        plan = read(root / "TRAINING-PLAN-V5.json")
        runner = Path(__file__).with_name("train_distill.py")
        assert sha(runner) == plan["runner_sha256"]
        env = dict(
            os.environ, CUDA_VISIBLE_DEVICES="GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a"
        )
        env["ISS_SR_HOME"] = str(a.project / ".cache/super_resolution")
        env["MPLCONFIGDIR"] = str(a.project / ".tmp/matplotlib-distill")
        command = [
            sys.executable,
            "-m",
            "benchmarks.super_resolution.mac_readability.train_distill",
            "--project",
            str(a.project),
        ]
        state(state="starting_training", teacher_sha256=frozen["data_sha256"])
        with (folder / "training.log").open("a") as log:
            child = subprocess.Popen(
                command, cwd=a.project, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            state(
                state="training",
                child_pid=child.pid,
                teacher_sha256=frozen["data_sha256"],
            )
            code = child.wait()
        if code:
            raise RuntimeError(
                f"Training exited with status {code}; inspect training.log"
            )
        assert read(root / "training-v4/STATUS.json")["state"] == "training_completed"
        state(state="training_completed", child_pid=child.pid, ocr_pending=True)
    except BaseException as e:
        state(state="failed", error_type=type(e).__name__, error=str(e))
        raise


if __name__ == "__main__":
    main()
