"""Mirror immutable completed validation images and score frozen development OCR.

Only pulls from the already authorized lab host. Never starts or stops training.
Selection is deferred until the complete ablation queue has been assessed.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from benchmarks.super_resolution.mac_readability.quality import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    p.add_argument("--hours", type=float, default=4)
    p.add_argument("--workers", type=int, default=12)
    a = p.parse_args()
    root = (
        a.project
        / "benchmarks/super_resolution/experiments/mac-readability-v2-20261102"
    )
    train = root / "training-v1"
    out = root / "training-ocr-v1"
    out.mkdir(exist_ok=True)
    lock = (out / "watch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    freeze = json.loads((root / "COMPARATOR-FROZEN.json").read_text())
    deadline = time.monotonic() + a.hours * 3600

    def status(**kw):
        save(out / "STATUS.json", dict(pid=os.getpid(), updated=time.time(), **kw))

    status(
        state="started",
        script_sha256=sha(Path(__file__)),
        comparator_sha256=sha(root / "COMPARATOR-FROZEN.json"),
    )
    while time.monotonic() < deadline:
        command = [
            "rsync",
            "-a",
            "--exclude=*.pt",
            "--exclude=queue.lock",
            "-e",
            "ssh -o BatchMode=yes -o ConnectTimeout=8",
            "lab:/home/chen/iShareScreen/benchmarks/super_resolution/experiments/mac-readability-v2-20261102/training-v1/",
            str(train) + "/",
        ]
        subprocess.run(command, check=True, timeout=180)
        pending = []
        rank = []
        for index in sorted(train.glob("models/*/validation-*/RESULTS.json")):
            j = json.loads(index.read_text())
            if not j.get("complete"):
                continue
            if j["step"] == 0 and j["arm"] != "matched_codec":
                continue
            target = out / j["arm"] / f"step-{j['step']:06d}"
            if not (target / "SUMMARY.json").exists():
                pending.append((index, target, j))
                continue
            summary = json.loads((target / "SUMMARY.json").read_text())
            assert len(summary) == 1
            s = summary[0]
            hard = [
                g["cer"] for g in s["groups"] if g["group"] in freeze["hard_groups"]
            ]
            assert len(hard) == len(freeze["hard_groups"])
            rank.append(
                dict(
                    arm=j["arm"],
                    step=j["step"],
                    hard_macro_cer=sum(hard) / len(hard),
                    macro_cer=s["macro_cer"],
                    exact_lines=s["exact_units"],
                    lines=s["units"],
                    digits=s["digits"],
                    digit_edits=s["digit_edits"],
                    punctuation=s["punctuation"],
                    punctuation_edits=s["punctuation_edits"],
                    groups=s["groups"],
                    results_sha256=sha(index),
                    ocr_summary_sha256=sha(target / "SUMMARY.json"),
                )
            )
        rank.sort(
            key=lambda r: (r["hard_macro_cer"], r["macro_cer"], r["arm"], r["step"])
        )
        save(
            out / "CURVES.json",
            dict(
                complete=False,
                comparator=freeze,
                checkpoints=rank,
                test_used=False,
                selection="pending complete queue",
            ),
        )
        if pending:
            index, target, j = pending[0]
            status(
                state="scoring",
                arm=j["arm"],
                step=j["step"],
                scored=len(rank),
                remaining_available=len(pending),
            )
            command = [
                sys.executable,
                "-m",
                "benchmarks.super_resolution.mac_readability.score_lines",
                "--source",
                str(
                    a.project
                    / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/reference"
                ),
                "--index",
                str(index),
                "--output",
                str(target),
                "--tessdata",
                str(a.project / "tools-runtime/tessdata"),
                "--frozen-ocr",
                str(root / "dense-clean-v1/PROTOCOL.json"),
                "--workers",
                str(a.workers),
            ]
            subprocess.run(command, check=True)
            continue
        state = json.loads((train / "STATUS.json").read_text())
        if state["state"] == "training_completed":
            assert len(rank) == 25, (
                len(rank),
                "Expected24 trained snapshots andone initialization parity control",
            )
            save(
                out / "CURVES.json",
                dict(
                    complete=True,
                    comparator=freeze,
                    checkpoints=rank,
                    test_used=False,
                    selection="All25 checkpoints scored; regression, bootstrap and latency review still required",
                ),
            )
            status(state="scoring_complete", scored=len(rank))
            return
        if state["state"] == "failed":
            raise RuntimeError(state)
        status(state="waiting_for_training", scored=len(rank))
        time.sleep(30)
    status(
        state="time_limit",
        note="No GPU process modified; resume this scoring watcher if needed",
    )


if __name__ == "__main__":
    main()
