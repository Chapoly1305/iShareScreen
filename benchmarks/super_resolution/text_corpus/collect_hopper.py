"""Collect completed Hopper runs and score their frozen OCR cases locally.

This process never starts, retries or modifies training. Each immutable result
is copied through the already authorized H100->Hopper SSH route. A transport
failure is recorded and stops collection rather than resubmitting any job.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import stat
import time
from types import SimpleNamespace

import numpy as np

from benchmarks.super_resolution.text_corpus import remote
from benchmarks.super_resolution.text_corpus.evaluate import score
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def execute(client, command):
    _, stdout, stderr = client.exec_command(command, timeout=60)
    output = stdout.read().decode()
    error = stderr.read().decode()
    status = stdout.channel.recv_exit_status()
    if status:
        raise RuntimeError(f"Remote command failed ({status}): {error[-2000:]}")
    return output


def hopper(client, command):
    if getattr(client, "text_direct", False):
        return execute(client, command)
    return execute(
        client,
        "ssh -o BatchMode=yes -o ConnectTimeout=15 hopper.orc.gmu.edu "
        + shlex.quote(command),
    )


def copy_tree(sftp, source, target):
    target.mkdir(parents=True, exist_ok=True)
    for item in sftp.listdir_attr(source):
        name = item.filename
        if "/" in name or name in (".", ".."):
            raise ValueError("Unsafe remote filename")
        path = target / name
        origin = source + "/" + name
        if stat.S_ISDIR(item.st_mode):
            copy_tree(sftp, origin, path)
        elif stat.S_ISREG(item.st_mode):
            if path.exists():
                raise ValueError("Result destination already exists: " + str(path))
            temp = path.with_name(path.name + ".download")
            sftp.get(origin, str(temp))
            if temp.stat().st_size != item.st_size:
                raise ValueError("Truncated result file")
            temp.replace(path)
            os.utime(path, (item.st_atime, item.st_mtime))
        else:
            raise ValueError("Unexpected nonregular result path")


def run(args):
    root = args.root.resolve()
    out = root / "hopper-results"
    if out.exists():
        raise ValueError(
            "Collector requires a fresh result directory; inspect prior partial results"
        )
    out.mkdir()
    client = remote.connect(args)
    finished = []
    failed = {}
    started = time.time()
    client.text_direct = args.direct

    def status(phase, **extra):
        atomic_json(
            out / "status.json",
            dict(
                phase=phase,
                completed=finished,
                failed_runs=failed,
                pid=os.getpid(),
                updated=time.time(),
                **extra,
            ),
        )

    try:
        plan = json.loads(
            hopper(client, "cat " + shlex.quote(args.hopper_root + "/campaign.json"))
        )
        submission = json.loads(
            hopper(client, "cat " + shlex.quote(args.hopper_root + "/submission.json"))
        )
        if args.direct:
            amendment = json.loads(
                hopper(
                    client,
                    "cat " + shlex.quote(args.hopper_root + "/runtime-amendment.json"),
                )
            )
            atomic_json(out / "runtime-amendment.json", amendment)
        if args.array_job:
            amendment = json.loads(
                hopper(
                    client,
                    "cat " + shlex.quote(args.hopper_root + "/runtime-amendment.json"),
                )
            )
            if (
                amendment["replacement_array_job"] != args.array_job
                or amendment["original_array_job"] != submission["array_job"]
            ):
                raise ValueError("Replacement job mapping differs")
            atomic_json(out / "runtime-amendment.json", amendment)
            submission = dict(
                submission,
                original_array_job=submission["array_job"],
                array_job=args.array_job,
            )
        if submission["state"] != "submitted":
            raise ValueError("Campaign submission did not complete")
        atomic_json(out / "campaign.json", plan)
        atomic_json(out / "submission.json", submission)
        for name in [
            "dataset-views-v3.json",
            "hevc-v2/pairs.json",
            "evaluation-protocol/protocol.json",
            "evaluation-protocol/source-compatibility.json",
        ]:
            if sha256(root / name) != plan["frozen_files"]["data/" + name]:
                raise ValueError("Local evaluation sources differ from Hopper")
        while len(finished) + len(failed) < len(plan["runs"]):
            if time.time() - started > args.timeout_hours * 3600:
                raise TimeoutError(
                    "Collection deadline exceeded; inspect Slurm for unfinished jobs"
                )
            names = hopper(
                client,
                "find "
                + shlex.quote(args.hopper_root + "/slurm")
                + ' -maxdepth 1 -name "*-COMPLETED.json" -printf "%f\\n"',
            ).splitlines()
            for spec in plan["runs"]:
                name = spec["name"]
                if name in finished or name + "-COMPLETED.json" not in names:
                    continue
                status("collecting", current=name)
                staging = args.staging_root + "/" + name
                if not args.direct:
                    execute(
                        client,
                        "mkdir -p "
                        + shlex.quote(staging + "/training")
                        + " "
                        + shlex.quote(staging + "/evaluation"),
                    )
                    for source, destination in [
                        ("/runs/" + name + "/", staging + "/training/"),
                        ("/data/evaluations/" + name + "/", staging + "/evaluation/"),
                    ]:
                        execute(
                            client,
                            "rsync -a -- "
                            + shlex.quote(
                                "hopper.orc.gmu.edu:" + args.hopper_root + source
                            )
                            + " "
                            + shlex.quote(destination),
                        )
                training = out / "runs" / name
                predictions = root / "evaluations" / name
                with client.open_sftp() as sftp:
                    copy_tree(
                        sftp,
                        args.hopper_root + "/runs/" + name
                        if args.direct
                        else staging + "/training",
                        training,
                    )
                    copy_tree(
                        sftp,
                        args.hopper_root + "/data/evaluations/" + name
                        if args.direct
                        else staging + "/evaluation",
                        predictions,
                    )
                rendered = json.loads((predictions / "rendered.json").read_text())
                if rendered["training_policy_sha256"] != sha256(
                    training / "protocol.json"
                ) or rendered["training_receipt_sha256"] != sha256(
                    training / "COMPLETED.json"
                ):
                    raise ValueError("Prediction/training provenance differs")
                score(
                    SimpleNamespace(root=root, output=predictions, workers=args.workers)
                )
                finished.append(name)
                status("waiting")
                print("COLLECTED", name, flush=True)
            accounting = (
                ""
                if args.direct
                else hopper(
                    client,
                    "sacct -n -X -P -j "
                    + shlex.quote(submission["array_job"])
                    + " --format=JobID%40,State%40",
                )
            )
            states = {
                line.split("|")[0].strip(): line.split("|")[1]
                .strip()
                .split()[0]
                .split("+")[0]
                for line in accounting.splitlines()
                if "|" in line and line.split("|")[1].strip()
            }
            failures = {
                "FAILED",
                "CANCELLED",
                "OUT_OF_MEMORY",
                "TIMEOUT",
                "NODE_FAIL",
                "PREEMPTED",
                "BOOT_FAIL",
                "DEADLINE",
                "REVOKED",
            }
            if args.direct:
                with client.open_sftp() as sftp:
                    for worker, indices in [(0, [0, 1, 2, 5]), (1, [3, 4])]:
                        try:
                            with sftp.file(
                                args.hopper_root
                                + f"/recovery-v1/worker-{worker}.exit.json"
                            ) as stream:
                                result = json.loads(stream.read())
                        except FileNotFoundError:
                            continue
                        if result["exit_code"]:
                            for index in indices:
                                name = plan["runs"][index]["name"]
                                if name not in finished:
                                    failed[name] = "H100 worker exited " + str(
                                        result["exit_code"]
                                    )
            for index, spec in enumerate(plan["runs"]):
                state = states.get(
                    submission["array_job"] + "_" + str(index),
                    states.get(submission["array_job"]),
                )
                if spec["name"] not in finished and state in failures:
                    failed[spec["name"]] = state
            if len(finished) + len(failed) < len(plan["runs"]):
                status("waiting")
                time.sleep(args.poll_seconds)
        if failed:
            raise RuntimeError(
                "Some Slurm tasks failed; successful per-run OCR results were retained: "
                + str(failed)
            )
        rows = []
        for spec in plan["runs"]:
            summary = json.loads(
                (root / "evaluations" / spec["name"] / "summary.json").read_text()
            )
            rows.extend(
                dict(
                    run=spec["name"],
                    seed=spec["seed"],
                    loss="l1" if spec["text_weight"] == 0 else "text-edge",
                    **row,
                )
                for row in summary
            )
        atomic_json(out / "all-run-scores.json", rows)
        lines = [
            "# Hopper text-loss ablation",
            "",
            "All six fixed-budget runs are reported. Each checkpoint was selected by the same fixed validation pixel/text/edge proxy. Test OCR did not select a seed.",
            "",
            "| Loss | Split | Trained mean line CER %, mean +/- seed SD | Trained equal-source CER %, mean +/- seed SD |",
            "|---|---|---:|---:|",
        ]
        aggregates = []
        for loss in ("l1", "text-edge"):
            for split in ("validation", "test"):
                values = [
                    r
                    for r in rows
                    if r["loss"] == loss
                    and r["split"] == split
                    and r["method"] == "trained"
                ]
                if len(values) != 3:
                    raise ValueError("Missing seed in fixed campaign")
                a = np.array([r["mean_line_cer"] for r in values]) * 100
                b = np.array([r["equal_source_group_cer"] for r in values]) * 100
                lines.append(
                    f"| {loss} | {split} | {a.mean():.3f} +/- {a.std(ddof=1):.3f} | {b.mean():.3f} +/- {b.std(ddof=1):.3f} |"
                )
                aggregates.append(
                    dict(
                        loss=loss,
                        split=split,
                        seeds=3,
                        mean_line_cer_percent_mean=float(a.mean()),
                        mean_line_cer_percent_seed_sd=float(a.std(ddof=1)),
                        equal_source_cer_percent_mean=float(b.mean()),
                        equal_source_cer_percent_seed_sd=float(b.std(ddof=1)),
                    )
                )
        lines += [
            "",
            "Seed spread is not a confidence interval over documents. Per-run reports retain clean-reference, interpolation, pretrained and initial-state baselines, digits/punctuation and exact predictions. These 96 test lines do not establish live-network usability.",
            "",
        ]
        (out / "REPORT.md").write_text("\n".join(lines))
        atomic_json(out / "aggregate.json", aggregates)
        atomic_json(
            out / "COMPLETED.json",
            dict(
                runs=finished,
                all_run_scores_sha256=sha256(out / "all-run-scores.json"),
                finished=time.time(),
            ),
        )
        status("completed")
    except BaseException as error:
        status("failed", error=repr(error))
        raise
    finally:
        client.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--hopper-root", required=True)
    p.add_argument("--staging-root", required=True)
    p.add_argument("--host", default="10.193.124.20")
    p.add_argument("--user", default="jchen73")
    p.add_argument("--key", type=Path, default=Path.home() / ".ssh/id_ed25519")
    p.add_argument("--known-hosts", type=Path, default=Path.home() / ".ssh/known_hosts")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--timeout-hours", type=float, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--array-job",
        help="Explicit runtime-amendment replacement array, preserving the original campaign",
    )
    p.add_argument(
        "--direct",
        action="store_true",
        help="Collect an explicit migrated H100 execution directly, without Slurm polling",
    )
    args = p.parse_args()
    if args.poll_seconds < 15 or args.timeout_hours <= 0:
        raise ValueError("Invalid collection polling/deadline")
    run(args)


if __name__ == "__main__":
    main()
