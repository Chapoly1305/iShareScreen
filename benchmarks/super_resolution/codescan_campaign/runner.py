"""Persistent sequential campaign. Successful stages are hash-bound and resumable."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import importlib
import importlib.util

from benchmarks.super_resolution.codescan_campaign.data import (
    prepare,
    read,
    source_rows,
)
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def assigned_models(models, plan, host):
    owners = plan["queues"]
    flattened = [model for queue in owners.values() for model in queue]
    assert len(flattened) == len(set(flattened)) and set(flattened) == set(models), (
        "Invalid or overlapping host allocation"
    )
    assert host in owners, "Host not present in allocation"
    return owners[host]


def verify_prepared(data_root, rows):
    data = read(data_root / "DATA.json")
    small = read(data_root / "SMOKE-DATA.json")
    assert data["complete"] and data["counts"] == {"train": 9600, "validation": 1200}
    assert data["recipe_sha256"] == sha256(data_root / "RECIPE.json")
    expected = {r["id"]: r for r in rows}
    assert {r["id"] for r in data["rows"]} == set(expected)
    assert not any(data["cross_split_overlap"].values())
    for row in data["rows"]:
        assert (
            row["split"] == expected[row["id"]]["split"]
            and row["group_id"] == expected[row["id"]]["group_id"]
        )
        for role in ("lr", "hr"):
            assert sha256(data_root / row[role]) == row[role + "_sha256"]
    assert {r["id"] for r in small["rows"]} <= set(expected) and small[
        "binding"
    ] == data["binding"]
    return data_root / "SMOKE-DATA.json", data_root / "DATA.json"


def event(root, kind, **details):
    record = dict(time=time.time(), kind=kind, **details)
    with (root / "EVENTS.jsonl").open("a") as f:
        f.write(json.dumps(record) + "\n")
    print("EVENT", json.dumps(record), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path.cwd())
    parser.add_argument(
        "--stage", choices=["all", "smoke", "full", "prepared"], default="all"
    )
    parser.add_argument("--model", help="Internal isolated training subprocess only")
    parser.add_argument("--data", type=Path)
    parser.add_argument("--smoke-job", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stop-after-epoch", type=int)
    a = parser.parse_args()
    root = a.root.resolve()
    project = a.project.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = read(root / "CAMPAIGN.json")
    hold = read(project / "benchmarks/super_resolution/training-hold.json")
    assert hold["status"] == "released", "Explicit training hold remains active"
    if a.model:
        if config.get("target_host") == "h100":
            from benchmarks.super_resolution.codescan_campaign.gpu_guard import (
                main as check_gpu,
            )

            check_gpu()
        trainer = importlib.import_module(
            config.get(
                "trainer_module", "benchmarks.super_resolution.codescan_campaign.train"
            )
        )
        if config.get("gpu_concurrency_lock"):
            from benchmarks.super_resolution.codescan_campaign.concurrent import install

            install(trainer, config["gpu_concurrency_lock"])
        if config.get("plot_gpu_label"):
            from benchmarks.super_resolution.codescan_campaign.report import (
                device_curves,
            )

            trainer.curves = lambda out, history, validations: device_curves(
                out, history, validations, config["plot_gpu_label"]
            )
        trainer.run(
            project, root, a.data, a.model, a.epochs, a.smoke_job, a.stop_after_epoch
        )
        return
    queue = config["models"]
    if (root / "PARALLEL-PLAN.json").exists():
        plan = read(root / "PARALLEL-PLAN.json")
        gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        host = plan["gpu_hosts"][gpu]
        queue = assigned_models(config["models"], plan, host)
    with (root / "campaign.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if (root / "FAILED.json").exists():
                failures = root / "failures"
                failures.mkdir(exist_ok=True)
                (root / "FAILED.json").replace(failures / f"{time.time_ns()}.json")
            rows = source_rows(project)
            data_root = root / "data"

            def status(stage, **kwargs):
                atomic_json(
                    root / "STATUS.json",
                    dict(stage=stage, updated=time.time(), pid=os.getpid(), **kwargs),
                )

            prepared_small = prepared_full = None
            if a.stage == "prepared":
                status("verifying_prepared_inputs")
                prepared_small, prepared_full = verify_prepared(data_root, rows)
                atomic_json(
                    root / "INPUT-COPY-VERIFIED.json",
                    dict(
                        passed=True,
                        samples=10800,
                        actual_files_rehashed=21600,
                        data_sha256=sha256(prepared_full),
                        recipe_sha256=sha256(data_root / "RECIPE.json"),
                        finished=time.time(),
                    ),
                )

            def job(model, data, smoke):
                command = [
                    sys.executable,
                    "-m",
                    __name__.replace(
                        "__main__",
                        "benchmarks.super_resolution.codescan_campaign.runner",
                    ),
                    "--root",
                    str(root),
                    "--project",
                    str(project),
                    "--model",
                    model,
                    "--data",
                    str(data),
                    "--epochs",
                    "2" if smoke else "100",
                ]
                if smoke:
                    command += ["--smoke-job"]
                status("smoke" if smoke else "training", model=model)
                folder = root / ("smoke" if smoke else "models") / model
                folder.mkdir(parents=True, exist_ok=True)
                event(root, "model_started", model=model, smoke=smoke)
                with (folder / "worker.log").open("a") as log:
                    if (
                        smoke
                        and not (folder / "INTERRUPTION-TEST.json").exists()
                        and not (folder / "COMPLETED.json").exists()
                    ):
                        subprocess.run(
                            command + ["--stop-after-epoch", "1"],
                            cwd=project,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
                    subprocess.run(
                        command,
                        cwd=project,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                done = read(folder / "COMPLETED.json")
                assert done["epochs"] == (2 if smoke else 100)
                if smoke:
                    assert (
                        done["steps"] == 8
                        and read(folder / "INTERRUPTION-TEST.json")["step"] == 4
                    )
                event(root, "model_completed", model=model, smoke=smoke, **done)

            if a.stage in ("all", "smoke", "prepared"):
                status("preparing_smoke")
                small = prepared_small or prepare(
                    project,
                    data_root,
                    rows,
                    workers=config["encode_workers"],
                    smoke=True,
                )
                for model in queue:
                    job(model, small, True)
                # Explicit recovery gate: completed jobs may be reopened without an update or checkpoint change.
                before = {m: sha256(root / "smoke" / m / "last.pt") for m in queue}
                for model in queue:
                    job(model, small, True)
                assert before == {
                    m: sha256(root / "smoke" / m / "last.pt") for m in queue
                }
                atomic_json(
                    root / "SMOKE-PASSED.json",
                    dict(
                        models=queue,
                        epochs_each=2,
                        train_samples=128,
                        validation_samples=16,
                        train_runner_sha256=sha256(
                            Path(
                                importlib.util.find_spec(
                                    config.get(
                                        "trainer_module",
                                        "benchmarks.super_resolution.codescan_campaign.train",
                                    )
                                ).origin
                            )
                        ),
                        checkpoint_interruption_and_resume=True,
                        official_epoch0_parity=True,
                        completed_stage_reopen_idempotent=True,
                        finished=time.time(),
                    ),
                )
                event(root, "smoke_passed", models=queue)
            if a.stage in ("all", "full", "prepared"):
                assert (root / "SMOKE-PASSED.json").exists(), (
                    "No automatic full training without smoke gate"
                )
                status("preparing_full")
                full = prepared_full or prepare(
                    project, data_root, rows, workers=config["encode_workers"]
                )
                for model in queue:
                    job(model, full, False)
                status("completed", models=queue)
                event(root, "campaign_completed", models=queue)
                atomic_json(
                    root / "COMPLETED.json",
                    dict(models=queue, epochs_each=100, finished=time.time()),
                )
        except Exception as exc:
            atomic_json(
                root / "FAILED.json",
                dict(
                    type=type(exc).__name__,
                    message=str(exc),
                    traceback=traceback.format_exc(),
                    time=time.time(),
                ),
            )
            atomic_json(
                root / "STATUS.json",
                dict(stage="failed", error=str(exc), updated=time.time()),
            )
            event(root, "failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
