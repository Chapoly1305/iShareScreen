"""Read-only lab monitor and H100 mirror. Does not send external messages."""

import argparse
import json
from pathlib import Path
import subprocess
import time

from benchmarks.super_resolution.training_state import atomic_json
from benchmarks.super_resolution.codescan_campaign.report import report


def heartbeat_age(root, status, now):
    if status.get("stage") not in (
        "preparing_smoke",
        "preparing_full",
        "smoke",
        "training",
    ):
        return None
    paths = [root / "STATUS.json"]
    if status["stage"].startswith("preparing"):
        paths.append(root / "data/PREPARATION-STATUS.json")
    elif status.get("model"):
        paths.append(
            root
            / ("smoke" if status["stage"] == "smoke" else "models")
            / status["model"]
            / "STATUS.json"
        )
    return now - max(p.stat().st_mtime for p in paths if p.exists())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    a = parser.parse_args()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan = (
        json.loads((root / "PARALLEL-PLAN.json").read_text())
        if (root / "PARALLEL-PLAN.json").exists()
        else None
    )
    remote = "/home/chen/iShareScreen/benchmarks/super_resolution/experiments/codescan-720p-600k-420-v1/"
    command = [
        "rsync",
        "-a",
        "--timeout=45",
        "--prune-empty-dirs",
        "--exclude=/data/raw/***",
        "--exclude=/data/pairs/***",
        "--exclude=/audit-prechecks/***",
        "--exclude=/code-snapshot/***",
        "--exclude=*.pt",
        "--exclude=*.log",
        "--include=*/",
        "--include=*.json",
        "--include=*.jsonl",
        "--include=*.csv",
        "--include=curves.png",
        "--include=curves.svg",
        "--exclude=*",
        "lab:" + remote,
        str(root) + "/",
    ]
    if plan:
        command[3:3] = [
            "--exclude=/models/" + model + "/***"
            for host, queue in plan["queues"].items()
            if host != "lab"
            for model in queue
        ]
    try:
        subprocess.run(command, check=True, timeout=80, capture_output=True)
        state = subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=10",
                "lab",
                "systemctl --user show isharescreen-codescan-v1.service -p ActiveState -p SubState -p NRestarts; nvidia-smi --query-gpu=uuid,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
        atomic_json(
            root / "MONITOR.json",
            dict(checked_at=time.time(), remote="chen@lab", state=state, ok=True),
        )
        status = (
            json.loads((root / "STATUS.json").read_text())
            if (root / "STATUS.json").exists()
            else {}
        )
        models = json.loads((root / "CAMPAIGN.json").read_text())["models"]
        rtx_status = {}
        rtx_state = ""
        if plan and plan.get("rtx4090_root"):
            rtx_root = Path(plan["rtx4090_root"])
            rtx_remote = "/home/alex/iShareScreen/benchmarks/super_resolution/experiments/codescan-720p-600k-420-v1-rtx4090/"
            sync = [
                "rsync",
                "-a",
                "--timeout=45",
                "--prune-empty-dirs",
                "--exclude=/data/***",
                "--exclude=/code-snapshot/***",
                "--exclude=*.pt",
                "--exclude=*.log",
                "--include=*/",
                "--include=*.json",
                "--include=*.jsonl",
                "--include=*.csv",
                "--include=curves.png",
                "--include=curves.svg",
                "--exclude=*",
                "rtx4090:" + rtx_remote,
                str(rtx_root) + "/",
            ]
            subprocess.run(sync, check=True, timeout=80, capture_output=True)
            rtx_state = subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    "-o",
                    "ConnectTimeout=10",
                    "rtx4090",
                    "systemctl --user show isharescreen-codescan-alex.service -p ActiveState -p SubState -p NRestarts",
                ],
                capture_output=True,
                text=True,
                timeout=20,
            ).stdout
            p = rtx_root / "STATUS.json"
            rtx_status = json.loads(p.read_text()) if p.exists() else {}
            age_rtx = heartbeat_age(rtx_root, rtx_status, time.time())
            if (
                age_rtx is not None
                and age_rtx > 1800
                and "ActiveState=active" in rtx_state
            ):
                restart = rtx_root / "MONITOR-LAST-RESTART.json"
                last = (
                    json.loads(restart.read_text()).get("time", 0)
                    if restart.exists()
                    else 0
                )
                if time.time() - last > 7200:
                    subprocess.run(
                        [
                            "ssh",
                            "-o",
                            "BatchMode=yes",
                            "rtx4090",
                            "systemctl --user restart isharescreen-codescan-alex.service",
                        ],
                        check=True,
                        timeout=45,
                    )
                    atomic_json(
                        restart,
                        dict(
                            time=time.time(), reason="alex4090 progress stale30 minutes"
                        ),
                    )
        h100_status = {}
        h100_state = ""
        if plan:
            h100_root = Path(plan["h100_root"])
            p = h100_root / "STATUS.json"
            h100_status = json.loads(p.read_text()) if p.exists() else {}
            extra_root = Path(str(h100_root) + "-extra")
            if (extra_root / "STATUS.json").exists():
                extra = json.loads((extra_root / "STATUS.json").read_text())
                atomic_json(root / "H100-EXTRA-STATUS.json", extra)
                h100_status = dict(h100_status, concurrent_efrlfn=extra)
                extra_age = heartbeat_age(extra_root, extra, time.time())
                extra_service = subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        "isharescreen-codescan-h100-extra.service",
                        "-p",
                        "ActiveState",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                if (
                    extra_age is not None
                    and extra_age > 1800
                    and "ActiveState=active" in extra_service
                ):
                    restart_record = extra_root / "MONITOR-LAST-RESTART.json"
                    last = (
                        json.loads(restart_record.read_text()).get("time", 0)
                        if restart_record.exists()
                        else 0
                    )
                    if time.time() - last > 7200:
                        subprocess.run(
                            [
                                "systemctl",
                                "--user",
                                "restart",
                                "isharescreen-codescan-h100-extra.service",
                            ],
                            check=True,
                            timeout=45,
                        )
                        atomic_json(
                            restart_record,
                            dict(
                                time=time.time(),
                                reason="Concurrent H100 worker stale for30 minutes",
                            ),
                        )
            gpu0_root = Path(str(h100_root) + "-gpu0")
            if (gpu0_root / "STATUS.json").exists():
                gpu0 = json.loads((gpu0_root / "STATUS.json").read_text())
                h100_status = dict(h100_status, safmn_gpu0=gpu0)
                atomic_json(root / "H100-GPU0-STATUS.json", gpu0)
                gpu0_age = heartbeat_age(gpu0_root, gpu0, time.time())
                gpu0_state = subprocess.run(
                    [
                        "systemctl",
                        "--user",
                        "show",
                        "isharescreen-codescan-h100-gpu0.service",
                        "-p",
                        "ActiveState",
                    ],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout
                if (
                    gpu0_age is not None
                    and gpu0_age > 1800
                    and "ActiveState=active" in gpu0_state
                ):
                    restart_record = gpu0_root / "MONITOR-LAST-RESTART.json"
                    last = (
                        json.loads(restart_record.read_text()).get("time", 0)
                        if restart_record.exists()
                        else 0
                    )
                    if time.time() - last > 7200:
                        subprocess.run(
                            [
                                "systemctl",
                                "--user",
                                "restart",
                                "isharescreen-codescan-h100-gpu0.service",
                            ],
                            check=True,
                            timeout=45,
                        )
                        atomic_json(
                            restart_record,
                            dict(
                                time=time.time(),
                                reason="GPU0 SAFMN progress stale for30 minutes",
                            ),
                        )
            h100_state = subprocess.run(
                [
                    "systemctl",
                    "--user",
                    "show",
                    "isharescreen-codescan-h100.service",
                    "-p",
                    "ActiveState",
                    "-p",
                    "SubState",
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            h100_age = heartbeat_age(h100_root, h100_status, time.time())
            if (
                h100_age is not None
                and h100_age > 1800
                and "ActiveState=active" in h100_state
            ):
                action = h100_root / "MONITOR-LAST-RESTART.json"
                previous = (
                    json.loads(action.read_text()).get("time", 0)
                    if action.exists()
                    else 0
                )
                if time.time() - previous > 7200:
                    subprocess.run(
                        [
                            "systemctl",
                            "--user",
                            "restart",
                            "isharescreen-codescan-h100.service",
                        ],
                        check=True,
                        timeout=45,
                    )
                    atomic_json(
                        action,
                        dict(
                            time=time.time(), reason="H100 progress stale for30 minutes"
                        ),
                    )
            completed = [
                model
                for model in models
                if (root / "models" / model / "COMPLETED.json").exists()
            ]
            atomic_json(
                root / "PARALLEL-STATUS.json",
                dict(
                    lab=status,
                    h100=h100_status,
                    h100_service=h100_state,
                    rtx4090=rtx_status,
                    rtx4090_service=rtx_state,
                    completed_models=completed,
                    total_models=len(models),
                    all_complete=len(completed) == len(models),
                    updated=time.time(),
                ),
            )
            if len(completed) == len(models):
                atomic_json(
                    root / "ALL-MODELS-COMPLETED.json",
                    dict(models=completed, epochs_each=100, updated=time.time()),
                )
        report(root, models)
        age = heartbeat_age(root, status, time.time())
        if age is not None and age > 1800 and "ActiveState=active" in state:
            action_path = root / "MONITOR-LAST-RESTART.json"
            previous = (
                json.loads(action_path.read_text()).get("time", 0)
                if action_path.exists()
                else 0
            )
            if time.time() - previous > 7200:
                action = dict(
                    time=time.time(),
                    reason="No stage/worker/data progress for30 minutes",
                    heartbeat_age_seconds=age,
                )
                # Only our explicitly named service; at most one stall restart in two hours.
                subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        "-o",
                        "ConnectTimeout=10",
                        "lab",
                        "systemctl --user restart isharescreen-codescan-v1.service",
                    ],
                    check=True,
                    timeout=45,
                )
                atomic_json(action_path, action)
                with (root / "MONITOR-ACTIONS.jsonl").open("a") as f:
                    f.write(json.dumps(action) + "\n")
        preparation = (
            json.loads((root / "data/PREPARATION-STATUS.json").read_text())
            if (root / "data/PREPARATION-STATUS.json").exists()
            else {}
        )
        lines = [
            "# CodeSCAN training campaign",
            "",
            f"Last checked: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
            "",
            "Queue: "
            + " → ".join(models)
            + ", 100 epochs each. CPU codec preparation and small smoke runs precede full training.",
            "",
            f"Stage: {status.get('stage', 'starting')}",
            "",
            "```",
            state.strip(),
            "```",
            "",
            f"Data preparation: {preparation}",
            "",
            "| Model | GPU | State | Epoch | Best epoch | Best Mac PSNR |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
        if plan:
            lines[4] = "Parallel allocation: " + "; ".join(
                host + "=" + ", ".join(queue) for host, queue in plan["queues"].items()
            )
            lines[6] = (
                f"Lab stage: {status.get('stage', 'starting')}; overall completion: {len(completed)}/{len(models)} models"
            )
        figures = []
        for key in models:
            folder = root / "models" / key
            p = folder / "STATUS.json"
            info = json.loads(p.read_text()) if p.exists() else {}
            done = folder / "COMPLETED.json"
            gpu = "H100" if plan and key in plan["queues"]["h100"] else "4090"
            if key == "safmn" and h100_status.get("safmn_gpu0"):
                gpu = "H100 GPU0"
            if plan and key in plan["queues"].get("rtx4090", []):
                gpu = "4090 alex"
            lines.append(
                f"| {key} | {gpu} | {'completed' if done.exists() else info.get('phase', 'queued')} | {info.get('epoch', '—')} | {info.get('best_epoch', '—')} | {info.get('best_mac_psnr', '—')} |"
            )
            if (folder / "curves.png").exists():
                figures.extend(["", f"![{key} curves]({folder}/curves.png)"])
        if plan:
            lines += [
                "",
                f"H100 stage: {h100_status.get('stage', 'starting')}",
                "",
                "```",
                h100_state.strip(),
                "```",
            ]
        if plan and plan.get("rtx4090_root"):
            lines += [
                "",
                f"alex4090 stage: {rtx_status.get('stage', 'starting')}",
                "",
                "```",
                rtx_state.strip(),
                "```",
            ]
        lines.extend(figures)
        lines.extend(
            [
                "",
                "CSV and SVG curves are saved beside each model. Checkpoints stay on their training host; metrics and plots consolidate here.",
                "",
                "FPS is batch1 GPU forward throughput measured on the labelled training GPU. Different GPUs do not form one comparable speed ranking. It excludes decoding, transfers, FHD resize and display, and does not establish Mac client speed.",
                "",
                "No chat push-notification integration is available. This monitor writes status and attention records; it does not send email or external messages.",
            ]
        )
        if (root / "COMPARISON.png").exists():
            lines.extend(["", f"![Model comparison curves]({root}/COMPARISON.png)"])
        (root / "STATUS.md").write_text("\n".join(lines) + "\n")
        failed = (
            status.get("stage") == "failed"
            or ("ActiveState=failed" in state)
            or h100_status.get("stage") == "failed"
            or ("ActiveState=failed" in h100_state)
            or h100_status.get("concurrent_efrlfn", {}).get("stage") == "failed"
            or h100_status.get("safmn_gpu0", {}).get("stage") == "failed"
            or rtx_status.get("stage") == "failed"
            or "ActiveState=failed" in rtx_state
        )
        if failed or (age is not None and age > 1800):
            (root / "NEEDS-ATTENTION.md").write_text(
                "Training pipeline has failed, is retrying, or has stale progress. Inspect FAILED.json, failures/, MONITOR-ACTIONS.jsonl and remote worker.log.\n"
            )
        elif (root / "NEEDS-ATTENTION.md").exists():
            (root / "NEEDS-ATTENTION.md").unlink()
    except Exception as exc:
        atomic_json(
            root / "MONITOR.json",
            dict(checked_at=time.time(), ok=False, error=str(exc)),
        )
        raise


if __name__ == "__main__":
    main()
