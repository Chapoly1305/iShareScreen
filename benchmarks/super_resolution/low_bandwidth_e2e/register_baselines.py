"""Register and validate pinned historical x2 model weights for matched reruns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from benchmarks.super_resolution.text_corpus.scratch import build
from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement import candidates
from isharescreen.enhancement.models import sha256, project_home


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def run(root: Path, corpus: Path) -> None:
    target = root / "BASELINES.json"
    if target.exists():
        raise FileExistsError("Baseline registry already frozen")
    home = project_home()
    spec = candidates.specification("quicksrnet")
    official = home / "candidates/quicksrnet/weights.pth"
    expected = spec["files"]["weights.pth"]["sha256"]
    if sha256(official) != expected:
        raise ValueError("Official pretrained weight differs from pinned source")
    rows = [
        dict(
            id="official_quicksrnet_x2",
            kind="official_pretrained",
            checkpoint=str(official),
            checkpoint_sha256=expected,
            source_revision=spec["revision"],
            license_id=spec["license_id"],
        )
    ]
    sources = [
        ("existing_official_finetune_x2", root / "baselines/official-finetuned"),
        ("existing_scratch_control_100ep", root / "baselines/scratch-control"),
    ]
    historical = corpus / "hopper-results/runs"
    sources += [
        (f"existing_scratch_{name}_x2", historical / name)
        for name in (
            "l1-seed-682",
            "l1-seed-683",
            "l1-seed-684",
            "text-edge-seed-682",
            "text-edge-seed-683",
            "text-edge-seed-684",
        )
    ]
    for identifier, folder in sources:
        policy = read(folder / "protocol.json")
        receipt = read(folder / "COMPLETED.json")
        weight = folder / "best.pt"
        if sha256(weight) != receipt["best_sha256"]:
            raise ValueError(f"Checkpoint completion hash differs: {identifier}")
        if fingerprint(policy) != receipt["protocol_binding"]:
            raise ValueError(f"Checkpoint protocol differs: {identifier}")
        payload = torch.load(weight, map_location="cpu", weights_only=True)
        if payload["protocol_binding"] != receipt["protocol_binding"]:
            raise ValueError(f"Checkpoint embeds another protocol: {identifier}")
        if payload["epoch"] != receipt["best_epoch"]:
            raise ValueError(f"Checkpoint epoch differs: {identifier}")
        model, _ = build(home, policy["seed"])
        model.load_state_dict(payload["state_dict"], strict=True)
        rows.append(
            dict(
                id=identifier,
                kind="official_initialized_text_training"
                if "finetune" in identifier
                else "scratch",
                checkpoint=str(weight.resolve()),
                checkpoint_sha256=sha256(weight),
                protocol=str((folder / "protocol.json").resolve()),
                protocol_sha256=sha256(folder / "protocol.json"),
                completion_receipt=str((folder / "COMPLETED.json").resolve()),
                completion_receipt_sha256=sha256(folder / "COMPLETED.json"),
                best_epoch=payload["epoch"],
                seed=policy["seed"],
                training_budget_epochs=policy.get("epochs"),
                initialization=policy["initialization"],
                strict_architecture_load_passed=True,
            )
        )
    atomic_json(
        target,
        dict(
            schema="isharescreen_low_bandwidth_baselines_v1",
            models=rows,
            runner_sha256=sha256(Path(__file__)),
            official_source_revision=spec["revision"],
            scope="Source-hash-bound weights; no claim of performance from checkpoint validity.",
            finished=time.time(),
        ),
    )
    print("REGISTERED BASELINES", len(rows), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.corpus.resolve())


if __name__ == "__main__":
    main()
