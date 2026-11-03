"""Build an isolated H100 replica from frozen Hopper code and verified data."""

import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    root = args.root
    old = "/scratch/jchen73/isharescreen-text-corpus-v1.20261021"
    original = Path("/home/jchen73/isharescreen-text-corpus-v1.20261021")
    if (root / "code").exists():
        raise ValueError("H100 replica already exists")
    subprocess.run(
        [
            "rsync",
            "-ar",
            "--exclude=__pycache__",
            "hopper.orc.gmu.edu:" + old + "/code",
            "hopper.orc.gmu.edu:" + old + "/assets",
            "hopper.orc.gmu.edu:" + old + "/campaign.json",
            "hopper.orc.gmu.edu:" + old + "/submission.json",
            str(root) + "/",
        ],
        check=True,
    )
    data = root / "data"
    data.mkdir()
    (data / "packed-v1").mkdir()
    (root / "slurm").mkdir()
    (root / "runs").mkdir()
    for source in (original / "data").iterdir():
        if source.name not in ["packed-v1", "evaluation-v1", "evaluations"]:
            (data / source.name).symlink_to(source, target_is_directory=source.is_dir())
    subprocess.run(
        [
            "rsync",
            "-a",
            "hopper.orc.gmu.edu:" + old + "/data/packed-v1/manifest.json",
            str(data / "packed-v1/manifest.json"),
        ],
        check=True,
    )
    remote = json.loads((data / "packed-v1/manifest.json").read_text())
    local = json.loads((original / "data/packed-v1/manifest.json").read_text())
    if (
        remote["source_pairs_sha256"] != local["source_pairs_sha256"]
        or remote["files"] != local["files"]
    ):
        raise ValueError("H100 and Hopper packed arrays differ")
    for entry in remote["files"].values():
        (data / "packed-v1" / entry["path"]).symlink_to(
            original / "data/packed-v1" / entry["path"]
        )
    (root / "PREPARED.json").write_text(
        json.dumps(
            dict(
                original_hopper_root=old,
                verified_identical_packed_tensors=remote["files"],
                prepared=time.time(),
            ),
            indent=2,
        )
    )
    print("H100 REPLICA PREPARED; ALL FOUR PACKED TENSOR HASHES MATCH", flush=True)


if __name__ == "__main__":
    main()
