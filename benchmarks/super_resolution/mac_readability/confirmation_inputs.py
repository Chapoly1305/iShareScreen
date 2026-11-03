"""Bind fresh confirmation captures and exact frozen-PTS codec inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from benchmarks.super_resolution.mac_readability.quality import save, sha


def read(path):
    return json.loads(path.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--codec-images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project
    source = read(args.source / "manifest.json")
    assert source["complete"] and len(source["states"]) == 32
    previous = [
        root
        / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/reference/manifest.json"
    ]
    previous += sorted(
        (root / "datasets/sources/mac-glyph-train-v1").glob(
            "batch-*/reference/manifest.json"
        )
    )
    hashes = set()
    files = set()
    prior = {}
    for path in previous:
        data = read(path)
        assert data["complete"]
        prior[str(path.relative_to(root))] = sha(path)
        for state in data["states"]:
            hashes.add(state["decoded_pixels_sha256"])
            files.add(state["reference"]["sha256"])
    train = read(root / "datasets/train/codescan/manifest.json")
    for row in train["rows"]:
        hashes.update(row["pixels"].values())
        files.update(row["files"].values())
    for state in source["states"]:
        assert state["decoded_pixels_sha256"] not in hashes
        assert state["reference"]["sha256"] not in files
    index = read(args.codec_images / "IMAGES.json")
    assert index["source_sha256"] == sha(args.source / "manifest.json")
    images = [x for x in index["images"] if x.get("height") == 720]
    assert len(images) == 32 and len({x["state_id"] for x in images}) == 32
    args.output.mkdir(parents=True, exist_ok=False)
    prepared = []
    states = {x["state_id"]: x for x in source["states"]}
    for row in images:
        path = Path(row["decoded_path"])
        assert sha(path) == row["decoded_sha256"]
        target = args.output / path.name
        shutil.copy2(path, target)
        assert sha(target) == row["decoded_sha256"]
        prepared.append(
            dict(
                row,
                input_path=target.name,
                decoded_path=str(target.resolve()),
                category=states[row["state_id"]]["category"],
            )
        )
    save(
        args.output / "INPUTS.json",
        dict(
            complete=True,
            images=prepared,
            source_sha256=index["source_sha256"],
            scope=__doc__,
            test_used=False,
            source_pixel_and_file_overlap=0,
            prior_manifests=prior,
            train_manifest_sha256=sha(root / "datasets/train/codescan/manifest.json"),
            pixel_audit_note="RGB pixel hashes against native captures; additionally compare recordedCodeSCANpixel/filehashes. Text/repository/near-duplicate audit is bound separately in source definitions.",
        ),
    )
    print(
        "32 exact-PTS inputs; captured pixels and files disjoint from prior native train/dev and recorded CodeSCAN train hashes"
    )


if __name__ == "__main__":
    main()
