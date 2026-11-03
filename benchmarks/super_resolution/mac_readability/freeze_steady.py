"""Validate and freeze the new train-only steady420 derivative on H100."""

import argparse
import hashlib
import json
from pathlib import Path
import av
import numpy as np
from PIL import Image
from benchmarks.super_resolution.mac_readability.quality import save, sha
from benchmarks.super_resolution.mac_readability.prepare_precodec import planar_nv12
from benchmarks.super_resolution.mac_readability.train_ablations import training_data


def read(p):
    return json.loads(p.read_text())


def pixel(p):
    return hashlib.sha256(
        np.asarray(Image.open(p).convert("RGB")).tobytes()
    ).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project", type=Path, required=True)
    a = p.parse_args()
    root = a.project / "datasets/derived/mac-steady-720p-420-v1"
    inputs = read(root / "INPUTS.json")
    manifest = read(root / "MANIFEST.json")
    assert manifest["complete"] and manifest["input_sha256"] == sha(
        root / "INPUTS.json"
    )
    original, bindings = training_data(a.project)
    assert bindings == inputs["original_bindings"]
    held = (
        a.project
        / "datasets/sources/mac-readability-v2-20261102/dense-development-v1/reference"
    )
    truth = read(held / "manifest.json")
    dev_pixels = {pixel(held / s["reference"]["path"]) for s in truth["states"]}
    dev_repos = {s["origin"].get("repository") for s in truth["states"]} - {None}
    original = {r["id"]: r for r in original}
    generated = {r["id"]: r for r in manifest["rows"]}
    rows = []
    smokes = []
    for row in inputs["rows"]:
        old = original[row["id"]]
        new = generated[row["id"]]
        assert row["split"] == "train" and row["repository"] not in dev_repos
        assert (
            sha(root / row["hr"])
            == row["hr_sha256"]
            == old["hr_sha256"]
            == new["hr_sha256"]
        )
        assert (
            pixel(root / row["hr"]) == row["hr_pixel_sha256"]
            and row["hr_pixel_sha256"] not in dev_pixels
        )
        assert (
            sha(root / new["lr"]) == new["lr_sha256"]
            and pixel(root / new["lr"]) == new["lr_pixel_sha256"]
        )
        rgb = np.asarray(Image.open(root / new["lr"]).convert("RGB"))
        assert rgb.shape == (720, 1280, 3)
        if "raw_smoke" in new:
            path = root / new["raw_smoke"]
            assert sha(path) == new["raw_sha256"]
            frame = av.VideoFrame.from_ndarray(
                planar_nv12(path.read_bytes(), 1280, 720), format="yuv420p"
            )
            frame.colorspace = 1
            frame.color_range = 1
            error = np.abs(
                frame.to_ndarray(format="rgb24").astype("i2") - rgb.astype("i2")
            )
            assert error.max() <= 2
            smokes.append(
                dict(
                    id=row["id"],
                    max_rgb_difference=int(error.max()),
                    mean_rgb_difference=float(error.mean()),
                )
            )
        result = dict(
            row,
            lr=new["lr"],
            lr_sha256=new["lr_sha256"],
            lr_pixel_sha256=new["lr_pixel_sha256"],
            degradation="ActualMacFFmpeg8.1 whole-frame Lanczos720,420 BT709 limited, no temporal codec",
            selected_pts_s=None,
        )
        result["parent_codec_receipt"] = result.pop("codec_receipt")
        rows.append(result)
    assert len(rows) == 512 and len(smokes) == 8
    for key in ["id", "lr_sha256", "lr_pixel_sha256", "hr_pixel_sha256"]:
        assert len({r[key] for r in rows}) == 512
    save(
        root / "SMOKE-VALIDATION.json",
        dict(
            passed=True,
            cases=smokes,
            scope="IndependentLinux decode of8 actualMacraw buffers,8-bit tolerance2; no source shift introduced",
        ),
    )
    save(
        root / "DATA.json",
        dict(
            schema="mac_steady_train_pairs_v1",
            role="train_only",
            complete=True,
            counts=dict(images=512, groups=len({r["group_id"] for r in rows})),
            rows=rows,
            parent_bindings=bindings,
            scope="Alternative degradation of existing512 training sources; not additional independent examples",
        ),
    )
    save(
        root / "FROZEN.json",
        dict(
            data_sha256=sha(root / "DATA.json"),
            input_sha256=sha(root / "INPUTS.json"),
            manifest_sha256=sha(root / "MANIFEST.json"),
            smoke_sha256=sha(root / "SMOKE-VALIDATION.json"),
            validator_sha256=sha(Path(__file__)),
            dev_reference_sha256=sha(held / "manifest.json"),
            test_read=False,
            complete=True,
        ),
    )
    print("Frozen512steady pairs;8 independent color checks passed")


if __name__ == "__main__":
    main()
