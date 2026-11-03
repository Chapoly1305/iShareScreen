"""Align real Screen Sharing tiles to controlled window references.

The two state markers reject mixed or unrelated windows. Snapshot sampling is
too sparse to establish input latency. Color normalization uses the embedded
ICC of the same Mac display; absolute decoder color accuracy is unqualified.
"""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms

from isharescreen.enhancement.live import compose, CanvasSnapshot
from isharescreen.proxy.media.tiles import TileFrame
from benchmarks.super_resolution.mac_readability.quality import sha, save, markers


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    capture = json.loads((args.capture / "STATUS.json").read_text())
    stimulus = json.loads((args.capture / "stimulus/manifest.json").read_text())
    source = json.loads((args.source / "manifest.json").read_text())
    assert (
        capture["state"] == "completed" and stimulus["complete"] and source["complete"]
    )
    window = next(
        w for w in stimulus["own_windows"] if w["number"] == stimulus["window_id"]
    )
    scale = stimulus["backing_scale"]
    bounds = window["bounds"]
    x, y, w, h = [round(bounds[key] * scale) for key in ["X", "Y", "Width", "Height"]]
    assert (w, h) == (1920, 1080)
    original = Image.open(
        args.capture / "stimulus" / stimulus["states"][0]["reference"]["path"]
    )
    icc = original.info["icc_profile"]
    profile = ImageCms.ImageCmsProfile(BytesIO(icc))
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))
    result = dict(
        schema="actual_screen_sharing_aligned_v1",
        scope=__doc__,
        capture_sha256=sha(args.capture / "STATUS.json"),
        source_sha256=sha(args.source / "manifest.json"),
        script_sha256=sha(Path(__file__)),
        window_bounds_pixels=[x, y, w, h],
        input_icc_sha256=hashlib.sha256(icc).hexdigest(),
        complete=False,
        snapshots=[],
        images=[],
    )
    for snapshot in capture["snapshots"]:
        path = args.capture / snapshot["path"]
        tiles = []
        with np.load(path) as arrays:
            for meta in snapshot["tiles"]:
                i = meta["index"]
                fields = {
                    k: v
                    for k, v in meta.items()
                    if k not in ["index", "received_ns", "biplanar"]
                }
                fields.update(
                    {
                        p: arrays[f"tile_{i}_{p}"].tobytes()
                        if f"tile_{i}_{p}" in arrays
                        else None
                        for p in ["y", "u", "v"]
                    }
                )
                tile = TileFrame(**fields)
                tiles.append((tile, tile.height))
        rgb = compose(CanvasSnapshot(*capture["canvas"], tuple(tiles)))[
            y : y + h, x : x + w
        ]
        found = markers(rgb)
        row = dict(
            path=path.name,
            sha256=sha(path),
            sampled_ns=snapshot["sampled_ns"],
            markers=found,
            accepted=False,
        )
        if found[0] == found[1] and 1 <= found[0] <= 8:
            state = next(s for s in stimulus["states"] if s["state_id"] == found[0])
            corrected = ImageCms.profileToProfile(
                Image.fromarray(rgb), profile, srgb, renderingIntent=1, outputMode="RGB"
            )
            out = args.output / (path.stem + ".png")
            corrected.save(out, icc_profile=srgb.tobytes())
            age = (snapshot["sampled_ns"] - state["draw_start_ns"]) / 1e9
            row.update(accepted=True, state_id=state["state_id"], source_age_s=age)
            result["images"].append(
                dict(
                    id=path.stem,
                    path=str(out.resolve()),
                    sha256=sha(out),
                    state_id=state["state_id"],
                    group=state["group"],
                    source_age_s=age,
                    config="screen-sharing-native-3360x2100-600k-offer",
                    method="native",
                    measurement="Latest per-tile canvas; state markers agree, not an atomic video frame",
                )
            )
        result["snapshots"].append(row)
    assert {r["state_id"] for r in result["images"]} == set(range(1, 9))
    # Retain all aligned snapshots; score a fixed predeclared age per state.
    selected = []
    for state in source["states"]:
        candidates = [r for r in result["images"] if r["state_id"] == state["state_id"]]
        for target in [0.2, 1.0, 2.8]:
            item = min(candidates, key=lambda r: abs(r["source_age_s"] - target))
            selected.append(
                dict(
                    item,
                    id=item["id"] + f"-target{target:.1f}",
                    actual_source_age_s=item["source_age_s"],
                    source_age_s=target,
                )
            )
    result["complete"] = True
    save(args.output / "ALIGNMENT.json", result)
    save(
        args.output / "SELECTED.json",
        dict(
            source_sha256=result["source_sha256"],
            images=selected,
            scope="Nearest of 0.5s snapshots to target ages. Actual ages retained; no latency claim.",
        ),
    )
    print(
        len(result["images"]),
        "aligned snapshots; all 8 source states; selected",
        len(selected),
    )


if __name__ == "__main__":
    main()
