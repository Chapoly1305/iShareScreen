"""Extract source-aligned, exact-PTS images for rate and temporal diagnostics."""

import argparse
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.mac_readability.quality import markers, save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--controls", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mode", choices=["matched", "temporal"], required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    source = json.loads((a.source / "manifest.json").read_text())
    manifest = json.loads((a.controls / "MANIFEST.json").read_text())
    assert manifest.get("source_sha256", manifest.get("source_manifest_sha256")) == sha(
        a.source / "manifest.json"
    )
    if a.mode == "matched":
        assert manifest["complete"]
        runs = []
        for group in manifest["groups"]:
            assert group["matched"]
            row = next(r for r in group["attempts"] if r["id"] == group["selected_id"])
            runs.append(dict(row, target_kbps=group["target_kbps"]))
        ages = [2.8]
    else:
        runs = manifest["runs"]
        ages = [0.2, 1.0]
    images = []
    missing = []
    for run in runs:
        path = a.controls / run["path"]
        assert sha(path) == run["sha256"]
        wanted = {
            round(k * 3 + age, 6): (state_id, age)
            for k, state_id in enumerate(run["state_ids"])
            for age in ages
        }
        with av.open(path) as container:
            for frame in container.decode(video=0):
                t = round(float(frame.pts * frame.time_base), 6)
                if t not in wanted:
                    continue
                state_id, age = wanted.pop(t)
                rgb = frame.to_ndarray(format="rgb24")
                assert markers(rgb) == [state_id] * 2
                state = next(s for s in source["states"] if s["state_id"] == state_id)
                name = f"{run['id']}-s{state_id:02d}-age{age:.1f}"
                output = a.output / (name + ".png")
                Image.fromarray(
                    cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                ).save(output)
                images.append(
                    dict(
                        id=name,
                        path=str(output.resolve()),
                        sha256=sha(output),
                        state_id=state_id,
                        group=state["group"],
                        category=state["category"],
                        config="1080p-matched-payload"
                        if a.mode == "matched"
                        else f"{run['height']}p-600k",
                        method="bicubic" if run["height"] == 720 else "native",
                        source_age_s=age,
                        pts_s=t,
                        height=run["height"],
                        actual_output_frames=int(
                            run["probe"]["streams"][0]["nb_read_frames"]
                        ),
                        requested_kbps=run["requested_kbps"],
                        actual_kbps=sum(int(p["size"]) for p in run["probe"]["packets"])
                        * 8
                        / 6000,
                        target_kbps=run.get("target_kbps"),
                        relative_rate_error=run.get("relative_rate_error"),
                    )
                )
        missing.extend(
            dict(run=run["id"], pts=pts, state_id=v[0]) for pts, v in wanted.items()
        )
    result = dict(
        source_sha256=sha(a.source / "manifest.json"),
        script_sha256=sha(Path(__file__)),
        manifest_sha256=sha(a.controls / "MANIFEST.json"),
        scope=__doc__ + " Media PTS age is not wall-clock delivery latency.",
        mode=a.mode,
        ages=ages,
        missing_samples=missing,
        complete=not missing,
        images=images,
    )
    save(a.output / "IMAGES.json", result)
    assert not missing, missing


if __name__ == "__main__":
    main()
