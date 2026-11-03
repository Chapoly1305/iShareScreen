"""Mac-side whole-frame720p/420 preprocessing without temporal codec loss.

Uses frozen train-only HR targets, never validation/test. This is a new view
of existing training examples, not additional independent source material.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import av
import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, j):
    q = p.with_suffix(".tmp")
    q.write_text(json.dumps(j, indent=2) + "\n")
    q.replace(p)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    os.umask(0o077)
    j = json.loads((a.root / "INPUTS.json").read_text())
    assert j["role"] == "train_only" and len(j["rows"]) == 512
    (a.root / "lr").mkdir(exist_ok=False)
    (a.root / "smoke-raw").mkdir()
    ffmpeg = "/opt/homebrew/bin/ffmpeg"
    result = dict(
        schema="mac_steady_train_derivative_v1",
        scope=__doc__,
        input_sha256=sha(a.root / "INPUTS.json"),
        script_sha256=sha(Path(__file__)),
        platform=platform.platform(),
        av_version=av.__version__,
        ffmpeg=subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
        started=time.time(),
        complete=False,
        rows=[],
    )
    save(a.root / "MANIFEST.json", result)
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    try:
        for i, row in enumerate(j["rows"]):
            src = a.root / row["hr"]
            assert row["split"] == "train" and sha(src) == row["hr_sha256"]
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-i",
                str(src),
                "-vf",
                "scale=1280:720:flags=lanczos:in_range=full:out_range=tv:out_color_matrix=bt709,format=nv12",
                "-frames:v",
                "1",
                "-c:v",
                "rawvideo",
                "-pix_fmt",
                "nv12",
                "-f",
                "rawvideo",
                "pipe:1",
            ]
            raw = subprocess.run(
                command, check=True, capture_output=True, timeout=30
            ).stdout
            n = 1280 * 720
            flat = np.frombuffer(raw, np.uint8)
            assert len(flat) == n * 3 // 2
            planar = np.concatenate((flat[:n], flat[n::2], flat[n + 1 :: 2])).reshape(
                1080, 1280
            )
            f = av.VideoFrame.from_ndarray(planar, format="yuv420p")
            f.colorspace = 1
            f.color_range = 1
            rgb = f.to_ndarray(format="rgb24")
            assert rgb.shape == (720, 1280, 3)
            path = a.root / "lr" / (row["id"] + ".png")
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "rgb24",
                    "-s",
                    "1280x720",
                    "-i",
                    "pipe:0",
                    "-frames:v",
                    "1",
                    "-c:v",
                    "png",
                    str(path),
                ],
                input=rgb.tobytes(),
                check=True,
                capture_output=True,
                timeout=30,
            )
            out = dict(
                id=row["id"],
                hr_sha256=row["hr_sha256"],
                lr=str(path.relative_to(a.root)),
                lr_sha256=sha(path),
                lr_pixel_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                command=command,
            )
            if i in [0, 1, 2, 3, 384, 385, 386, 387]:
                rawpath = a.root / "smoke-raw" / (row["id"] + ".nv12")
                rawpath.write_bytes(raw)
                out["raw_smoke"] = str(rawpath.relative_to(a.root))
                out["raw_sha256"] = sha(rawpath)
            result["rows"].append(out)
            if (i + 1) % 32 == 0:
                save(a.root / "MANIFEST.json", result)
                print(i + 1, "/512", flush=True)
        result.update(complete=True, finished=time.time())
        save(a.root / "MANIFEST.json", result)
    finally:
        awake.terminate()
        awake.wait(timeout=5)


if __name__ == "__main__":
    main()
