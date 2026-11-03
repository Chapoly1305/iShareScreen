"""Export exact Mac pre-encoder controls, without lossy video compression.

Use the same FFmpeg build and scale/range/matrix filter as dense_codec.py.
RGB resize is a resolution-only diagnostic, not a new chroma-format sweep.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.umask(0o077)
    source = json.loads((a.source / "manifest.json").read_text())
    assert source["complete"] and len(source["states"]) == 32
    ffmpeg = "/opt/homebrew/bin/ffmpeg"
    result = dict(
        schema="mac_precodec_controls_v1",
        source_sha256=sha(a.source / "manifest.json"),
        script_sha256=sha(Path(__file__)),
        platform=platform.platform(),
        ffmpeg=subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0],
        scope=__doc__,
        started=time.time(),
        complete=False,
        images=[],
    )
    save(a.output / "MANIFEST.json", result)
    for state in source["states"]:
        path = a.source / state["reference"]["path"]
        assert sha(path) == state["reference"]["sha256"]
        for stage, height in [
            ("rgb-resize", 720),
            ("420-no-codec", 720),
            ("420-no-codec", 1080),
        ]:
            width = height * 16 // 9
            name = f"{state['id']}-{height}p-{stage}"
            raw = stage == "420-no-codec"
            output = a.output / (name + (".nv12" if raw else ".png"))
            filters = f"scale={width}:{height}:flags=lanczos"
            filters += (
                ":in_range=full:out_range=tv:out_color_matrix=bt709,format=nv12"
                if raw
                else ",format=rgb24"
            )
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-n",
                "-i",
                str(path),
                "-vf",
                filters,
                "-frames:v",
                "1",
            ]
            command += (
                ["-c:v", "rawvideo", "-pix_fmt", "nv12", "-f", "rawvideo"]
                if raw
                else ["-c:v", "png", "-pix_fmt", "rgb24"]
            )
            command.append(str(output))
            subprocess.run(command, check=True, capture_output=True, timeout=30)
            if raw:
                assert output.stat().st_size == width * height * 3 // 2
            result["images"].append(
                dict(
                    id=name,
                    group=state["group"],
                    category=state["category"],
                    state_id=state["state_id"],
                    stage=stage,
                    height=height,
                    width=width,
                    path=output.name,
                    sha256=sha(output),
                    command=command,
                    source_image_sha256=state["reference"]["sha256"],
                    color_range="tv" if raw else "full",
                    matrix="bt709" if raw else "RGB",
                )
            )
        save(a.output / "MANIFEST.json", result)
        print(state["id"], flush=True)
    result.update(complete=True, finished=time.time())
    save(a.output / "MANIFEST.json", result)


if __name__ == "__main__":
    main()
