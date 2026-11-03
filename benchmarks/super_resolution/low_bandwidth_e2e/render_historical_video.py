"""Render a lossless-input, labeled viewing copy of the historical video comparison.

All four panels use the same decoded frame index. The H.264 output is for review
only; the metrics use the verified source encodes and native PNGs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from itertools import zip_longest
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from isharescreen.enhancement.candidates import TorchUpscaler
from isharescreen.enhancement.models import sha256


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--scene",
        choices=("light-code", "terminal-light", "web-document"),
        required=True,
    )
    parser.add_argument("--kbps", type=int, default=1250)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root / "manifest.json").read_text())
    plan = json.loads((root / "PLAN.json").read_text())
    truth = json.loads(
        Path(
            "benchmarks/super_resolution/artifacts/base-sweep-v1/text-ground-truth.json"
        ).read_text()
    )
    region = next(r for r in truth["regions"] if r["scene"] == args.scene)
    cells = {}
    for height in (720, 1080):
        cell = next(
            r
            for r in manifest["rows"]
            if r["scene"] == args.scene
            and r["target_kbps"] == args.kbps
            and r["height"] == height
        )
        if sha256(Path(cell["video"])) != cell["video_sha256"]:
            raise ValueError("source encode changed")
        cells[height] = cell
    import torch

    torch.set_num_threads(4)
    cv2.setNumThreads(4)
    model = TorchUpscaler("quicksrnet", device="cpu", precision="fp32")
    if (
        model.provenance["checkpoint_sha256"]
        != plan["official_quicksrnet_checkpoint_sha256"]
    ):
        raise ValueError("model changed")
    output = root / "gallery" / f"{args.scene}-{args.kbps}k-comparison.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    x, y, w, h = region["xywh"]
    zoom = 2
    label_h = 34
    panel_w, panel_h = w * zoom, h * zoom
    size = (panel_w, (panel_h + label_h) * 4)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 21)
    labels = ("Original recording", "720p direct", "720p QuickSRNet x2", "1080p direct")
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        f"{size[0]}x{size[1]}",
        "-framerate",
        "30",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    reference = next(s for s in plan["sources"] if s["id"] == args.scene)["reference"]
    with (
        av.open(reference) as ref_container,
        av.open(cells[720]["video"]) as low_container,
        av.open(cells[1080]["video"]) as high_container,
    ):
        ref_frames = (
            f for i, f in enumerate(ref_container.decode(video=0)) if i % 2 == 0
        )
        low_frames = low_container.decode(video=0)
        high_frames = high_container.decode(video=0)
        with (output.with_suffix(".log")).open("w") as log:
            proc = subprocess.Popen(
                command, stdin=subprocess.PIPE, stdout=log, stderr=log
            )
            count = 0
            try:
                for frames in zip_longest(ref_frames, low_frames, high_frames):
                    if any(frame is None for frame in frames):
                        raise ValueError("unequal frame counts")
                    clean, low, high = (
                        frame.to_ndarray(format="rgb24") for frame in frames
                    )
                    direct_720 = cv2.resize(
                        low, (1920, 1080), interpolation=cv2.INTER_CUBIC
                    )
                    quick, _ = model(low)
                    quick = cv2.resize(
                        quick, (1920, 1080), interpolation=cv2.INTER_AREA
                    )
                    image = Image.new("RGB", size, "#eef2f7")
                    draw = ImageDraw.Draw(image)
                    for index, (label, rgb) in enumerate(
                        zip(labels, (clean, direct_720, quick, high))
                    ):
                        top = index * (panel_h + label_h)
                        draw.text((10, top + 4), label, font=font, fill="#17212b")
                        crop = Image.fromarray(rgb).crop((x, y, x + w, y + h))
                        image.paste(
                            crop.resize((panel_w, panel_h), Image.Resampling.NEAREST),
                            (0, top + label_h),
                        )
                    if count == 60:
                        image.save(
                            output.with_name(
                                output.stem.replace("comparison", "preview") + ".png"
                            )
                        )
                    assert proc.stdin is not None
                    proc.stdin.write(np.asarray(image).tobytes())
                    count += 1
                    if count % 30 == 0:
                        print(f"FRAME {count}/120 {args.scene}", flush=True)
            finally:
                if proc.stdin:
                    proc.stdin.close()
                rc = proc.wait()
            if rc != 0 or count != 120:
                raise ValueError(f"viewing encode failed: rc={rc} frames={count}")
    receipt = dict(
        scene=args.scene,
        target_kbps=args.kbps,
        frames=count,
        fps=30,
        output=str(output),
        sha256=sha256(output),
        source_video_sha256={str(k): v["video_sha256"] for k, v in cells.items()},
        model_sha256=model.provenance["checkpoint_sha256"],
        crop=region["xywh"],
        zoom=zoom,
        caveat="H.264 comparison is a viewing copy; metrics use native HEVC decode and PNG outputs.",
    )
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
