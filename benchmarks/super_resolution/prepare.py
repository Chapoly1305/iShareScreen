"""Inventory capture files and make reproducible aligned single-frame pairs."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw
from isharescreen.enhancement.models import sha256


def run(command):
    return subprocess.run(command, check=True, capture_output=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("benchmarks/super_resolution/artifacts/pairs"))
    p.add_argument("--frames-per-clip", type=int, default=1)
    p.add_argument("--review", type=Path, help="Optional JSON with excluded, content and session_group fields")
    p.add_argument("--color", choices=["full709","ffmpeg-default"], default="full709")
    args = p.parse_args()
    if args.frames_per_clip < 1:
        p.error("frames-per-clip must be positive")
    if (args.output/"manifest.json").exists():
        p.error("Output already contains a manifest; use a new output directory to preserve provenance")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"generator_environment.json").write_text(json.dumps({
        "generator_sha256":sha256(Path(__file__)),"color":args.color,
        "ffmpeg":run(["ffmpeg","-version"]).stdout.decode().splitlines()[0]},indent=2))
    inventory, pairs, thumbnails = [], [], []
    review = json.loads(args.review.read_text()) if args.review else {"excluded": {}, "content": {}, "session_group": "unspecified"}
    for video in sorted(args.dataset.glob("*.mov")):
        meta = json.loads(run(["ffprobe", "-v", "error", "-show_streams", "-show_format",
                               "-of", "json", str(video)]).stdout)
        stream = next(s for s in meta["streams"] if s["codec_type"] == "video")
        duration = float(meta["format"]["duration"])
        inventory.append({"file": str(video), "bytes": video.stat().st_size,
                          "duration_s": duration, "stream": stream})
        if video.stem in review["excluded"]:
            inventory[-1]["excluded_reason"] = review["excluded"][video.stem]
            print(f"inventory only: {video.name}: {review['excluded'][video.stem]}", flush=True)
            continue
        for frame_index in range(args.frames_per_clip):
            timestamp = duration * (frame_index + 1) / (args.frames_per_clip + 1)
            sample = f"{video.stem}_f{frame_index:02d}"
            folder = args.output / sample
            folder.mkdir(exist_ok=True)
            raw = folder / "decoded.png"
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(timestamp),
                 "-i", str(video), "-frames:v", "1", str(raw)])
            im = np.array(Image.open(raw).convert("RGB"))
            h, w = im.shape[:2]
            cw, ch = min(1920, w//2*2), min(1080, h//2*2)
            x, y = (w-cw)//2, (h-ch)//2
            hr = im[y:y+ch, x:x+cw]
            Image.fromarray(hr).save(folder / "hr.png")
            hr_hash = sha256(folder/"hr.png")
            raw.unlink()
            # PIL's reducing bicubic filter includes antialiasing. OpenCV's
            # INTER_CUBIC reduction does not and creates a different aliasing
            # distribution from the models' original training inputs.
            lr = np.array(Image.fromarray(hr).resize((cw//2,ch//2), Image.Resampling.BICUBIC))
            Image.fromarray(lr).save(folder / "clean.png")
            thumb = Image.fromarray(hr); thumb.thumbnail((320, 180))
            thumbnails.append((sample, thumb.copy()))
            for degradation in ["clean", "hevc18", "hevc28", "hevc38"]:
                commands = []
                if degradation != "clean":
                    crf = degradation.removeprefix("hevc")
                    encoded = folder / f"{degradation}.hevc"
                    color_options = (["-vf","scale=in_range=full:out_range=full:out_color_matrix=bt709",
                                      "-color_range","pc","-colorspace","bt709",
                                      "-color_primaries","bt709","-color_trc","bt709"]
                                     if args.color=="full709" else [])
                    # One intra frame: explicitly not a temporal Apple-stream simulation.
                    commands = [
                        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(folder/"clean.png"),
                         "-frames:v", "1", *color_options, "-c:v", "libx265", "-pix_fmt", "yuv444p", "-crf", crf,
                         "-x265-params", "pools=2:frame-threads=1:log-level=error", str(encoded)],
                        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(encoded),
                         "-frames:v", "1", str(folder/f"{degradation}.png")]]
                    for command in commands:
                        run(command)
                pairs.append({"id": f"{sample}_{degradation}", "capture": video.stem,
                              "session_group": review["session_group"],
                              "content": review["content"].get(video.stem,"unreviewed"),
                              "split": "exploratory_pilot", "source": str(video),
                              "timestamp_s": timestamp, "crop_xywh": [x, y, cw, ch],
                              "downsample": "Pillow BICUBIC antialiased, scale 0.5",
                              "hevc_color":args.color,"hr_sha256":hr_hash,
                              "lr_sha256":sha256(folder/f"{degradation}.png"),
                              "hr": f"{sample}/hr.png", "lr": f"{sample}/{degradation}.png",
                              "degradation": degradation, "commands": commands,
                              "reference": "decoded recording proxy, not lossless acquisition"})
            print(f"prepared {sample}: {cw}×{ch} reference, {cw//2}×{ch//2} input", flush=True)
    (args.output / "inventory.json").write_text(json.dumps(inventory, indent=2))
    (args.output / "manifest.json").write_text(json.dumps(pairs, indent=2))
    sheet = Image.new("RGB", (4*320, ((len(thumbnails)+3)//4)*210), "#222222")
    draw = ImageDraw.Draw(sheet)
    for i, (name, thumb) in enumerate(thumbnails):
        x, y = i%4*320, i//4*210
        sheet.paste(thumb, (x,y)); draw.text((x+4,y+182), name, fill="white")
    sheet.save(args.output / "contact_sheet.jpg")
    print(f"complete: {len(inventory)} videos, {len(pairs)} aligned pairs", flush=True)


if __name__ == "__main__":
    main()
