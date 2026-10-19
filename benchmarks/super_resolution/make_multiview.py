"""Make a synchronized six-panel viewing copy from an audited recorded-video run.

The first chapter shows the whole desktop, the second repeats the same timeline
with an identical 640x360 crop enlarged 2x. This is not a metric input.
"""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.prepare_video import probe
from benchmarks.super_resolution.video_report import audit
from isharescreen.enhancement.models import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Fresh directory")
    parser.add_argument("--font", type=Path, help="Optional TrueType font for panel labels")
    args = parser.parse_args()
    manifest, _, _, results = audit(args.run, args.manifest)
    clip = next(c for c in manifest["clips"] if c["id"] == args.clip)
    if Fraction(clip["fps"]) != 60 or clip["frames"] % 2:
        raise ValueError("This viewing layout requires an even-length 60fps clip")
    paths = [args.manifest.parent / clip["reference"], args.manifest.parent / clip["low"]]
    methods = ["bicubic", "edsr", "imdn", "safmn"]
    for method in methods:
        result = next(r for r in results if (r["clip"], r["model"]) == (args.clip, method))
        paths.append(args.run / result["output"])
    for i, path in enumerate(paths):
        stream = probe(path)
        expected_size = (960, 540) if i == 1 else (1920, 1080)
        assert (stream["width"], stream["height"]) == expected_size
        assert int(stream["nb_read_frames"]) == clip["frames"]
        assert Fraction(stream["avg_frame_rate"]) == 60
    args.output.mkdir(parents=True, exist_ok=False)
    labels = ["ORIGINAL 1920x1080", "INPUT 960x540 HEVC (nearest)", "BICUBIC x2",
              "EDSR x2 (pretrained)", "IMDN x2 (pretrained)", "SAFMN x2 (pretrained)"]
    commands = []
    frames_per_chapter = clip["frames"] // 2
    chapters = []
    for chapter in ["overview", "detail"]:
        # Static labels are rendered with Pillow because local FFmpeg lacks drawtext.
        canvas = Image.new("RGBA", (3840, 1536))
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype(str(args.font), 29) if args.font else ImageFont.load_default(size=29)
        for i, label in enumerate(labels):
            x, y = (i % 3) * 1280, (i // 3) * 768
            draw.rectangle((x, y, x + 1279, y + 47), fill=(12, 16, 24, 255))
            suffix = " | FULL DESKTOP" if chapter == "overview" else " | SAME REGION, 2x DETAIL"
            draw.text((x + 18, y + 8), label + suffix, font=font, fill="white")
        label_path = args.output / f"{chapter}-labels.png"
        canvas.save(label_path)
        command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-filter_complex_threads", "4"]
        for path in paths:
            command += ["-threads", "1", "-i", str(path)]
        command += ["-loop", "1", "-framerate", "30", "-i", str(label_path)]
        filters = []
        for i in range(6):
            conversion = ("scale=1920:1080:flags=neighbor:in_color_matrix=bt709:in_range=full,format=rgb24,"
                          if i == 1 else "format=rgb24,")
            resize = ("scale=1280:720:flags=lanczos" if chapter == "overview" else
                      "crop=640:360:600:220,scale=1280:720:flags=neighbor")
            filters.append(f"[{i}:v]setpts=PTS-STARTPTS,fps=30,{conversion}{resize},"
                           f"setsar=1,pad=1280:768:0:48:color=black[p{i}]")
        filters += ["".join(f"[p{i}]" for i in range(6)) +
                    "xstack=inputs=6:layout=0_0|1280_0|2560_0|0_768|1280_768|2560_768:shortest=1[grid]",
                    "[grid][6:v]overlay=shortest=1,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]"]
        destination = args.output / f"{chapter}.mp4"
        command += ["-filter_complex", ";".join(filters), "-map", "[v]", "-frames:v", str(frames_per_chapter),
                    "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-threads", "4",
                    "-color_range", "tv", "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-movflags", "+faststart", str(destination)]
        commands.append(command)
        print(f"RENDER {chapter}", flush=True)
        subprocess.run(command, check=True)
        stream = probe(destination)
        assert int(stream["nb_read_frames"]) == frames_per_chapter
        assert Fraction(stream["avg_frame_rate"]) == 30
        chapters.append(destination)
    concat = args.output / "chapters.txt"
    concat.write_text("".join(f"file '{path.name}'\n" for path in chapters))
    metadata = args.output / "chapters.ffmetadata"
    duration_ms = int(Fraction(clip["frames"], 60) * 1000)
    metadata.write_text(";FFMETADATA1\n" + "".join(
        f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={i * duration_ms}\nEND={(i + 1) * duration_ms}\nTITLE={title}\n"
        for i, title in enumerate(["Full desktop - six synchronized views", "Same text region - 2x detail"])))
    output = args.output / "all-models-overview-and-detail.mp4"
    command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "concat", "-safe", "0",
               "-i", str(concat), "-i", str(metadata), "-map", "0:v:0", "-map_metadata", "1", "-map_chapters", "1",
               "-c", "copy", "-movflags", "+faststart", str(output)]
    commands.append(command)
    subprocess.run(command, check=True)
    stream = probe(output)
    assert int(stream["nb_read_frames"]) == clip["frames"]
    assert (stream["width"], stream["height"]) == (3840, 1536)
    assert Fraction(stream["avg_frame_rate"]) == 30
    info = {"file": str(output), "sha256": sha256(output), "clip": args.clip, "sources": [str(p) for p in paths],
            "source_frames": clip["frames"], "source_fps": 60, "viewing_fps": 30, "stream": stream,
            "layout": labels, "detail_crop_xywh": [600, 220, 640, 360], "commands": commands,
            "note": "Lossy viewing copy only; original evaluation remains 60fps and lossless RGB outputs are unchanged."}
    (args.output / "manifest.json").write_text(json.dumps(info, indent=2))
    print(f"COMPLETED {output}", flush=True)


if __name__ == "__main__":
    main()
