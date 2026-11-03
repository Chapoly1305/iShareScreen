"""Real Mac VideoToolbox codec controls, distinct from Screen Sharing.

Two source-known states per clip, 3 seconds each. Native window content and
font sizes stay fixed across resolutions. This is a codec diagnostic, not a
network test, a Screen Sharing capture or a final test dataset.
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    source = json.loads((args.source / "manifest.json").read_text())
    assert source["complete"] and len(source["states"]) == 8
    for state in source["states"]:
        assert (
            sha(args.source / state["reference"]["path"])
            == state["reference"]["sha256"]
        )
        assert state["reference"]["dimensions"] == [1920, 1080]
    report = dict(
        schema="mac_videotoolbox_controls_v1",
        scope=__doc__,
        platform=platform.platform(),
        started=time.time(),
        script_sha256=sha(Path(__file__)),
        source_manifest_sha256=sha(args.source / "manifest.json"),
        source_role="development_diagnostics_only",
        hardware_required=True,
        chroma="420",
        fps=30,
        encoder="hevc_videotoolbox",
        source_state_frames=[[0, 89], [90, 179]],
        runs=[],
        complete=False,
    )
    path = args.output / "MANIFEST.json"
    if args.resume:
        report = json.loads(path.read_text())
        assert report["source_manifest_sha256"] == sha(args.source / "manifest.json")
        report.setdefault("resume_history", []).append(
            dict(time=time.time(), script_sha256=sha(Path(__file__)))
        )

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    save()
    ffmpeg = "/opt/homebrew/bin/ffmpeg"
    ffprobe = "/opt/homebrew/bin/ffprobe"
    report["ffmpeg"] = subprocess.check_output(
        [ffmpeg, "-version"], text=True
    ).splitlines()[0]
    for group in ["code", "table", "paper", "terminal"]:
        pair = [args.source / (f"{group}-{v}.png") for v in (0, 1)]
        for height in [720, 1080]:
            width = height * 16 // 9
            for kbps in [600, 1000]:
                output = args.output / f"{group}-{height}p-{kbps}k-420.mp4"
                filters = (
                    "[0:v]trim=duration=3,setpts=PTS-STARTPTS[a];"
                    "[1:v]trim=duration=3,setpts=PTS-STARTPTS[b];"
                    f"[a][b]concat=n=2:v=1:a=0,scale={width}:{height}:flags=lanczos:"
                    "in_range=full:out_range=tv:out_color_matrix=bt709,format=nv12[v]"
                )
                command = [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "warning",
                    "-nostdin",
                    "-n",
                    "-loop",
                    "1",
                    "-framerate",
                    "30",
                    "-i",
                    str(pair[0]),
                    "-loop",
                    "1",
                    "-framerate",
                    "30",
                    "-i",
                    str(pair[1]),
                    "-filter_complex",
                    filters,
                    "-map",
                    "[v]",
                    "-an",
                    "-frames:v",
                    "180",
                    "-c:v",
                    "hevc_videotoolbox",
                    "-profile:v",
                    "main",
                    "-allow_sw",
                    "0",
                    "-realtime",
                    "1",
                    "-constant_bit_rate",
                    "1",
                    "-b:v",
                    f"{kbps}k",
                    "-maxrate",
                    f"{kbps}k",
                    "-bufsize",
                    f"{kbps}k",
                    "-g",
                    "60",
                    "-bf",
                    "0",
                    "-color_range",
                    "tv",
                    "-colorspace",
                    "bt709",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-tag:v",
                    "hvc1",
                    str(output),
                ]
                previous = [
                    r
                    for r in report["runs"]
                    if (r["group"], r["height"], r["requested_kbps"])
                    == (group, height, kbps)
                ]
                if previous:
                    assert len(previous) == 1 and previous[0]["command"] == command
                    if "path" in previous[0]:
                        assert sha(output) == previous[0]["sha256"]
                    continue
                start = time.monotonic()
                run = subprocess.run(
                    command, capture_output=True, text=True, timeout=180
                )
                (args.output / (output.stem + ".log")).write_text(run.stderr)
                row = dict(
                    group=group,
                    height=height,
                    width=width,
                    requested_kbps=kbps,
                    command=command,
                    returncode=run.returncode,
                    elapsed_seconds=time.monotonic() - start,
                    source_sha256=[sha(x) for x in pair],
                )
                if run.returncode == 0:
                    probe = json.loads(
                        subprocess.check_output(
                            [
                                ffprobe,
                                "-v",
                                "error",
                                "-select_streams",
                                "v:0",
                                "-count_frames",
                                "-show_streams",
                                "-show_packets",
                                "-show_entries",
                                "packet=pts_time,dts_time,size,flags:stream=codec_name,profile,width,height,pix_fmt,avg_frame_rate,nb_read_frames,color_range,color_space,color_transfer,color_primaries",
                                "-of",
                                "json",
                                str(output),
                            ],
                            text=True,
                        )
                    )
                    row.update(
                        path=output.name,
                        sha256=sha(output),
                        bytes=output.stat().st_size,
                        probe=probe,
                    )
                    stream = probe["streams"][0]
                    row["validated"] = (
                        stream["pix_fmt"] == "yuv420p"
                        and int(stream["nb_read_frames"]) == 180
                        and int(stream["height"]) == height
                    )
                else:
                    row.update(validated=False, error=run.stderr[-1500:])
                report["runs"].append(row)
                save()
                print(output.name, "valid", row["validated"], flush=True)
    report.update(
        matrix_complete=len(report["runs"]) == 16,
        complete=all(r["validated"] for r in report["runs"]),
        finished=time.time(),
    )
    save()


if __name__ == "__main__":
    main()
