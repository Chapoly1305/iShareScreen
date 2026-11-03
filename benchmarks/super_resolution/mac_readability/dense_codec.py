"""Encode frozen dense references on actual Mac HEVC hardware, 420 at 600 Kbps.

This is a matched codec control, separate from Screen Sharing transport.
Capture all frame counts and packet sizes; an offered rate is not a link cap.
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


def save(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    source = json.loads((args.source / "manifest.json").read_text())
    assert source["complete"] and len(source["states"]) == 32
    for state in source["states"]:
        assert (
            sha(args.source / state["reference"]["path"])
            == state["reference"]["sha256"]
        )
    record = args.output / "MANIFEST.json"
    result = dict(
        schema="mac_dense_codec_v1",
        scope=__doc__,
        source_manifest_sha256=sha(args.source / "manifest.json"),
        script_sha256=sha(Path(__file__)),
        platform=platform.platform(),
        fps=30,
        state_duration_s=3,
        roles="independent development validation; never train or final-test",
        ffmpeg=subprocess.check_output(
            ["/opt/homebrew/bin/ffmpeg", "-version"], text=True
        ).splitlines()[0],
        started=time.time(),
        runs=[],
        complete=False,
    )
    if args.resume:
        result = json.loads(record.read_text())
        assert result["source_manifest_sha256"] == sha(
            args.source / "manifest.json"
        ) and result["script_sha256"] == sha(Path(__file__))
    save(record, result)
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    try:
        for group in dict.fromkeys(s["group"] for s in source["states"]):
            pair = [s for s in source["states"] if s["group"] == group]
            assert len(pair) == 2
            for height in [720, 1080]:
                name = f"{group}-{height}p-600k-420"
                output = args.output / (name + ".mp4")
                prior = [r for r in result["runs"] if r["id"] == name]
                if prior:
                    assert len(prior) == 1 and sha(output) == prior[0]["sha256"]
                    continue
                filters = (
                    "[0:v]trim=duration=3,setpts=PTS-STARTPTS[a];[1:v]trim=duration=3,setpts=PTS-STARTPTS[b];"
                    f"[a][b]concat=n=2:v=1:a=0,scale={height * 16 // 9}:{height}:flags=lanczos:in_range=full:out_range=tv:out_color_matrix=bt709,format=nv12[v]"
                )
                command = [
                    "/opt/homebrew/bin/ffmpeg",
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
                    str(args.source / pair[0]["reference"]["path"]),
                    "-loop",
                    "1",
                    "-framerate",
                    "30",
                    "-i",
                    str(args.source / pair[1]["reference"]["path"]),
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
                    "600k",
                    "-maxrate",
                    "600k",
                    "-bufsize",
                    "600k",
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
                start = time.time()
                run = subprocess.run(
                    command, capture_output=True, text=True, timeout=180
                )
                (args.output / (name + ".log")).write_text(run.stderr)
                if run.returncode:
                    raise RuntimeError(name + ": " + run.stderr[-1000:])
                probe = json.loads(
                    subprocess.check_output(
                        [
                            "/opt/homebrew/bin/ffprobe",
                            "-v",
                            "error",
                            "-select_streams",
                            "v:0",
                            "-count_frames",
                            "-show_streams",
                            "-show_packets",
                            "-show_entries",
                            "packet=pts_time,dts_time,size,flags:stream=codec_name,width,height,pix_fmt,nb_read_frames,color_range,color_space,color_transfer,color_primaries",
                            "-of",
                            "json",
                            str(output),
                        ],
                        text=True,
                    )
                )
                stream = probe["streams"][0]
                row = dict(
                    id=name,
                    group=group,
                    category=pair[0]["category"],
                    height=height,
                    requested_kbps=600,
                    state_ids=[s["state_id"] for s in pair],
                    path=output.name,
                    sha256=sha(output),
                    command=command,
                    probe=probe,
                    elapsed_s=time.time() - start,
                    validated=int(stream["nb_read_frames"]) == 180
                    and stream["height"] == height
                    and stream["pix_fmt"] == "yuv420p",
                )
                result["runs"].append(row)
                save(record, result)
                print(name, "validated", row["validated"], flush=True)
        result.update(
            complete=len(result["runs"]) == 32
            and all(r["validated"] for r in result["runs"]),
            finished=time.time(),
        )
        save(record, result)
    finally:
        awake.terminate()
        awake.wait(timeout=5)


if __name__ == "__main__":
    main()
