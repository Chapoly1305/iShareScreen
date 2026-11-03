"""Bounded real-Mac hardware degradation of separately frozen train sources."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

from dense_codec import save, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=384)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    assert 1 <= args.limit <= 512
    os.umask(0o077)
    args.output.mkdir(parents=True, exist_ok=args.resume)
    inputs = json.loads((args.source / "INPUTS.json").read_text())
    assert inputs["complete"] and inputs["role"] == "train_only"
    # Interleave languages so the 8-case smoke covers 8 languages.
    languages = sorted({r["language"] for r in inputs["rows"]})
    pools = {
        language: sorted(
            (r for r in inputs["rows"] if r["language"] == language),
            key=lambda r: r["id"],
        )
        for language in languages
    }
    rows = [
        pools[language][index]
        for index in range(max(map(len, pools.values())))
        for language in languages
        if index < len(pools[language])
    ][: args.limit]
    assert len(rows) == args.limit
    result = dict(
        schema="mac_vt_train_codec_v1",
        role="train_only",
        input_manifest_sha256=sha(args.source / "INPUTS.json"),
        script_sha256=sha(Path(__file__)),
        recipe=inputs["recipe"],
        platform=platform.platform(),
        ffmpeg=subprocess.check_output(
            ["/opt/homebrew/bin/ffmpeg", "-version"], text=True
        ).splitlines()[0],
        selected_ids=[r["id"] for r in rows],
        started=time.time(),
        runs=[],
        complete=False,
        retry_policy="At most3 attempts solely for missing exact1.5s output timestamp; keep every attempt. Never select by image quality.",
    )
    path = args.output / "MANIFEST.json"
    if args.resume:
        prior = json.loads(path.read_text())
        assert (
            prior["input_manifest_sha256"] == result["input_manifest_sha256"]
            and prior["script_sha256"] == result["script_sha256"]
            and prior["selected_ids"] == result["selected_ids"]
        )
        result = prior
    save(path, result)
    awake = subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(os.getpid())])
    try:
        for row in rows:
            prior = [r for r in result["runs"] if r["id"] == row["id"]]
            if prior:
                assert (
                    len(prior) == 1
                    and sha(args.output / prior[0]["lr"]) == prior[0]["lr_sha256"]
                )
                continue
            source = args.source / row["input_path"]
            assert sha(source) == row["input_sha256"]
            folder = args.output / row["id"]
            folder.mkdir(exist_ok=True)
            background = "0x" + "".join(f"{v:02x}" for v in row["background_rgb"])
            vf = (
                f"format=rgb24,pad=1920:1110:0:0:color={background},"
                "crop=1920:1080:0:'if(lt(n,30),0,round(30*(n-30)/29))',"
                "scale=1280:720:flags=lanczos:in_range=full:out_range=tv:out_color_matrix=bt709,format=nv12"
            )
            attempts = []
            selected = None
            for attempt in range(3):
                video = folder / f"attempt-{attempt}.mp4"
                cmd = [
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
                    str(source),
                    "-an",
                    "-vf",
                    vf,
                    "-frames:v",
                    "60",
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
                    str(video),
                ]
                started = time.time()
                encoded = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=120
                )
                (folder / f"attempt-{attempt}.log").write_text(encoded.stderr)
                if encoded.returncode:
                    raise RuntimeError(row["id"] + ": " + encoded.stderr[-1200:])
                probe = json.loads(
                    subprocess.check_output(
                        [
                            "/opt/homebrew/bin/ffprobe",
                            "-v",
                            "error",
                            "-select_streams",
                            "v:0",
                            "-show_packets",
                            "-show_streams",
                            "-count_frames",
                            "-show_entries",
                            "packet=pts_time,size,flags:stream=width,height,pix_fmt,nb_read_frames,color_range,color_space",
                            "-of",
                            "json",
                            str(video),
                        ],
                        text=True,
                    )
                )
                stream = probe["streams"][0]
                assert (stream["width"], stream["height"], stream["pix_fmt"]) == (
                    1280,
                    720,
                    "yuv420p",
                )
                exact = any(
                    abs(float(p["pts_time"]) - 1.5) < 1e-6 for p in probe["packets"]
                )
                attempts.append(
                    dict(
                        attempt=attempt,
                        command=cmd,
                        video=str(video.relative_to(args.output)),
                        video_sha256=sha(video),
                        probe=probe,
                        exact_target_pts_present=exact,
                        elapsed_s=time.time() - started,
                    )
                )
                if exact:
                    selected = attempt
                    break
            if selected is None:
                save(folder / "FAILED.json", dict(id=row["id"], attempts=attempts))
                raise RuntimeError("Exact target missing after3 attempts: " + row["id"])
            output = folder / "lr.png"
            subprocess.run(
                [
                    "/opt/homebrew/bin/ffmpeg",
                    "-v",
                    "error",
                    "-nostdin",
                    "-n",
                    "-i",
                    str(video),
                    "-vf",
                    "select=gte(t\\,1.5)",
                    "-frames:v",
                    "1",
                    str(output),
                ],
                check=True,
                capture_output=True,
                timeout=30,
            )
            item = dict(
                id=row["id"],
                split="train",
                language=row["language"],
                repository=row["repository"],
                group_id=row["group_id"],
                source_sha256=row["source_sha256"],
                input_sha256=row["input_sha256"],
                selected_pts_s=1.5,
                selected_attempt=selected,
                lr=str(output.relative_to(args.output)),
                lr_sha256=sha(output),
                attempts=attempts,
            )
            save(folder / "COMPLETED.json", item)
            result["runs"].append(item)
            save(path, result)
            print("encoded", len(result["runs"]), "/", len(rows), flush=True)
        result.update(complete=len(result["runs"]) == len(rows), finished=time.time())
        save(path, result)
    finally:
        awake.terminate()
        awake.wait(timeout=5)


if __name__ == "__main__":
    main()
