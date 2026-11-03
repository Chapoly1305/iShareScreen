"""Encode temporal HEVC 420 episodes and retain aligned training pairs.

One new HR view contributes one LR/HR pair, not thirty repeated examples.
Rate requests, measured encoded sizes, keyframe identities and encoder warnings
are recorded separately. This is codec degradation, not network emulation.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess
import time

import av
import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement.models import sha256


def read(path):
    return json.loads(path.read_text())


def number(text):
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def bitrate(view):
    value = number(view["id"]) % 100
    return (
        128000
        if value < 10
        else 256000
        if value < 30
        else 600000
        if value < 65
        else 1000000
        if value < 90
        else 2000000
    )


def make_plan(root, index, stage="hevc-v1"):
    data = read(index)
    groups = defaultdict(list)
    for view in data["views"]:
        groups[(view["split"], view["source_group"], bitrate(view))].append(view)
    jobs = []
    for (split, group, bps), views in sorted(groups.items()):
        views.sort(key=lambda v: v["id"])
        for start in range(0, len(views), 12):
            chosen = views[start : start + 12]
            identity = fingerprint(dict(ids=[v["id"] for v in chosen], bps=bps))[:20]
            jobs.append(
                dict(
                    id="episode-" + identity,
                    split=split,
                    source_group=group,
                    bps=bps,
                    views=chosen,
                )
            )
    return dict(
        schema="isharescreen_hevc_text_degradation_v1",
        stage=stage,
        created=time.time(),
        index=str(index.resolve()),
        index_sha256=sha256(index),
        script_sha256=sha256(Path(__file__)),
        jobs=jobs,
        fps=30,
        frames_per_view=30,
        geometry=[960, 540],
        scale=2,
        output_reference=[1920, 1080],
        resampler="OpenCV Lanczos4 RGB reduction; full-range BT.709 conversion and HEVC 420 by FFmpeg",
        encoder="libx265 fast/zerolatency, no B frames or lookahead, fixed 60-frame GOP, scene-cut disabled, one-second VBV, HRD/strict CBR request",
        sample="One actual decoded frame per new view at local index 0/5/15/29; exact reference retained. 20% of episodes use a small content scroll before the aligned final sample.",
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True),
        opencv=cv2.__version__,
        pyav=av.__version__,
    )


def encode(root, plan, job):
    folder = root / plan["stage"] / "episodes" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "COMPLETED.json"
    if receipt.exists():
        row = read(receipt)
        if (
            row["job_binding"] != fingerprint(job)
            or sha256(folder / "episode.mkv") != row["video_sha256"]
        ):
            raise ValueError("Changed encoded episode")
        for pair in row["pairs"]:
            if sha256(root / pair["lr"]) != pair["lr_sha256"]:
                raise ValueError("Changed decoded training input")
        return row
    if (folder / "episode.mkv").exists():
        raise ValueError("Incomplete episode requires inspection before explicit retry")
    kbps = job["bps"] // 1000
    path = folder / "episode.mkv"
    log = folder / "encode.log"
    params = f"pools=1:frame-threads=1:keyint=60:min-keyint=60:scenecut=0:bframes=0:rc-lookahead=0:bitrate={kbps}:vbv-maxrate={kbps}:vbv-bufsize={kbps}:vbv-init=0.5:strict-cbr=1:hrd=1:log-level=info"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "960x540",
        "-framerate",
        "30",
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "scale=in_range=full:out_range=full:out_color_matrix=bt709",
        "-c:v",
        "libx265",
        "-preset",
        "fast",
        "-tune",
        "zerolatency",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "pc",
        "-colorspace",
        "bt709",
        "-color_trc",
        "bt709",
        "-color_primaries",
        "bt709",
        "-x265-params",
        params,
        str(path),
    ]
    samples = []
    scroll = number(job["id"]) % 5 == 0
    with log.open("w") as stderr:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr
        )
        try:
            for index, view in enumerate(job["views"]):
                if sha256(root / view["hr"]) != view["hr_sha256"]:
                    raise ValueError("HR changed before encoding")
                image = np.array(Image.open(root / view["hr"]).convert("RGB"))
                low = cv2.resize(image, (960, 540), interpolation=cv2.INTER_LANCZOS4)
                phase = (
                    29 if scroll else [0, 5, 15, 29][number(view["id"] + "sample") % 4]
                )
                samples.append(
                    dict(
                        view=view,
                        index=index * 30 + phase,
                        local_index=phase,
                        scroll=scroll,
                    )
                )
                for frame in range(30):
                    pixels = low
                    if scroll and frame < 10:
                        offset = 2 * (10 - frame)
                        pixels = low.copy()
                        pixels[24 + offset :] = low[24 : 540 - offset]
                        pixels[24 : 24 + offset] = low[24]
                    process.stdin.write(pixels.tobytes())
            process.stdin.close()
            code = process.wait(timeout=180)
        except BaseException:
            if process.stdin and not process.stdin.closed:
                process.stdin.close()
            process.wait(timeout=180)
            raise
    if code:
        raise RuntimeError(f"Encoder failed for {job['id']}; inspect {log}")
    wanted = {s["index"]: s for s in samples}
    pairs = []
    count = 0
    packet_bytes = 0
    with av.open(path) as container:
        for packet in container.demux(video=0):
            packet_bytes += packet.size
            for frame in packet.decode():
                if (
                    frame.is_corrupt
                    or frame.width != 960
                    or frame.height != 540
                    or frame.format.name not in ("yuv420p", "yuvj420p")
                    or frame.color_range != 2
                    or frame.colorspace != 1
                ):
                    raise ValueError("Unexpected decoded HEVC pixels/color")
                if abs(float(frame.pts * frame.time_base) - count / 30) > 0.0011:
                    raise ValueError("Nonaligned decoded timestamp")
                if count in wanted:
                    sample = wanted[count]
                    view = sample["view"]
                    rgb = frame.to_ndarray(format="rgb24")
                    image = folder / (view["id"] + "-lr.png")
                    Image.fromarray(rgb).save(image, compress_level=3)
                    pairs.append(
                        dict(
                            view,
                            lr=str(image.relative_to(root)),
                            lr_sha256=sha256(image),
                            lr_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                            encoded=str(path.relative_to(root)),
                            frame=count,
                            time_s=count / 30,
                            local_frame=sample["local_index"],
                            keyframe=bool(frame.key_frame),
                            requested_bps=job["bps"],
                            scroll_context=scroll,
                            decoded_pixel_format=frame.format.name,
                            validation_eligible=view["split"] != "train",
                        )
                    )
                count += 1
    if count != len(job["views"]) * 30 or len(pairs) != len(job["views"]):
        raise ValueError("Missing decoded training pairs")
    result = dict(
        job_binding=fingerprint(job),
        video_sha256=sha256(path),
        command=command,
        frames=count,
        elementary_video_bytes=packet_bytes,
        actual_average_video_bps=packet_bytes * 8 / (count / 30),
        vbv_underflows=log.read_text().count("VBV underflow"),
        pairs=pairs,
        finished=time.time(),
    )
    atomic_json(receipt, result)
    return result


def run(args):
    cv2.setNumThreads(1)
    root = args.root.resolve()
    index = root / args.index
    out = root / args.stage
    out.mkdir(exist_ok=True)
    plan_path = out / "plan.json"
    if not plan_path.exists():
        plan = make_plan(root, index, args.stage)
        atomic_json(plan_path, plan)
        (out / "runner.py").write_bytes(Path(__file__).read_bytes())
    plan = read(plan_path)
    if plan["index_sha256"] != sha256(index) or plan["script_sha256"] != sha256(
        Path(__file__)
    ):
        raise ValueError("Degradation inputs/runner changed")
    jobs = plan["jobs"][: args.limit] if args.limit else plan["jobs"]
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for n, result in enumerate(pool.map(lambda j: encode(root, plan, j), jobs), 1):
            results.append(result)
            if n % 10 == 0:
                print(
                    "ENCODE",
                    n,
                    "/",
                    len(jobs),
                    "pairs",
                    sum(len(r["pairs"]) for r in results),
                    flush=True,
                )
    pairs = [p for r in results for p in r["pairs"]]
    for result in results:
        for pair in result["pairs"]:
            pair.update(
                actual_episode_video_bps=result["actual_average_video_bps"],
                episode_vbv_underflows=result["vbv_underflows"],
            )
    manifest = dict(
        schema="isharescreen_text_training_pairs_v1",
        plan_sha256=sha256(plan_path),
        views_sha256=sha256(index),
        complete=len(jobs) == len(plan["jobs"]),
        pairs=pairs,
        episodes=len(results),
        count=len(pairs),
        rate_counts=dict(Counter(p["requested_bps"] for p in pairs)),
        keyframe_samples=sum(p["keyframe"] for p in pairs),
        split_counts=dict(Counter(p["split"] for p in pairs)),
        vbv_underflow_episodes=sum(r["vbv_underflows"] > 0 for r in results),
        scope="Decoded codec degradation; requested rates are not constrained-link feasibility evidence. Underflows are explicit training degradation metadata, not encoder-budget passes.",
        finished=time.time(),
    )
    atomic_json(out / "pairs.json", manifest)
    atomic_json(
        out / "COMPLETED.json",
        dict(pairs_sha256=sha256(out / "pairs.json"), complete=manifest["complete"]),
    )
    print(
        json.dumps({k: v for k, v in manifest.items() if k != "pairs"}, indent=2),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--index", default="dataset-views.json")
    p.add_argument("--stage", default="hevc-v1")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    run(p.parse_args())


if __name__ == "__main__":
    main()
