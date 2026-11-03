"""Generate matched HEVC inputs for every frozen independent evaluation case."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import fcntl
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import time

import av
from PIL import Image

from benchmarks.super_resolution.bitrate_poc import packet_stats
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def encode_one(
    root: Path, protocol: dict, case: dict, width: int, height: int, bps: int
) -> dict:
    name = f"{width}x{height}-{bps // 1000}k"
    folder = root / "evaluation-inputs" / case["id"] / name
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / "COMPLETED.json"
    reference = root / case["hr"]
    if sha256(reference) != case["hr_sha256"]:
        raise ValueError(f"Source-rendered reference changed: {case['id']}")
    binding = dict(
        protocol_sha256=sha256(root / "EVAL_PROTOCOL.json"),
        case=case["id"],
        reference_sha256=case["hr_sha256"],
        geometry=[width, height],
        requested_bps=bps,
    )
    if receipt.exists():
        row = read(receipt)
        if (
            row["binding"] != binding
            or sha256(folder / "encoded.mkv") != row["video_sha256"]
        ):
            raise ValueError(f"Completed encode differs: {folder}")
        for frame in row["sampled_frames"]:
            if sha256(folder / frame["path"]) != frame["sha256"]:
                raise ValueError(f"Sampled decode changed: {folder}")
        return row
    if (folder / "encoded.mkv").exists():
        raise ValueError(f"Incomplete encode requires explicit inspection: {folder}")
    kbps = bps // 1000
    params = (
        f"pools=1:frame-threads=1:keyint=60:min-keyint=60:scenecut=0:"
        f"bframes=0:rc-lookahead=0:bitrate={kbps}:vbv-maxrate={kbps}:"
        f"vbv-bufsize={kbps}:vbv-init=0.5:strict-cbr=1:hrd=1:log-level=info"
    )
    command = [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-loop",
        "1",
        "-framerate",
        "30",
        "-i",
        str(reference),
        "-frames:v",
        "120",
        "-an",
        "-vf",
        f"scale={width}:{height}:flags=lanczos:in_range=full:out_range=full:out_color_matrix=bt709",
        "-c:v",
        "libx265",
        "-preset",
        "fast",
        "-tune",
        "zerolatency",
        "-b:v",
        str(bps),
        "-minrate",
        str(bps),
        "-maxrate",
        str(bps),
        "-bufsize",
        str(bps),
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
        str(folder / "encoded.mkv"),
    ]
    with (folder / "encode.log").open("w") as log:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=log)
    samples = []
    count = 0
    packet_bytes = 0
    retained_bytes = 0
    with av.open(str(folder / "encoded.mkv")) as container:
        for packet in container.demux(video=0):
            if packet.pts is not None:
                packet_bytes += packet.size
                if float(packet.pts * packet.time_base) >= 1.0:
                    retained_bytes += packet.size
            for frame in packet.decode():
                if (
                    frame.is_corrupt
                    or frame.width != width
                    or frame.height != height
                    or frame.format.name not in ("yuv420p", "yuvj420p")
                ):
                    raise ValueError(f"Invalid decoded frame: {folder}")
                if abs(float(frame.pts * frame.time_base) - count / 30) > 0.0011:
                    raise ValueError(f"Decoded frame timestamp differs: {folder}")
                if count in (60, 90):
                    path = f"frame-{count:03d}.png"
                    Image.fromarray(frame.to_ndarray(format="rgb24")).save(
                        folder / path
                    )
                    samples.append(
                        dict(
                            index=count,
                            time_s=count / 30,
                            keyframe=bool(frame.key_frame),
                            path=path,
                            sha256=sha256(folder / path),
                        )
                    )
                count += 1
    if count != 120 or len(samples) != 2 or not samples[0]["keyframe"]:
        raise ValueError(f"Unexpected GOP/frame count: {folder}")
    measured = retained_bytes * 8 / 3.0
    underflows = (folder / "encode.log").read_text().count("VBV underflow")
    row = dict(
        binding=binding,
        split=case["split"],
        source_group=case["source_group"],
        content_category=case["content_category"],
        video="encoded.mkv",
        video_sha256=sha256(folder / "encoded.mkv"),
        command=command,
        log_sha256=sha256(folder / "encode.log"),
        frames=count,
        packet_bytes=packet_bytes,
        retained_packet_bytes=retained_bytes,
        measured_video_bps_after_first_second=measured,
        vbv_underflows=underflows,
        encoder_eligible=measured <= bps * 1.05 and underflows == 0,
        sampled_frames=samples,
        finished=time.time(),
    )
    atomic_json(receipt, row)
    return row


def run(root: Path, workers: int, limit_cases: int | None) -> None:
    out = root / "evaluation-inputs"
    out.mkdir(exist_ok=True)
    with (out / "TASK.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = read(root / "EVAL_PROTOCOL.json")
        verified = read(root / "protocol-verification.json")
        if verified["protocol_sha256"] != sha256(root / "EVAL_PROTOCOL.json"):
            raise ValueError("Independent protocol changed after verification")
        cases = (
            protocol["cases"][:limit_cases]
            if limit_cases is not None
            else protocol["cases"]
        )
        jobs = [
            (case, width, height, bps)
            for case in cases
            for width, height in protocol["main_grid"]["base_geometry"]
            for bps in protocol["main_grid"]["requested_video_bps"]
        ]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = []
            for index, row in enumerate(
                pool.map(lambda job: encode_one(root, protocol, *job), jobs), 1
            ):
                rows.append(row)
                if index % 12 == 0 or index == len(jobs):
                    atomic_json(
                        out / "status.json",
                        dict(
                            phase="encoding",
                            completed=index,
                            full_grid=len(protocol["cases"]) * 6,
                            eligible=sum(r["encoder_eligible"] for r in rows),
                            updated=time.time(),
                        ),
                    )
                    print("EVAL ENCODE", index, "/", len(jobs), flush=True)
        manifest = dict(
            schema="isharescreen_low_bandwidth_eval_inputs_v1",
            protocol_sha256=sha256(root / "EVAL_PROTOCOL.json"),
            complete=len(cases) == len(protocol["cases"]) and len(rows) == 648,
            counts=dict(Counter(row["split"] for row in rows)),
            eligible=sum(row["encoder_eligible"] for row in rows),
            ineligible=len(rows) - sum(row["encoder_eligible"] for row in rows),
            rows=rows,
            runner_sha256=sha256(Path(__file__)),
            finished=time.time(),
        )
        path = out / (
            "smoke-manifest.json" if limit_cases is not None else "manifest.json"
        )
        atomic_json(path, manifest)
        if manifest["complete"]:
            atomic_json(
                out / "INPUTS-COMPLETED.json",
                dict(
                    manifest_sha256=sha256(path), count=len(rows), finished=time.time()
                ),
            )
        print("EVAL INPUTS", len(rows), "complete", manifest["complete"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-cases", type=int)
    args = parser.parse_args()
    run(args.root.resolve(), args.workers, args.limit_cases)


if __name__ == "__main__":
    main()
