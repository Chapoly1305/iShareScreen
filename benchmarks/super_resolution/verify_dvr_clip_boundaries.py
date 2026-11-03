"""Independent PyAV source timestamp check in addition to full FFmpeg pixel hashes."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path

import av

from benchmarks.super_resolution.export_dvr_clips import frame_hashes
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


def source_frame(path, target):
    with av.open(str(path)) as video:
        stream = video.streams.video[0]
        stream.thread_count = 2
        video.seek(int(target / stream.time_base), stream=stream, backward=True)
        for frame in video.decode(stream):
            time = Fraction(frame.pts) * frame.time_base
            if time < target:
                continue
            assert time == target, (path, str(time), str(target))
            return dict(
                timestamp=str(time),
                pixel_md5=hashlib.md5(
                    frame.to_ndarray(format="yuv420p").tobytes()
                ).hexdigest(),
            )
    raise ValueError("Missing target source frame")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path("datasets/sources/dvr-clips-v1"))
    p.add_argument("--require-count", type=int, default=29)
    a = p.parse_args()
    root = a.root.resolve()
    receipts = sorted((root / "receipts").glob("*.json"))
    assert len(receipts) == a.require_count
    records = []
    for path in receipts:
        row = json.loads(path.read_text())
        fps = Fraction(row["fps"])
        hashes = frame_hashes(root / row["source_frame_md5"])
        points = []
        for offset in [0, row["reference_frames"] // 2, row["reference_frames"] - 1]:
            frame = source_frame(
                row["source_path"],
                Fraction(row["source_start_frame"] + offset, 1) / fps,
            )
            assert frame["pixel_md5"] == hashes[offset][1], (
                row["id"],
                offset,
                frame["pixel_md5"],
                hashes[offset][1],
            )
            points.append(dict(clip_frame=offset, **frame))
        records.append(
            dict(
                id=row["id"],
                passed=True,
                receipt_sha256=sha256(path),
                checked_source_frames=points,
            )
        )
        print("BOUNDARIES VERIFIED", row["id"], flush=True)
    atomic_json(
        root / "BOUNDARY-AUDIT.json",
        dict(
            passed=True,
            clips=len(records),
            independent_source_timestamp_and_pixel_checks=len(records) * 3,
            manifest_sha256=sha256(root / "manifest.json")
            if (root / "manifest.json").exists()
            else None,
            decoder="PyAV direct source seek, compares actual source PTS and raw YUV420 MD5 to FFmpeg excerpt hashes",
            records=records,
        ),
    )


if __name__ == "__main__":
    main()
