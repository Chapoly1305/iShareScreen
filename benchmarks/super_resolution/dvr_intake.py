"""Read-only DVR intake: metadata, stable SHA256 and sampled visual evidence."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256


FRACTIONS = (0.05, 0.25, 0.5, 0.75, 0.95)


def inspect(path, index, root):
    before = path.stat()
    folder = root / "samples" / f"{index:02d}"
    folder.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    (folder / "probe.json").write_text(process.stdout)
    (folder / "probe-errors.log").write_text(process.stderr)
    if process.returncode:
        return dict(
            index=index,
            file=path.name,
            path=str(path),
            probe_ok=False,
            error=process.stderr,
        )
    data = json.loads(process.stdout)
    video = next(s for s in data["streams"] if s["codec_type"] == "video")
    duration = float(video.get("duration") or data["format"]["duration"])
    samples = []
    for number, fraction in enumerate(FRACTIONS):
        timestamp = min(max(0, duration * fraction), max(0, duration - 0.2))
        target = folder / f"{number}.png"
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-threads",
            "2",
            "-ss",
            str(timestamp),
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-an",
            "-frames:v",
            "1",
            "-filter_threads",
            "1",
            "-pix_fmt",
            "rgb24",
            "-y",
            str(target),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        (folder / f"{number}-decode.log").write_text(result.stderr)
        row = dict(
            number=number,
            fraction=fraction,
            requested_time_s=timestamp,
            decode_ok=result.returncode == 0 and target.exists(),
        )
        if row["decode_ok"]:
            with Image.open(target) as image:
                rgb = np.asarray(image.convert("RGB"))
                size = image.size
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(
                np.float32
            )
            dct = cv2.dct(small)[:8, :8].flatten()[1:]
            phash = np.packbits(dct > np.median(dct)).tobytes().hex()
            row.update(
                image=str(target.relative_to(root)),
                image_sha256=sha256(target),
                size=list(size),
                rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                phash=phash,
                gray_std=float(gray.std()),
                black_fraction=float((gray < 8).mean()),
                white_fraction=float((gray > 247).mean()),
                laplacian_variance=float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            )
        else:
            row["error"] = result.stderr[-1000:]
        samples.append(row)
    after = path.stat()
    stable = (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    return dict(
        index=index,
        file=path.name,
        path=str(path),
        probe_ok=True,
        bytes=before.st_size,
        mtime_ns=before.st_mtime_ns,
        stable=stable,
        duration_s=duration,
        width=video["width"],
        height=video["height"],
        codec=video["codec_name"],
        pix_fmt=video.get("pix_fmt"),
        fps=video.get("avg_frame_rate"),
        video_bps=int(video.get("bit_rate", 0)),
        color_range=video.get("color_range"),
        color_space=video.get("color_space"),
        encoder=video.get("tags", {}).get("encoder"),
        creation_time=video.get("tags", {}).get("creation_time"),
        samples=samples,
        sampled_decode_ok=all(s["decode_ok"] for s in samples),
        full_decode_verified=False,
    )


def sheets(root, rows):
    folder = root / "contact-sheets"
    folder.mkdir(exist_ok=True)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    for offset in range(0, len(rows), 8):
        subset = rows[offset : offset + 8]
        canvas = Image.new("RGB", (1600, len(subset) * 180), "white")
        draw = ImageDraw.Draw(canvas)
        for i, row in enumerate(subset):
            label = f"{row['index']:02d}  {row['file']} | {row.get('width')}x{row.get('height')} | {row.get('duration_s', 0) / 60:.1f}min | {row.get('video_bps', 0) / 1e6:.2f}Mbps"
            draw.text((4, i * 180 + 2), label, fill="black", font=font)
            for sample in row.get("samples", []):
                x = sample["number"] * 320
                y = i * 180 + 25
                if sample["decode_ok"]:
                    image = Image.open(root / sample["image"]).convert("RGB")
                    image.thumbnail((318, 140))
                    canvas.paste(image, (x + (318 - image.width) // 2, y))
                draw.text(
                    (x + 4, i * 180 + 165),
                    f"{sample['requested_time_s'] / 60:.1f}min ({sample['fraction']:.0%})",
                    fill="black",
                    font=small,
                )
        canvas.save(folder / f"{offset // 8:02d}.jpg", quality=93)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["inspect", "hash"])
    p.add_argument("--source", type=Path, default=Path("DVR"))
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    source = a.source.resolve()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("*.mov"))
    if a.action == "hash":
        rows = []
        started = time.time()
        with ThreadPoolExecutor(max_workers=2) as pool:

            def hash_file(path):
                before = path.stat()
                value = sha256(path)
                after = path.stat()
                assert (before.st_size, before.st_mtime_ns) == (
                    after.st_size,
                    after.st_mtime_ns,
                ), path
                return dict(
                    file=path.name,
                    bytes=before.st_size,
                    mtime_ns=before.st_mtime_ns,
                    sha256=value,
                )

            for f in as_completed([pool.submit(hash_file, path) for path in files]):
                rows.append(f.result())
                atomic_json(
                    root / "HASH-PROGRESS.json",
                    dict(
                        completed=len(rows),
                        total=len(files),
                        elapsed_s=time.time() - started,
                    ),
                )
                print("HASHED", len(rows), "/", len(files), flush=True)
        groups = {}
        for row in rows:
            groups.setdefault(row["sha256"], []).append(row["file"])
        prior = json.loads(
            Path(
                "datasets/validation/mac/processed_recordings/manifest.json"
            ).read_text()
        )
        old = {r["source_sha256"] for r in prior["clips"]}
        atomic_json(
            root / "HASHES.json",
            dict(
                complete=True,
                files=sorted(rows, key=lambda r: r["file"]),
                duplicates=[v for v in groups.values() if len(v) > 1],
                old_validation_original_sha256_matches=[
                    r["file"] for r in rows if r["sha256"] in old
                ],
                limits="File hashes do not rule out shared document/project content or re-encoded/subclipped copies.",
                elapsed_s=time.time() - started,
            ),
        )
        return
    rows = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(inspect, path, index, root) for index, path in enumerate(files)
        ]
        for f in as_completed(futures):
            row = f.result()
            rows.append(row)
            atomic_json(
                root / "INTAKE-PROGRESS.json",
                dict(
                    completed=len(rows),
                    total=len(files),
                    latest=row["file"],
                    elapsed_s=time.time() - started,
                ),
            )
            print("INSPECTED", len(rows), "/", len(files), row["file"], flush=True)
    rows.sort(key=lambda r: r["index"])
    sheets(root, rows)
    atomic_json(
        root / "INVENTORY.json",
        dict(
            source=str(source),
            files=rows,
            total_files=len(rows),
            total_bytes=sum(r.get("bytes", 0) for r in rows),
            total_hours=sum(r.get("duration_s", 0) for r in rows) / 3600,
            dimensions=dict(
                Counter(f"{r.get('width')}x{r.get('height')}" for r in rows)
            ),
            sampling=list(FRACTIONS),
            script_sha256=sha256(Path(__file__)),
            new_training_started=False,
            originals_modified=False,
            elapsed_s=time.time() - started,
        ),
    )


if __name__ == "__main__":
    main()
