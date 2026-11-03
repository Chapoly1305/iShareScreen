"""Probe short candidate intervals; never transcode or admit training references."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw
import cv2

from benchmarks.super_resolution.training_state import atomic_json


def main():
    root = Path("datasets/sources/dvr-intake-2026-11-01")
    rows = json.loads((root / "INVENTORY.json").read_text())["files"]
    jobs = []
    for row in rows:
        if row["index"] in [17, 18, 19, 20, 34, 50, 76]:
            indices = range(5)
        elif row["index"] in [38, 75]:
            indices = [0, 2]
        elif row["index"] == 52:
            indices = [0, 1, 2]
        elif row["index"] == 56:
            indices = [2, 3, 4]
        elif row["index"] == 66:
            indices = [2, 3]
        else:
            continue
        for number in indices:
            jobs.append((row, number))
    folder = root / "motion-probes"
    folder.mkdir(exist_ok=True)

    def probe(job):
        row, n = job
        sample = row["samples"][n]
        center = sample["requested_time_s"]
        start = max(0, min(center - 2, row["duration_s"] - 4))
        width, height = row["width"], row["height"]
        # Remove top clock and most lower face-camera area from motion statistics.
        x, y = int(width * 0.03) // 2 * 2, int(height * 0.06) // 2 * 2
        w, h = int(width * 0.94) // 2 * 2, int(height * 0.60) // 2 * 2
        command = [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-threads",
            "2",
            "-ss",
            str(start),
            "-i",
            row["path"],
            "-an",
            "-t",
            "4",
            "-filter_threads",
            "1",
            "-vf",
            f"crop={w}:{h}:{x}:{y},fps=3,scale=480:240:flags=area",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "gray",
            "pipe:1",
        ]
        result = subprocess.run(command, capture_output=True, timeout=120)
        if result.returncode:
            raise RuntimeError(row["file"] + result.stderr.decode()[-1000:])
        images = np.frombuffer(result.stdout, np.uint8).reshape(-1, 240, 480)
        assert len(images) >= 10
        delta = np.abs(np.diff(images.astype(np.int16), axis=0))
        changed = (delta > 8).mean(axis=(1, 2))
        mad = delta.mean(axis=(1, 2))
        canvas = Image.new("RGB", (960, 2 * 265), "white")
        draw = ImageDraw.Draw(canvas)
        for i, k in enumerate(
            [0, len(images) // 3, 2 * len(images) // 3, len(images) - 1]
        ):
            x0 = i % 2 * 480
            y0 = i // 2 * 265
            canvas.paste(Image.fromarray(images[k]).convert("RGB"), (x0, y0 + 24))
            draw.text((x0 + 3, y0 + 3), f"{start + k / 3:.2f}s", fill="black")
        preview = folder / f"{row['index']:02d}-{n}.jpg"
        canvas.save(preview, quality=90)
        return dict(
            index=row["index"],
            file=row["file"],
            sample=n,
            center_s=center,
            probe_start_s=start,
            probe_duration_s=4,
            decoded_preview_frames=len(images),
            desktop_mean_abs_change=float(mad.mean()),
            peak_changed_fraction=float(changed.max()),
            desktop_activity="visible_change_in_probe"
            if changed.max() >= 0.01
            else "near_static_in_probe",
            preview=str(preview.relative_to(root)),
            limit="4-second probe only, upper desktop area; not proof that full recording or20-second candidate is static/moving. Viewpoint/screen updates also cause changes, not only scrolling.",
        )

    results = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(probe, j) for j in jobs]):
            results.append(future.result())
            print("PROBED", len(results), "/", len(jobs), flush=True)
    atomic_json(
        root / "MOTION-PROBES.json",
        dict(
            probes=sorted(results, key=lambda r: (r["index"], r["sample"])),
            new_training_started=False,
        ),
    )


if __name__ == "__main__":
    main()
