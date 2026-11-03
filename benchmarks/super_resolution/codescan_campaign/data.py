"""Deterministic whole-frame HEVC pairs; no test inputs are admitted."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import zipfile

import cv2
import numpy as np
from PIL import Image

from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement.models import sha256


RECIPE = dict(
    schema="codescan_codec_recipe_v1",
    display=[1920, 1080],
    input=[1280, 720],
    fps=30,
    frames=60,
    sample_frame=45,
    bitrate=600000,
    pixel_format="yuv420p",
    encoder="libx265",
    preset="fast",
    tune="zerolatency",
    gop=60,
    vbv=600000,
    color="full-range BT.709",
    rendering="Aspect-preserving fit to FHD, Lanczos, centered; padding is median corner RGB.",
    motion="Synthetic: first30 frames static, next30 translate vertically 0 to -30 reference pixels, corner-color fill. Not real scrolling.",
    sampling="One aligned P-frame at index45 per source; code-line anchors when valid; no flips or rotations.",
    loss_geometry="LR128 -> native256 -> area192 target; even LR crop origins for exact 3:2 reference alignment.",
)


def read(p):
    return json.loads(Path(p).read_text())


def source_rows(project):
    """Verify frozen manifests; never load the test sample list for preparation."""
    frozen = read(project / "datasets/sources/codescan/split-v1/FROZEN.json")
    rows = []
    for split in ("train", "validation"):
        name = f"datasets/{split}/codescan/manifest.json"
        path = project / name
        assert sha256(path) == frozen["files_sha256"][name], name
        manifest = read(path)
        for row in manifest["rows"]:
            assert row["split"] == split
            rows.append(row)
    return rows


def extract(project, root, rows):
    archive = project / "datasets/sources/codescan/codescan.zip"
    expected = read(project / "datasets/sources/codescan/split-v1/PLAN.json")[
        "archive_sha256"
    ]
    receipt = root / "EXTRACTED.json"
    raw = root / "raw"
    if receipt.exists():
        assert read(receipt)["archive_sha256"] == expected
        return raw
    assert sha256(archive) == expected, "Source archive changed"
    raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for index, row in enumerate(rows, 1):
            relative = Path(row["sample"])
            assert not relative.is_absolute() and ".." not in relative.parts
            for name in ("img.png", "annotations.json"):
                path = raw / relative / name
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and sha256(path) == row["files"][name]:
                    continue
                data = z.read("codescan/" + row["sample"] + "/" + name)
                assert hashlib.sha256(data).hexdigest() == row["files"][name]
                temp = path.with_suffix(".tmp")
                temp.write_bytes(data)
                temp.replace(path)
            if index % 1000 == 0:
                print("EXTRACTED", index, "/", len(rows), flush=True)
    atomic_json(
        receipt,
        dict(
            archive_sha256=expected,
            samples=len(rows),
            roles=["img.png", "annotations.json"],
        ),
    )
    return raw


def render(raw, row):
    path = raw / row["sample"] / "img.png"
    assert sha256(path) == row["files"]["img.png"]
    image = np.asarray(Image.open(path).convert("RGB"))
    h, w = image.shape[:2]
    scale = min(1920 / w, 1080 / h)
    nw, nh = round(w * scale), round(h * scale)
    background = np.median(
        np.array([image[0, 0], image[0, -1], image[-1, 0], image[-1, -1]]), axis=0
    ).astype(np.uint8)
    canvas = np.empty((1080, 1920, 3), np.uint8)
    canvas[:] = background
    x, y = (1920 - nw) // 2, (1080 - nh) // 2
    canvas[y : y + nh, x : x + nw] = cv2.resize(
        image, (nw, nh), interpolation=cv2.INTER_LANCZOS4
    )
    # Effective x/y scales account for integer image dimensions.
    sx, sy = nw / w, nh / h
    shift = -round(30 * (45 - 30) / 29)
    annotation = raw / row["sample"] / "annotations.json"
    assert sha256(annotation) == row["files"]["annotations.json"]
    anchors = []
    for ann in read(annotation)["annotations"]:
        for line in ann.get("lines_data", []):
            if not (line.get("text") or "").strip():
                continue
            lx, ly = line["x"] * sx + x, line["y"] * sy + y + shift
            lw, lh = (
                line.get("code_width", line.get("width", 0)) * sx,
                line["height"] * sy,
            )
            if lw > 0 and lh > 0 and ly + lh > 0 and ly < 1080:
                anchors.append(
                    [max(0, lx), max(0, ly), min(1920, lx + lw), min(1080, ly + lh)]
                )
    anchors = [a for a in anchors if a[2] > a[0] and a[3] > a[1]]
    return (
        canvas,
        background,
        dict(
            source_size=[w, h],
            render_size=[nw, nh],
            scale_xy=[sx, sy],
            offset_xy=[x, y],
            sample_shift_y=shift,
            content_box=[x, max(0, y + shift), x + nw, min(1080, y + nh + shift)],
            line_boxes=anchors,
        ),
    )


def frame_at(canvas, background, index):
    shift = 0 if index < 30 else -round(30 * (index - 30) / 29)
    frame = np.empty_like(canvas)
    frame[:] = background
    if shift == 0:
        frame[:] = canvas
    else:
        frame[: 1080 + shift] = canvas[-shift:]
    return frame


def encode_sample(root, raw, row, binding):
    folder = root / "pairs" / row["split"] / row["id"]
    receipt = folder / "COMPLETED.json"
    if receipt.exists():
        result = read(receipt)
        assert result["binding"] == binding
        for role in ("lr", "hr"):
            assert sha256(root / result[role]) == result[role + "_sha256"]
        return result
    folder.mkdir(parents=True, exist_ok=True)
    canvas, bg, transform = render(raw, row)
    video = folder / "encoded.mp4"
    temp = folder / "encoded.tmp.mp4"
    params = "pools=1:frame-threads=1:keyint=60:min-keyint=1:bframes=0:rc-lookahead=0:bitrate=600:vbv-maxrate=600:vbv-bufsize=600:vbv-init=0.5:strict-cbr=1:hrd=1:log-level=info"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1920x1080",
        "-r",
        "30",
        "-i",
        "pipe:0",
        "-an",
        "-filter_threads",
        "1",
        "-vf",
        "scale=1280:720:flags=lanczos:in_range=full:out_range=full:out_color_matrix=bt709",
        "-c:v",
        "libx265",
        "-preset",
        "fast",
        "-tune",
        "zerolatency",
        "-b:v",
        "600000",
        "-minrate",
        "600000",
        "-maxrate",
        "600000",
        "-bufsize",
        "600000",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "pc",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-x265-params",
        params,
        "-tag:v",
        "hvc1",
        str(temp),
    ]
    started = time.time()
    with (folder / "encode.log").open("wb") as log:
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=log, stderr=log)
        try:
            for i in range(60):
                process.stdin.write(frame_at(canvas, bg, i).tobytes())
            process.stdin.close()
            code = process.wait(timeout=180)
            if code:
                raise RuntimeError(f"ffmpeg failed {code}: {folder}")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    temp.replace(video)
    info = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_packets",
                "-show_entries",
                "stream=width,height,pix_fmt,color_range,color_space,has_b_frames:packet=size,pts_time",
                "-of",
                "json",
                str(video),
            ]
        )
    )
    stream = info["streams"][0]
    packets = info["packets"]
    assert len(packets) == 60
    assert (stream["width"], stream["height"], stream["pix_fmt"]) == (
        1280,
        720,
        "yuvj420p",
    ) or (stream["width"], stream["height"], stream["pix_fmt"]) == (
        1280,
        720,
        "yuv420p",
    )
    assert (
        stream["has_b_frames"] == 0
        and stream["color_range"] == "pc"
        and stream["color_space"] == "bt709"
    )
    retained = [p for p in packets if float(p["pts_time"]) >= 1 - 1e-6]
    rate = sum(int(p["size"]) * 8 for p in retained) / (len(retained) / 30)
    underflows = (folder / "encode.log").read_text().count("VBV underflow")
    assert underflows == 0 and rate <= 630000, (row["sample"], rate, underflows)
    rgb = subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            str(video),
            "-vf",
            "select=eq(n\\,45)",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-threads",
            "1",
            "pipe:1",
        ]
    )
    low = np.frombuffer(rgb, np.uint8).reshape(720, 1280, 3)
    high = frame_at(canvas, bg, 45)
    Image.fromarray(low).save(folder / "lr.png")
    Image.fromarray(high).save(folder / "hr.png")
    result = dict(
        id=row["id"],
        sample=row["sample"],
        split=row["split"],
        repository=row["repository"],
        group_id=row["group_id"],
        binding=binding,
        runner_sha256=sha256(Path(__file__)),
        source_sha256=row["files"]["img.png"],
        transform=transform,
        actual_bps_after_first_second=rate,
        vbv_underflows=underflows,
        encoder_command=cmd,
        stream=stream,
        seconds=time.time() - started,
        video_sha256=sha256(video),
    )
    for role, array in [("lr", low), ("hr", high)]:
        path = folder / (role + ".png")
        result[role] = str(path.relative_to(root))
        result[role + "_sha256"] = sha256(path)
        result[role + "_pixel_sha256"] = hashlib.sha256(
            f"RGB:{array.shape[1]}x{array.shape[0]}:".encode() + array.tobytes()
        ).hexdigest()
    result["video"] = str(video.relative_to(root))
    atomic_json(receipt, result)
    return result


def frozen_recipe(root, recipe):
    protocol = root / "RECIPE.json"
    if not protocol.exists():
        atomic_json(protocol, recipe)
        return recipe
    original = read(protocol)
    if original != recipe:
        repair = read(root / "RUNNER-REPAIR.json")
        assert repair["original_runner_sha256"] == original["runner_sha256"]
        assert repair["repaired_runner_sha256"] == recipe["runner_sha256"]
        assert {k: v for k, v in original.items() if k != "runner_sha256"} == {
            k: v for k, v in recipe.items() if k != "runner_sha256"
        }, "Frozen codec recipe changed"
    return original


def prepare(project, root, rows, workers=4, smoke=False):
    cv2.setNumThreads(1)
    root.mkdir(parents=True, exist_ok=True)
    recipe = dict(
        RECIPE,
        ffmpeg_version=subprocess.check_output(
            ["ffmpeg", "-version"], text=True
        ).splitlines()[0],
        runner_sha256=sha256(Path(__file__)),
    )
    recipe = frozen_recipe(root, recipe)
    binding = fingerprint(recipe)
    protocol = root / "RECIPE.json"
    raw = extract(project, root, source_rows(project))
    selected = rows
    if smoke:
        selected = []
        for split, count in [("train", 128), ("validation", 16)]:
            candidates = sorted(
                (r for r in rows if r["split"] == split),
                key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest(),
            )
            selected.extend(candidates[:count])
    receipt = root / ("SMOKE-DATA.json" if smoke else "DATA.json")
    if receipt.exists():
        data = read(receipt)
        assert data["binding"] == binding and {r["id"] for r in data["rows"]} == {
            r["id"] for r in selected
        }
        for row in data["rows"]:
            for role in ("lr", "hr"):
                assert sha256(root / row[role]) == row[role + "_sha256"]
        return receipt
    results = []
    started = time.time()
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = [pool.submit(encode_sample, root, raw, row, binding) for row in selected]
    try:
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if len(results) % 20 == 0 or len(results) == len(selected):
                atomic_json(
                    root / "PREPARATION-STATUS.json",
                    dict(
                        stage="smoke" if smoke else "full",
                        complete=len(results),
                        total=len(selected),
                        elapsed_s=time.time() - started,
                    ),
                )
                print("PAIRS", len(results), "/", len(selected), flush=True)
    except BaseException:
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    results.sort(key=lambda row: (row["split"], row["id"]))
    # Recheck generated data: rendering/encoding can expose duplicates absent in original PNGs.
    sets = {
        s: {
            k: set()
            for k in (
                "repository",
                "group_id",
                "source_sha256",
                "lr_sha256",
                "hr_sha256",
                "lr_pixel_sha256",
                "hr_pixel_sha256",
            )
        }
        for s in ("train", "validation")
    }
    for row in results:
        for key in sets[row["split"]]:
            sets[row["split"]][key].add(row[key])
    overlap = {k: len(sets["train"][k] & sets["validation"][k]) for k in sets["train"]}
    assert not any(overlap.values()), overlap
    atomic_json(
        receipt,
        dict(
            schema="codescan_codec_pairs_v1",
            binding=binding,
            recipe_sha256=sha256(protocol),
            complete=True,
            runner_repair_sha256=sha256(root / "RUNNER-REPAIR.json")
            if (root / "RUNNER-REPAIR.json").exists()
            else None,
            counts={s: sum(r["split"] == s for r in results) for s in sets},
            cross_split_overlap=overlap,
            rows=results,
        ),
    )
    return receipt
