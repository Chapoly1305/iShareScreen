"""Frame-exact DVR reference excerpts with lossless decoded-pixel verification."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import time

from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement.models import sha256


def read(path):
    return json.loads(Path(path).read_text())


def frame_interval(start_s, duration_s, fps):
    fps = Fraction(fps)
    start = math.ceil(Fraction(str(start_s)) * fps)
    count = Fraction(str(duration_s)) * fps
    assert count.denominator == 1
    return start, start + int(count)


def frame_hashes(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = [v.strip() for v in line.split(",")]
        assert len(parts) == 6
        rows.append((int(parts[4]), parts[5]))
    return rows


def command(args, log):
    with log.open("w") as stream:
        subprocess.run(
            args, check=True, stdout=stream, stderr=subprocess.STDOUT, timeout=600
        )


def export_one(root, record, source, binding):
    name = record["id"]
    receipt = root / "receipts" / f"{name}.json"
    if receipt.exists():
        value = read(receipt)
        assert value["binding"] == binding
        for key in ["reference", "preview", "poster"]:
            assert sha256(root / value[key]) == value[key + "_sha256"]
        return value
    fps = Fraction(source["fps"])
    start, end = frame_interval(
        record["start_s"], record["end_s"] - record["start_s"], fps
    )
    count = end - start
    anchor = max(0, start // fps - 2)
    anchor = int(anchor)
    offset = start - int(anchor * fps)
    assert (
        source["pix_fmt"] == "yuv420p"
        and source["color_range"] == "tv"
        and source["color_space"] == "bt709"
    )
    reference = root / "references" / f"{name}__{record['content']}.mkv"
    temporary = reference.with_name(reference.stem + ".tmp.mkv")
    preview = root / "previews" / f"{name}.mp4"
    poster = root / "posters" / f"{name}.jpg"
    source_hash = root / "frame-hashes" / f"{name}.source.framemd5"
    ref_hash = root / "frame-hashes" / f"{name}.reference.framemd5"
    filters = (
        f"trim=start_frame={offset}:end_frame={offset + count},setpts=PTS-STARTPTS"
    )
    started = time.time()
    # Integer-second input seek, followed by an exact count of source-CFR frames.
    base = [
        "ffmpeg",
        "-hide_banner",
        "-v",
        "error",
        "-nostdin",
        "-y",
        "-threads",
        "2",
        "-ss",
        str(anchor),
        "-i",
        record["path"],
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
        "-filter_threads",
        "1",
        "-vf",
        filters,
        "-frames:v",
        str(count),
    ]
    cmd = base + [
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-coder",
        "1",
        "-context",
        "1",
        "-g",
        "1",
        "-slicecrc",
        "1",
        "-pix_fmt",
        "yuv420p",
        "-color_range",
        "tv",
        "-colorspace",
        "bt709",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-fps_mode",
        "passthrough",
        "-threads",
        "2",
        str(temporary),
    ]
    command(cmd, root / "logs" / f"{name}.export.log")
    temporary.replace(reference)
    command(
        base + ["-pix_fmt", "yuv420p", "-f", "framemd5", str(source_hash)],
        root / "logs" / f"{name}.source-hash.log",
    )
    command(
        [
            "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-threads",
            "2",
            "-i",
            str(reference),
            "-map",
            "0:v:0",
            "-an",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "framemd5",
            str(ref_hash),
        ],
        root / "logs" / f"{name}.reference-hash.log",
    )
    original = frame_hashes(source_hash)
    decoded = frame_hashes(ref_hash)
    assert len(original) == len(decoded) == count, (
        name,
        len(original),
        len(decoded),
        count,
    )
    assert original == decoded, "Lossless pixel mismatch " + name
    stream = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(reference),
            ]
        )
    )
    video = stream["streams"][0]
    duration = float(stream["format"]["duration"])
    assert (video["width"], video["height"]) == (source["width"], source["height"])
    assert Fraction(video["avg_frame_rate"]) == fps and abs(duration - 20) < 0.002
    assert (
        video["pix_fmt"] == "yuv420p"
        and video["color_range"] == "tv"
        and video["color_space"] == "bt709"
    )
    command(
        [
            "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-threads",
            "2",
            "-i",
            str(reference),
            "-an",
            "-filter_threads",
            "1",
            "-vf",
            "fps=30,scale=1280:-2:flags=lanczos",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-threads",
            "2",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(preview),
        ],
        root / "logs" / f"{name}.preview.log",
    )
    command(
        [
            "ffmpeg",
            "-hide_banner",
            "-v",
            "error",
            "-nostdin",
            "-y",
            "-threads",
            "2",
            "-ss",
            "10",
            "-i",
            str(reference),
            "-an",
            "-frames:v",
            "1",
            "-filter_threads",
            "1",
            "-vf",
            "scale=960:-2:flags=lanczos",
            "-q:v",
            "2",
            str(poster),
        ],
        root / "logs" / f"{name}.poster.log",
    )
    row = dict(
        id=name,
        content=record["content"],
        binding=binding,
        source_file=record["file"],
        source_path=record["path"],
        source_sha256=record["source_sha256"],
        requested_start_s=record["start_s"],
        requested_end_s=record["end_s"],
        source_start_frame=start,
        source_end_frame_exclusive=end,
        actual_source_start_s=float(start / fps),
        actual_source_end_s=float(end / fps),
        rounding="First CFR source frame at or after requested start; exactly20s. End is exclusive.",
        reference_start_s=0,
        reference_duration_s=duration,
        reference_frames=count,
        fps=str(fps),
        source_size=[source["width"], source["height"]],
        reference_codec="ffv1",
        pixel_format="yuv420p",
        color_range="tv",
        color_space="bt709",
        native_source_lossy=True,
        decoded_pixel_identity_passed=True,
        source_frame_md5=str(source_hash.relative_to(root)),
        reference_frame_md5=str(ref_hash.relative_to(root)),
        frame_hashes_sha256={
            "source": sha256(source_hash),
            "reference": sha256(ref_hash),
        },
        split=None,
        training_admitted=False,
        preview_policy="Lossy H264 CRF18,1280px wide preserving aspect ratio,30fps; browsing only. Use reference MKV for training/evaluation.",
        audio="omitted; visual-only corpus",
        elapsed_s=time.time() - started,
        export_command=cmd,
    )
    for key, path in [
        ("reference", reference),
        ("preview", preview),
        ("poster", poster),
    ]:
        row[key] = str(path.relative_to(root))
        row[key + "_sha256"] = sha256(path)
        row[key + "_bytes"] = path.stat().st_size
    atomic_json(receipt, row)
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--intake", type=Path, default=Path("datasets/sources/dvr-intake-2026-11-01")
    )
    p.add_argument("--root", type=Path, default=Path("datasets/sources/dvr-clips-v1"))
    p.add_argument("--workers", type=int, default=3)
    a = p.parse_args()
    intake = a.intake.resolve()
    root = a.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name in [
        "references",
        "previews",
        "posters",
        "frame-hashes",
        "logs",
        "receipts",
    ]:
        (root / name).mkdir(exist_ok=True)
    plan = read(intake / "SELECTION.json")
    inventory = read(intake / "INVENTORY.json")
    sources = {v["file"]: v for v in inventory["files"]}
    protocol = dict(
        schema="dvr_exact_reference_clips_v1",
        selection_sha256=sha256(intake / "SELECTION.json"),
        intake_audit_sha256=sha256(intake / "AUDIT.json"),
        script_sha256=sha256(Path(__file__)),
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[
            0
        ],
        policy="29 frame-exact20s excerpts, native size/fps/decoded420 pixels with lossless FFV1; per-frame MD5 equality. No model or new training; splits unassigned.",
        limits="FFV1 preserves decoded pixels of already-lossy sources, not pristine acquisition truth. Crops remain unmasked; timestamp/camera/text quality and content leakage need admission checks.",
    )
    binding = fingerprint(protocol)
    if (root / "PROTOCOL.json").exists():
        assert read(root / "PROTOCOL.json") == protocol
    else:
        atomic_json(root / "PROTOCOL.json", protocol)
    locks = read(intake / "HASHES.json")
    hashes = {r["file"]: r for r in locks["files"]}
    selected_names = sorted({r["file"] for r in plan["priority_segments"]})
    # New export independently confirms the eight source bytes still match intake.
    with ThreadPoolExecutor(max_workers=2) as pool:

        def verify_source(name):
            path = Path(sources[name]["path"])
            before = path.stat()
            actual = sha256(path)
            after = path.stat()
            assert actual == hashes[name]["sha256"] and (
                before.st_size,
                before.st_mtime_ns,
            ) == (after.st_size, after.st_mtime_ns), name
            return dict(
                file=name,
                sha256=actual,
                bytes=before.st_size,
                mtime_ns=before.st_mtime_ns,
            )

        verified = list(pool.map(verify_source, selected_names))
    atomic_json(root / "SOURCE-REVERIFIED.json", dict(passed=True, sources=verified))
    print("SOURCE HASHES VERIFIED", len(verified), flush=True)
    rows = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed(
            [
                pool.submit(export_one, root, row, sources[row["file"]], binding)
                for row in plan["priority_segments"]
            ]
        ):
            row = f.result()
            rows.append(row)
            atomic_json(
                root / "STATUS.json",
                dict(
                    completed=len(rows),
                    total=29,
                    latest=row["id"],
                    elapsed_s=time.time() - started,
                ),
            )
            print(
                "EXPORTED",
                len(rows),
                "/29",
                row["id"],
                row["reference_frames"],
                "frames",
                flush=True,
            )
    rows.sort(key=lambda r: r["id"])
    assert len(rows) == 29 and len({r["reference_sha256"] for r in rows}) == 29
    atomic_json(
        root / "manifest.json",
        dict(
            schema="dvr_lossless_reference_clips_v1",
            protocol_sha256=sha256(root / "PROTOCOL.json"),
            complete=True,
            count=29,
            total_duration_s=sum(r["reference_duration_s"] for r in rows),
            source_files=8,
            training_admitted=False,
            rows=rows,
        ),
    )
    sections = []
    for row in rows:
        sections.append(
            f'<article data-content="{html.escape(row["content"])}"><h2>{row["id"]} · {html.escape(row["content"])}</h2><p>{html.escape(row["source_file"])} · source {row["actual_source_start_s"]:.3f}–{row["actual_source_end_s"]:.3f}s · {row["source_size"]} · {row["fps"]}fps</p><video controls preload="none" poster="{row["poster"]}" src="{row["preview"]}"></video><p><a href="{row["reference"]}">无损参考MKV（评估用）</a> · <a href="{row["preview"]}">MP4预览（非参考）</a> · <a href="receipts/{row["id"]}.json">来源/帧校验</a></p></article>'
        )
    options = "".join(
        f"<option>{html.escape(v)}</option>"
        for v in sorted({r["content"] for r in rows})
    )
    (root / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>DVR clips v1</title><style>body{font:16px system-ui;max-width:1300px;margin:24px auto;padding:0 16px;background:#f4f5f7}article{background:white;padding:18px;margin:18px 0;border-radius:10px}video{width:100%;max-height:65vh}select{font:inherit;padding:8px}p{line-height:1.6}</style><h1>DVR首批29段参考视频</h1><p>29×20秒，共9分40秒。原始尺寸/帧率和解码像素保存在FFV1参考文件中，逐帧校验通过。下方播放的是缩小的有损MP4预览，不作为画质评估参考。原来源已有损；暂无train/validation/test分配，不要将这些片段直接混入训练或当作独立最终测试。</p><select id="filter"><option value="all">全部片段</option>'
        + options
        + "</select>"
        + "".join(sections)
        + '<script>document.getElementById("filter").onchange=e=>document.querySelectorAll("article").forEach(a=>{a.hidden=e.target.value!=="all"&&a.dataset.content!==e.target.value;if(a.hidden)a.querySelector("video").pause()})</script>'
    )
    import csv

    with (root / "clips.csv").open("w") as f:
        fields = [
            "id",
            "content",
            "source_file",
            "actual_source_start_s",
            "actual_source_end_s",
            "source_start_frame",
            "source_end_frame_exclusive",
            "reference_duration_s",
            "reference_frames",
            "fps",
            "reference",
            "reference_sha256",
            "preview",
            "source_sha256",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows({k: r[k] for k in fields} for r in rows)
    audit = dict(
        passed=True,
        clips=29,
        total_source_files=8,
        total_duration_s=sum(r["reference_duration_s"] for r in rows),
        total_frames=sum(r["reference_frames"] for r in rows),
        every_frame_pixel_hash_matches_source=True,
        source_rehashed=True,
        source_originals_modified=False,
        training_started=False,
        split_assignment=None,
        total_reference_bytes=sum(r["reference_bytes"] for r in rows),
        total_preview_bytes=sum(r["preview_bytes"] for r in rows),
        manifest_sha256=sha256(root / "manifest.json"),
    )
    atomic_json(root / "AUDIT.json", audit)
    (root / "README.md").write_text(
        """# DVR reusable reference clips v1\n\nOpen index.html for video previews; use references/ for training/evaluation inputs after separate admission. previews/ contains lossy browser-only H264 copies. manifest.json and clips.csv map every clip to its immutable source filename, SHA256 and exact CFR frame interval. No audio, crop, stretch or fps conversion is applied to references.\n\nThe requested timestamps are rounded forward to the next source frame (at most1/30 or1/60second) and exactly20seconds are retained. LosslessFFV1 keeps every decoded YUV420 byte, native2240x1400/30fps or3440x1440/60fps, limited-rangeBT.709. Source/reference per-frame MD5 lists match. SourceMOVs remain unchanged. FFV1 introduces no extra loss but cannot undo original source compression.\n\nThese29 excerpts represent selected material for reuse, not admitted training data or a frozen final test. Split is unassigned. Keep derivatives from the same recording/session/document/project together; cross-check existing validation/test content before admission. Review crop masks for timestamps/camera overlays and native text quality before codec pair preparation. Preserve frozenCodeSCAN and currentMac validation. No new training was launched.\n"""
    )
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
