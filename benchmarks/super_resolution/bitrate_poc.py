"""Small bitrate-matched FHD-restoration PoC using the existing SR adapters.

Encode contiguous excerpts before sampling. Keep the dataset's RGB reference,
HEVC 4:4:4/full BT.709 conversion, shared metrics and lossless-output checks.
"""

import argparse
from collections import defaultdict
from fractions import Fraction
import gc
import html
from itertools import zip_longest
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

import av
import cv2
import numpy as np
from PIL import Image
import torch

from benchmarks.super_resolution.gallery import crop_strip
from benchmarks.super_resolution.metrics import metrics
from benchmarks.super_resolution.prepare_video import probe
from benchmarks.super_resolution.train import perceptual_network
from benchmarks.super_resolution.training_state import atomic_json
from benchmarks.super_resolution.video_benchmark import LosslessWriter, verify_lossless
from benchmarks.super_resolution.video_perceptual import normalized_tensor
from isharescreen.enhancement import candidates
from isharescreen.enhancement.models import sha256


MODELS = ("quicksrnet", "plainusr", "rt4ksr", "efrlfn")
SCENES = ("light-code", "terminal-light", "web-document")


def run(command, log):
    with log.open("w") as handle:
        subprocess.run(command, check=True, stdout=handle, stderr=handle)


def packet_stats(path, fps, skip=0):
    data = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_entries",
                "packet=size,pts_time,dts_time,duration_time",
                "-of",
                "json",
                str(path),
            ],
            text=True,
        )
    )
    packets = data["packets"]
    duration = len(packets) / float(fps)
    windows = defaultdict(int)
    for packet in packets:
        windows[int(float(packet["pts_time"]))] += int(packet["size"]) * 8
    retained = [
        p for p in packets if float(p["pts_time"]) >= skip / float(fps) - 0.00001
    ]
    return dict(
        video_packet_bytes=sum(int(p["size"]) for p in packets),
        video_bps=sum(int(p["size"]) for p in packets) * 8 / duration,
        retained_video_bps=sum(int(p["size"]) for p in retained)
        * 8
        / ((len(packets) - skip) / float(fps)),
        duration_s=duration,
        one_second_pts_bins_bps=dict(sorted(windows.items())),
        container_bytes=path.stat().st_size,
        packets=len(packets),
    )


def prepare(args):
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    spec = json.loads(args.spec.read_text())
    scenes = [
        next(s for s in spec["scenes"] if s["id"] == name) for name in args.scenes
    ]
    manifest = dict(
        schema="sr_bitrate_poc_inputs_v1",
        command=sys.argv,
        spec=str(args.spec.resolve()),
        spec_sha256=sha256(args.spec),
        generator_sha256=sha256(Path(__file__)),
        scenes=[],
        fps="60",
        bitrate_bps=args.bitrate,
        frames=args.frames,
        skip=args.skip,
        sample_indices=sorted(
            set(range(args.skip, args.frames, 30)) | {args.frames - 1}
        ),
        reference="Decoded RGB8 proxy from original recording, not lossless acquisition",
        color="HEVC yuv444p, full-range BT.709, Lanczos reduction; matches prepare_video",
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[
            0
        ],
    )
    for scene in scenes:
        source = args.dataset / (scene["capture"] + ".mov")
        info = probe(source)
        if (info["width"], info["height"]) != (1920, 1080) or Fraction(
            info["avg_frame_rate"]
        ) != 60:
            raise ValueError("PoC requires the existing 1920x1080 60fps sources")
        start = scene["frame"] - args.frames // 2
        if start < 0 or start + args.frames > int(info["nb_read_frames"]):
            raise ValueError("Excerpt falls outside source")
        folder = args.output / scene["id"]
        folder.mkdir()
        reference = folder / "reference.mkv"
        commands = [
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-nostdin",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-vf",
                f"trim=start_frame={start}:end_frame={start + args.frames},setpts=PTS-STARTPTS,format=rgb24",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "libx264rgb",
                "-crf",
                "0",
                "-preset",
                "fast",
                "-g",
                "120",
                "-threads",
                "4",
                "-pix_fmt",
                "rgb24",
                str(reference),
            ]
        ]
        run(commands[0], folder / "reference.log")
        files = dict(
            reference=dict(
                path=str(reference.relative_to(args.output)), sha256=sha256(reference)
            )
        )
        for name, width, height in (("fhd", 1920, 1080), ("half", 960, 540)):
            target = folder / f"{name}-hevc-cbr.mp4"
            kbps = args.bitrate // 1000
            params = (
                f"pools=4:frame-threads=2:keyint=120:min-keyint=1:bitrate={kbps}:"
                f"vbv-maxrate={kbps}:vbv-bufsize={kbps}:vbv-init=0.5:strict-cbr=1:hrd=1:log-level=info"
            )
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-nostdin",
                "-i",
                str(reference),
                "-map",
                "0:v:0",
                "-an",
                "-vf",
                f"scale={width}:{height}:flags=lanczos:in_range=full:out_range=full:out_color_matrix=bt709",
                "-fps_mode",
                "passthrough",
                "-c:v",
                "libx265",
                "-preset",
                "fast",
                "-b:v",
                str(args.bitrate),
                "-minrate",
                str(args.bitrate),
                "-maxrate",
                str(args.bitrate),
                "-bufsize",
                str(args.bitrate),
                "-pix_fmt",
                "yuv444p",
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
                "-movflags",
                "+faststart",
                str(target),
            ]
            run(command, folder / f"{name}-encode.log")
            commands.append(command)
            stream = probe(target)
            if (stream["width"], stream["height"], stream["pix_fmt"]) != (
                width,
                height,
                "yuv444p",
            ):
                raise ValueError("Encoded geometry or sampling changed")
            if (
                int(stream["nb_read_frames"]) != args.frames
                or Fraction(stream["avg_frame_rate"]) != 60
            ):
                raise ValueError("Encoded count or FPS changed")
            if (
                stream.get("color_range") != "pc"
                or stream.get("color_space") != "bt709"
            ):
                raise ValueError("Encoded color metadata changed")
            stats = packet_stats(target, Fraction(60), args.skip)
            files[name] = dict(
                path=str(target.relative_to(args.output)),
                sha256=sha256(target),
                stream=stream,
                rate=stats,
            )
            print(
                f"ENCODED {scene['id']} {name}: {stats['video_bps'] / 1e6:.4f} Mbps",
                flush=True,
            )
        hr = probe(reference)
        if int(hr["nb_read_frames"]) != args.frames:
            raise ValueError("Reference count changed")
        manifest["scenes"].append(
            dict(
                scene,
                source=str(source),
                source_sha256=sha256(source),
                excerpt_start_frame=start,
                frames=args.frames,
                fps="60",
                files=files,
                commands=commands,
            )
        )
        atomic_json(args.output / "manifest.json", manifest)
    atomic_json(
        args.output / "PREPARED.json",
        dict(manifest_sha256=sha256(args.output / "manifest.json")),
    )


def decoded(root, scene):
    """Check every frame's PTS, count and dimensions before handing it to inference."""
    containers = [
        av.open(str(root / scene["files"][role]["path"]))
        for role in ("reference", "half", "fhd")
    ]
    count = 0
    try:
        for i, frames in enumerate(
            zip_longest(*(c.decode(video=0) for c in containers))
        ):
            for frame, size in zip(frames, ((1920, 1080), (960, 540), (1920, 1080))):
                if (
                    frame is None
                    or (frame.width, frame.height) != size
                    or frame.pts is None
                ):
                    raise ValueError("Unequal or malformed decoded frames")
                if abs(float(frame.pts * frame.time_base) - i / 60) > max(
                    0.0011, float(frame.time_base) * 1.1
                ):
                    raise ValueError(f"Frame timestamp mismatch at {i}")
            count += 1
            yield i, tuple(f.to_ndarray(format="rgb24") for f in frames)
        if count != scene["frames"]:
            raise ValueError("Truncated decoded inputs")
    finally:
        for container in containers:
            container.close()


def reencode(args):
    """Reuse the exact lossless excerpts when exploring another bandwidth cap."""
    parent = json.loads((args.prepared / "manifest.json").read_text())
    receipt = json.loads((args.prepared / "PREPARED.json").read_text())
    if receipt["manifest_sha256"] != sha256(args.prepared / "manifest.json"):
        raise ValueError("Changed parent input manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    data = dict(
        parent,
        bitrate_bps=args.bitrate,
        scenes=[],
        command=sys.argv,
        generator_sha256=sha256(Path(__file__)),
        parent_manifest=str((args.prepared / "manifest.json").resolve()),
        parent_manifest_sha256=sha256(args.prepared / "manifest.json"),
    )
    for scene in parent["scenes"]:
        copied = json.loads(json.dumps(scene))
        folder = args.output / scene["id"]
        folder.mkdir()
        source = args.prepared / scene["files"]["reference"]["path"]
        if sha256(source) != scene["files"]["reference"]["sha256"]:
            raise ValueError("Lossless excerpt changed")
        reference = args.output / scene["files"]["reference"]["path"]
        shutil.copyfile(source, reference)
        copied["reference_origin"] = dict(
            parent_manifest_sha256=data["parent_manifest_sha256"],
            path=str(source.resolve()),
            sha256=sha256(reference),
            original_commands=scene["commands"],
        )
        copied["commands"] = [["copy", str(source), str(reference)]]
        for index, role in enumerate(("fhd", "half"), 1):
            command = list(scene["commands"][index])
            command[command.index("-i") + 1] = str(reference)
            target = args.output / scene["files"][role]["path"]
            command[-1] = str(target)
            for flag in ("-b:v", "-minrate", "-maxrate", "-bufsize"):
                command[command.index(flag) + 1] = str(args.bitrate)
            i = command.index("-x265-params") + 1
            command[i] = re.sub(
                r"(?<![\w-])(bitrate|vbv-maxrate|vbv-bufsize)=\d+",
                lambda m: m[1] + "=" + str(args.bitrate // 1000),
                command[i],
            )
            run(command, folder / f"{role}-encode.log")
            stream = probe(target)
            expected = (1920, 1080) if role == "fhd" else (960, 540)
            if (
                (stream["width"], stream["height"]) != expected
                or stream["pix_fmt"] != "yuv444p"
                or int(stream["nb_read_frames"]) != data["frames"]
                or Fraction(stream["avg_frame_rate"]) != 60
                or stream.get("color_range") != "pc"
                or stream.get("color_space") != "bt709"
            ):
                raise ValueError("Re-encoded geometry, count, FPS or color changed")
            stats = packet_stats(target, Fraction(60), data["skip"])
            copied["files"][role] = dict(
                path=str(target.relative_to(args.output)),
                sha256=sha256(target),
                stream=stream,
                rate=stats,
            )
            copied["commands"].append(command)
            print(
                f"ENCODED {args.bitrate} {scene['id']} {role}: scored {stats['retained_video_bps'] / 1000:.1f} kbps",
                flush=True,
            )
        data["scenes"].append(copied)
        atomic_json(args.output / "manifest.json", data)
    atomic_json(
        args.output / "PREPARED.json",
        dict(manifest_sha256=sha256(args.output / "manifest.json")),
    )


def evaluate(args):
    root = args.prepared
    data = json.loads((root / "manifest.json").read_text())
    receipt = json.loads((root / "PREPARED.json").read_text())
    if receipt["manifest_sha256"] != sha256(root / "manifest.json"):
        raise ValueError("Changed prepared manifest")
    for scene in data["scenes"]:
        for item in scene["files"].values():
            if sha256(root / item["path"]) != item["sha256"]:
                raise ValueError("Changed prepared video")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "runner.py").write_bytes(Path(__file__).read_bytes())
    network, lpips_identity = perceptual_network(args.device)
    report = dict(
        schema="sr_bitrate_poc_results_v1",
        command=sys.argv,
        manifest_sha256=sha256(root / "manifest.json"),
        input_root=str(root.resolve()),
        generator_sha256=sha256(Path(__file__)),
        python=sys.version,
        torch=torch.__version__,
        platform=platform.platform(),
        device=args.device,
        precision="fp32",
        lpips_identity=lpips_identity,
        models={},
        records=[],
        videos=[],
        complete=False,
        started=time.time(),
        downsample="Existing TorchUpscaler RGB8 output at 3840x2160 -> OpenCV INTER_AREA to 1920x1080",
        scope="Three selected excerpts; shared RGB PSNR/SSIM and AlexNet-v0.1 LPIPS at native 1080p. Offline quality PoC, no live-FPS or text-correctness claim.",
    )
    report["rates"] = {
        scene["id"]: {
            role: packet_stats(
                root / scene["files"][role]["path"], Fraction(60), data["skip"]
            )
            for role in ("fhd", "half")
        }
        for scene in data["scenes"]
    }
    from benchmarks.super_resolution import (
        gallery as gallery_module,
        metrics as metrics_module,
        video_benchmark,
    )

    report["sources"] = {
        str(Path(module.__file__)): sha256(Path(module.__file__))
        for module in (candidates, gallery_module, metrics_module, video_benchmark)
    }
    selected = set(data["sample_indices"])

    def quality(reference, output):
        score = metrics(reference, output)
        with torch.inference_mode():
            score["lpips"] = float(
                network(
                    normalized_tensor(reference, args.device),
                    normalized_tensor(output, args.device),
                ).item()
            )
        return score

    def save(scene, frame, method, output, reference, timing=None):
        folder = args.output / scene["id"] / f"frame-{frame:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{method}.png"
        Image.fromarray(output).save(target)
        report["records"].append(
            dict(
                scene=scene["id"],
                capture=scene["capture"],
                frame=frame,
                source_frame=scene["excerpt_start_frame"] + frame,
                method=method,
                output=str(target.relative_to(args.output)),
                sha256=sha256(target),
                quality=quality(reference, output),
                timing=timing,
            )
        )

    for scene in data["scenes"]:
        for i, (reference, half, fhd) in decoded(root, scene):
            if i not in selected:
                continue
            folder = args.output / scene["id"] / f"frame-{i:04d}"
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(reference).save(folder / "reference.png")
            Image.fromarray(half).save(folder / "input-540p.png")
            save(scene, i, "fhd-direct", fhd, reference)
            save(
                scene,
                i,
                "half-bicubic",
                cv2.resize(half, (1920, 1080), interpolation=cv2.INTER_CUBIC),
                reference,
            )
        print(f"BASELINES {scene['id']}", flush=True)
    atomic_json(args.output / "results.json", report)
    for key in args.models:
        adapter = candidates.TorchUpscaler(key, device=args.device, precision="fp32")
        report["models"][key] = adapter.provenance
        for scene in data["scenes"]:
            folder = args.output / scene["id"]
            sinks = {}
            if key == args.video_model:
                sinks = {
                    role: LosslessWriter(
                        folder / f"{role}-{key}-lossless.mkv", 1920, 1080, Fraction(60)
                    )
                    for role in ("half", "fhd")
                }
            try:
                for i, (reference, half, fhd) in decoded(root, scene):
                    if i < data["skip"] or (i not in selected and not sinks):
                        continue
                    for role, rgb in (("half", half), ("fhd", fhd)):
                        output, timing = adapter(rgb)
                        if role == "fhd":
                            if output.shape != (2160, 3840, 3):
                                raise ValueError(
                                    "FHD input did not produce native 4K output"
                                )
                            started = time.perf_counter()
                            output = cv2.resize(
                                output, (1920, 1080), interpolation=cv2.INTER_AREA
                            )
                            timing["downsample_ms"] = (
                                time.perf_counter() - started
                            ) * 1000
                        if output.shape != reference.shape:
                            raise ValueError(
                                "Final display output must be native 1080p"
                            )
                        if i in selected:
                            save(scene, i, f"{role}-{key}", output, reference, timing)
                        if sinks:
                            sinks[role].write(output)
                    if sinks and (i + 1) % 60 == 0:
                        print(
                            f"VIDEO {key} {scene['id']} {i + 1}/{scene['frames']}",
                            flush=True,
                        )
            finally:
                for sink in sinks.values():
                    sink.close()
            for role, sink in sinks.items():
                target = folder / f"{role}-{key}-lossless.mkv"
                verify_lossless(
                    target, data["frames"] - data["skip"], sink.digest.hexdigest()
                )
                report["videos"].append(
                    dict(
                        scene=scene["id"],
                        method=f"{role}-{key}",
                        path=str(target.relative_to(args.output)),
                        frames=sink.count,
                        rgb_sequence_sha256=sink.digest.hexdigest(),
                        sha256=sha256(target),
                        verified=True,
                    )
                )
            atomic_json(args.output / "results.json", report)
            print(f"SCORED {key} {scene['id']}", flush=True)
        del adapter
        gc.collect()
        if args.device == "mps":
            torch.mps.empty_cache()
    groups = defaultdict(list)
    for row in report["records"]:
        groups[(row["method"], row["scene"])].append(row["quality"])
    report["per_scene"] = [
        dict(
            method=method,
            scene=scene,
            samples=len(values),
            **{
                metric: float(np.mean([v[metric] for v in values]))
                for metric in ("psnr", "ssim", "lpips")
            },
        )
        for (method, scene), values in groups.items()
    ]
    report["summary"] = [
        dict(
            method=method,
            **{
                metric: float(
                    np.mean(
                        [
                            r[metric]
                            for r in report["per_scene"]
                            if r["method"] == method
                        ]
                    )
                )
                for metric in ("psnr", "ssim", "lpips")
            },
        )
        for method in sorted({r["method"] for r in report["records"]})
    ]
    report.update(complete=True, finished=time.time())
    atomic_json(args.output / "results.json", report)
    atomic_json(
        args.output / "COMPLETED.json",
        dict(
            results_sha256=sha256(args.output / "results.json"),
            records=len(report["records"]),
            models=list(report["models"]),
            videos=len(report["videos"]),
        ),
    )


def gallery(args):
    root = args.run
    report = json.loads((root / "results.json").read_text())
    data = json.loads((Path(report["input_root"]) / "manifest.json").read_text())
    receipt = json.loads((root / "COMPLETED.json").read_text())
    if not report["complete"] or receipt["results_sha256"] != sha256(
        root / "results.json"
    ):
        raise ValueError("Inference has not completed")
    for row in report["records"]:
        if sha256(root / row["output"]) != row["sha256"]:
            raise ValueError("Reconstruction changed after scoring")
    validity = {}
    for scene in data["scenes"]:
        validity[scene["id"]] = {}
        for role in ("fhd", "half"):
            stats = packet_stats(
                Path(report["input_root"]) / scene["files"][role]["path"],
                Fraction(60),
                data["skip"],
            )
            log = Path(report["input_root"]) / scene["id"] / f"{role}-encode.log"
            underflows = log.read_text().count("VBV underflow")
            validity[scene["id"]][role] = dict(
                **stats,
                target_bps=data["bitrate_bps"],
                average_within_5pct=abs(
                    stats["retained_video_bps"] / data["bitrate_bps"] - 1
                )
                <= 0.05,
                vbv_underflows=underflows,
                encoder_log_sha256=sha256(log),
            )
    atomic_json(root / "rate-validity.json", validity)
    methods = [
        "reference",
        "fhd-direct",
        *(f"fhd-{key}" for key in report["models"]),
        "reference",
        "half-bicubic",
        *(f"half-{key}" for key in report["models"]),
    ]
    labels = [
        "Original",
        "1080p direct",
        *(f"1080p {key} x2 -> x0.5" for key in report["models"]),
        "Original",
        "540p bicubic",
        *(f"540p {key} x2" for key in report["models"]),
    ]
    rate = f"{data['bitrate_bps'] / 1000:g} Kbps"
    lines = [
        f"# {rate} HEVC screen-content PoC",
        "",
        f"60 fps; yuv444p, full-range BT.709; x265 fast, {rate} ABR + equal VBV maxrate/bufsize + strict-cbr/HRD. "
        "4-second contiguous encodes; first second omitted from scoring and reconstruction videos. "
        "Seven paired quality frames per scene; means average frames within scenes, then scenes equally. "
        "Only official pretrained native x2 weights; no fine-tuning. FHD SR is native 4K RGB8 followed by area downsampling to 1080p.",
        "",
        "| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |",
        "|---|---:|---:|---:|",
    ]
    for row in report["summary"]:
        lines.append(
            f"| {row['method']} | {row['psnr']:.3f} | {row['ssim']:.5f} | {row['lpips']:.5f} |"
        )
    lines.extend(
        [
            "",
            "## Actual video packet bitrate",
            "",
            "| Scene | 1080p whole / scored Mbps | 540p whole / scored Mbps |",
            "|---|---:|---:|",
        ]
    )
    for scene in data["scenes"]:
        rates = [
            packet_stats(
                Path(report["input_root"]) / scene["files"][role]["path"],
                Fraction(60),
                data["skip"],
            )
            for role in ("fhd", "half")
        ]
        lines.append(
            f"| {scene['id']} | "
            + " | ".join(
                f"{r['video_bps'] / 1e6:.4f} / {r['retained_video_bps'] / 1e6:.4f}"
                for r in rates
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Short, selected development excerpts only. Encoder startup, content and preset matter. "
            "CBR is a VBV/HRD rate-control constraint, not equal packet sizes; packet rates above exclude MP4 overhead. "
            "Quality is sampled; continuous reconstructions, when present, process all retained frames for the selected video model. "
            "Viewing MP4s are lossy and never used for metrics. No OCR or live-throughput claim.",
            "",
            "Reproduce with the commands and hashes in the input manifest and results.json.",
        ]
    )
    invalid = [
        (scene, role, r)
        for scene, streams in validity.items()
        for role, r in streams.items()
        if not r["average_within_5pct"] or r["vbv_underflows"]
    ]
    if invalid:
        lines.extend(
            [
                "",
                "## Rate-control failures",
                "",
                "This is a target-bitrate run with rate-control violations. Do not interpret its quality table as a valid equal-budget comparison.",
                "",
                "| Scene | Stream | Scored Kbps | VBV underflows |",
                "|---|---|---:|---:|",
            ]
        )
        for scene, role, r in invalid:
            lines.append(
                f"| {scene} | {role} | {r['retained_video_bps'] / 1000:.2f} | {r['vbv_underflows']} |"
            )
    lines.extend(
        [
            "",
            "Rate-control reference: [x265 strict CBR and VBV documentation](https://x265.readthedocs.io/en/master/cli.html#cmdoption-strict-cbr).",
        ]
    )
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    page = [
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
        f"<title>{rate} HEVC PoC</title><style>body{{font-family:system-ui;margin:28px;background:#f8fafc;color:#0f172a}} "
        "img{max-width:100%;height:auto;border:1px solid #94a3b8}video{max-width:100%}table{border-collapse:collapse} "
        "td,th{padding:8px 15px;border:1px solid #cbd5e1}a{color:#0369a1}section{margin:36px 0}</style><body>",
        f"<h1>{rate} HEVC · FHD versus 540p</h1><p>60 fps · 4:4:4 · full BT.709 · official pretrained weights.</p>",
        "<p>Top: 1080p direct / 1080p → model ×2 → area downsample. Bottom: 540p → bicubic / model ×2. "
        "All final images are 1920×1080. Detail crops use nearest-neighbor ×2 for inspection.</p>",
        '<p><a href="REPORT.md">Protocol and results</a> · <a href="results.json">Raw metrics and provenance</a></p>',
        "<table><tr><th>Method</th><th>PSNR ↑</th><th>SSIM ↑</th><th>LPIPS ↓</th></tr>",
    ]
    if invalid:
        page.insert(
            -1,
            '<p style="padding:16px;background:#fef3c7"><strong>Rate-control violations detected.</strong> '
            "Configured bitrate was not achieved reliably. This quality table is not a valid equal-budget comparison. "
            '<a href="rate-validity.json">Actual rates and VBV underflows</a></p>',
        )
    for row in report["summary"]:
        page.append(
            f"<tr><td>{row['method']}</td><td>{row['psnr']:.3f}</td><td>{row['ssim']:.5f}</td><td>{row['lpips']:.5f}</td></tr>"
        )
    page.append(
        "</table><p>Three selected excerpts, seven quality samples each. "
        "The videos below are viewing copies; metrics use lossless native 1080p pixels.</p>"
    )
    for scene in data["scenes"]:
        poster = min(data["sample_indices"], key=lambda i: abs(i - data["frames"] // 2))
        folder = root / scene["id"] / f"frame-{poster:04d}"
        images = [
            Image.open(folder / f"{method}.png").convert("RGB") for method in methods
        ]
        page.append(
            f"<section><h2>{html.escape(scene['title'])}</h2><p>Source {scene['capture']}, frame {scene['excerpt_start_frame'] + poster}.</p>"
        )
        for crop in scene["crops"]:
            target = root / scene["id"] / f"comparison-{crop['id']}.png"
            crop_strip(
                images, crop["xywh"], labels=labels, columns=len(methods) // 2
            ).save(target)
            relative = str(target.relative_to(root))
            page.append(
                f'<h3>{html.escape(crop["title"])}</h3><a href="{relative}"><img src="{relative}" loading="lazy"></a>'
            )
        page.append("<details><summary>Native 1080p images</summary><ul>")
        for method in methods[: len(methods) // 2] + methods[len(methods) // 2 + 1 :]:
            relative = str((folder / f"{method}.png").relative_to(root))
            page.append(f'<li><a href="{relative}">{html.escape(method)}</a></li>')
        page.append("</ul></details>")
        if args.video_model in report["models"]:
            # Six native-coordinate detail panels preserve text size in the viewing copy.
            crop = scene["crops"][0]["xywh"]
            x, y, width, height = crop
            inputs = [
                str(Path(report["input_root"]) / scene["files"][role]["path"])
                for role in ("reference", "fhd", "half")
            ]
            inputs += [
                str(root / scene["id"] / f"{role}-{args.video_model}-lossless.mkv")
                for role in ("fhd", "half")
            ]
            titles = [
                "Original",
                "1080p direct",
                "1080p SR then downsample",
                "Original",
                "540p bicubic",
                "540p SR x2",
            ]
            target = root / scene["id"] / "quicksrnet-detail.mp4"
            if not target.exists():
                from itertools import islice

                containers = [av.open(path) for path in inputs]
                streams = [
                    islice(c.decode(video=0), data["skip"] if i < 3 else 0, None)
                    for i, c in enumerate(containers)
                ]
                lossless = target.with_suffix(".mkv")
                sink = LosslessWriter(
                    lossless, width * 6 + 12, height * 4 + 86, Fraction(60)
                )
                try:
                    for frames in zip_longest(*streams):
                        if any(f is None for f in frames):
                            raise ValueError("Viewing inputs have unequal frame counts")
                        ref, direct, half, fhd_sr, half_sr = [
                            f.to_ndarray(format="rgb24") for f in frames
                        ]
                        panels = [
                            ref,
                            direct,
                            fhd_sr,
                            ref,
                            cv2.resize(
                                half, (1920, 1080), interpolation=cv2.INTER_CUBIC
                            ),
                            half_sr,
                        ]
                        image = crop_strip(
                            [Image.fromarray(p) for p in panels],
                            crop,
                            labels=titles,
                            columns=3,
                        )
                        sink.write(np.array(image))
                finally:
                    sink.close()
                    for container in containers:
                        container.close()
                verify_lossless(
                    lossless, data["frames"] - data["skip"], sink.digest.hexdigest()
                )
                command = [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-nostdin",
                    "-i",
                    str(lossless),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-crf",
                    "16",
                    "-preset",
                    "fast",
                    "-threads",
                    "4",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(target),
                ]
                run(command, root / scene["id"] / "viewing-encode.log")
            stream = probe(target)
            if int(stream["nb_read_frames"]) != data["frames"] - data["skip"]:
                raise ValueError("Viewing comparison frame count changed")
            page.append(
                f'<h3>QuickSRNet · 3-second continuous detail comparison</h3><video controls preload="metadata" src="{scene["id"]}/quicksrnet-detail.mp4"></video>'
            )
        page.append("</section>")
    page.append("</body></html>")
    (root / "index.html").write_text("\n".join(page) + "\n")
    print(f"GALLERY {root / 'index.html'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scenes", nargs="+", default=list(SCENES))
    p.add_argument("--bitrate", type=int, default=1_000_000)
    p.add_argument("--frames", type=int, default=240)
    p.add_argument("--skip", type=int, default=60)
    p = sub.add_parser("reencode")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--bitrate", type=int, required=True)
    p = sub.add_parser("evaluate")
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    p.add_argument("--video-model", choices=(*MODELS, "none"), default="quicksrnet")
    p.add_argument("--device", default="mps")
    p = sub.add_parser("gallery")
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--video-model", choices=(*MODELS, "none"), default="quicksrnet")
    args = parser.parse_args()
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.action == "prepare" and (
        args.bitrate <= 0 or args.bitrate % 1000 or not 0 <= args.skip < args.frames
    ):
        parser.error("Use a positive multiple-of-1000 bitrate and 0 <= skip < frames")
    if args.action == "reencode" and (args.bitrate <= 0 or args.bitrate % 1000):
        parser.error("Use a positive multiple-of-1000 bitrate")
    dict(prepare=prepare, reencode=reencode, evaluate=evaluate, gallery=gallery)[
        args.action
    ](args)


if __name__ == "__main__":
    main()
