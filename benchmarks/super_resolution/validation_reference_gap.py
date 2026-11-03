"""Matched source, no-codec resize, compressed and SR reference-gap diagnostics."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from benchmarks.super_resolution.legacy_validation_compare import adapter
from benchmarks.super_resolution.training_state import atomic_json
from benchmarks.super_resolution.validation_gpu_quality import quality
from isharescreen.enhancement.models import sha256


def read(path):
    return json.loads(Path(path).read_text())


def score(a, b):
    mse = float(np.mean((a.astype(np.float64) - b) ** 2))
    return dict(psnr=10 * math.log10(65025 / mse) if mse else None, mse=mse)


def clean_versions(root, case):
    folder = root / "frames" / case["clip"] / str(case["frame"])
    folder.mkdir(parents=True, exist_ok=True)
    hr = np.asarray(Image.open(case["hr"]).convert("RGB"))
    base = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1920x1080",
        "-i",
        "pipe:0",
        "-filter_threads",
        "1",
        "-vf",
        "scale=1280:720:flags=lanczos:in_range=full:out_range=full:out_color_matrix=bt709",
        "-frames:v",
        "1",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
    ]
    rgb = subprocess.check_output(base + ["rgb24", "pipe:1"], input=hr.tobytes())
    rgb = np.frombuffer(rgb, np.uint8).reshape(720, 1280, 3)
    yuv = subprocess.check_output(base + ["yuv420p", "pipe:1"], input=hr.tobytes())
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "yuv420p",
        "-s",
        "1280x720",
        "-color_range",
        "pc",
        "-colorspace",
        "bt709",
        "-i",
        "pipe:0",
        "-filter_threads",
        "1",
        "-vf",
        "scale=in_range=full:out_range=full:in_color_matrix=bt709",
        "-frames:v",
        "1",
        "-threads",
        "1",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    clean420 = np.frombuffer(
        subprocess.check_output(command, input=yuv), np.uint8
    ).reshape(720, 1280, 3)
    for label, image in [("clean_rgb720", rgb), ("clean_420720", clean420)]:
        Image.fromarray(image).save(folder / (label + ".png"))
    return dict(case, folder=str(folder), clean_rgb720=rgb, clean_420720=clean420)


def figure(root, case, images, records):
    # Pick the most detailed fixed-size reference region, without looking at SR gains.
    hr = images["original"]
    gray = cv2.cvtColor(hr, cv2.COLOR_RGB2GRAY)
    edge = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    choices = [
        (float(edge[y : y + 128, x : x + 384].mean()), x, y)
        for y in range(128, 900, 96)
        for x in range(128, 1500, 192)
    ]
    _, x, y = max(choices)
    box = (x, y, x + 384, y + 128)
    labels = [
        ("original", "Original1080p reference"),
        ("clean_420", "No HEVC:720p420 + bicubic"),
        ("compressed", "600 Kbps420 + bicubic"),
        ("official", "600 Kbps + official EDSR"),
        ("trained", "600 Kbps + trained EDSR"),
    ]
    canvas = Image.new("RGB", (1050, len(labels) * 300 + 55), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    draw.text(
        (12, 10),
        f"{case['content']} / {case['clip']} / frame{case['frame']}; same crop,2x nearest display",
        fill="black",
        font=font,
    )
    for i, (key, label) in enumerate(labels):
        crop = Image.fromarray(images[key]).crop(box)
        crop.save(root / f"{case['content']}-{key}-crop.png")
        metrics = score(np.asarray(Image.fromarray(hr).crop(box)), np.asarray(crop))
        text = f"{label}\nFull PSNR: {records.get(key, {}).get('psnr', 'infinite')}\nCrop PSNR: {metrics['psnr']}"
        yy = 55 + i * 300
        draw.text((12, yy + 12), text, fill="black", font=font)
        canvas.paste(crop.resize((768, 256), Image.Resampling.NEAREST), (275, yy))
    canvas.save(root / f"{case['content']}-comparison.png")
    return dict(
        clip=case["clip"],
        frame=case["frame"],
        content=case["content"],
        crop=box,
        selection="Middle sampled frame of the clip with median trained-vs-compressed PSNR gain within content class; ROI maximizes reference-only edge energy, not SR gain.",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    root = a.root.resolve()
    base = a.baseline.resolve()
    root.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    inputs = read(base / "INPUTS.json")
    cases = []
    for run in inputs["runs"]:
        if run["config"] != "720p-600k-420":
            continue
        for sample in run["samples"]:
            for role in ("lr", "hr"):
                assert sha256(base / sample[role]) == sample[role + "_sha256"]
            cases.append(
                dict(
                    clip=run["clip"],
                    frame=sample["frame"],
                    content=run["content"],
                    lr=str(base / sample["lr"]),
                    hr=str(base / sample["hr"]),
                )
            )
    assert len(cases) == 72
    checkpoint = root / "trained-best-snapshot.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = adapter("edsr", a.device)
    protocol = dict(
        frames=72,
        clips=24,
        config="720p/600Kbps/420",
        display="1920x1080",
        source="Frozen processed RGB8 Mac recordings, not original MOV acquisition",
        aggregation="Mean frame PSNR within clip, then equal mean over24 clips; same macro rule as training validation.",
        checkpoint_epoch=payload["epoch"],
        checkpoint_sha256=sha256(checkpoint),
        input_manifest_sha256=sha256(base / "INPUTS.json"),
        ffmpeg=subprocess.check_output(["ffmpeg", "-version"], text=True).splitlines()[
            0
        ],
        script_sha256=sha256(Path(__file__)),
        limits="No-codec resizes diagnose degradation and are not a theoretical SR ceiling. Current FFmpeg conversion may differ slightly from historic encode-host FFmpeg. Static scores do not establish visual discrimination or temporal quality.",
    )
    atomic_json(root / "PROTOCOL.json", protocol)
    results = []
    stored = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, case in enumerate(pool.map(lambda c: clean_versions(root, c), cases)):
            hr = np.asarray(Image.open(case["hr"]).convert("RGB"))
            lr = np.asarray(Image.open(case["lr"]).convert("RGB"))
            images = {
                "original": hr,
                "clean_rgb": cv2.resize(
                    case["clean_rgb720"], (1920, 1080), interpolation=cv2.INTER_CUBIC
                ),
                "clean_420": cv2.resize(
                    case["clean_420720"], (1920, 1080), interpolation=cv2.INTER_CUBIC
                ),
                "compressed": cv2.resize(
                    lr, (1920, 1080), interpolation=cv2.INTER_CUBIC
                ),
            }
            rows = []
            for key in ("clean_rgb", "clean_420", "compressed"):
                q = quality(hr, images[key])
                q["mse"] = score(hr, images[key])["mse"]
                rows.append(dict(method=key, **q))
            q = score(case["clean_420720"], lr)
            rows.append(dict(method="hevc_vs_clean_720", **q))
            stored[(case["clip"], case["frame"])] = (case, images, rows)
            results.extend(
                dict(
                    clip=case["clip"], frame=case["frame"], content=case["content"], **r
                )
                for r in rows
            )
            if i % 12 == 0:
                print("CLEAN", i + 1, "/72", flush=True)
    for key in ("official", "trained"):
        if key == "trained":
            torch.nn.Module.load_state_dict(
                model.model, payload["state_dict"], strict=True
            )
        for (clip, frame), (case, images, rows) in stored.items():
            lr = np.asarray(Image.open(case["lr"]).convert("RGB"))
            native, _ = model(lr)
            output = cv2.resize(native, (1920, 1080), interpolation=cv2.INTER_AREA)
            images[key] = output
            path = Path(case["folder"]) / (key + ".png")
            Image.fromarray(output).save(path)
            q = quality(images["original"], output)
            q["mse"] = score(images["original"], output)["mse"]
            row = dict(method=key, **q)
            rows.append(row)
            results.append(dict(clip=clip, frame=frame, content=case["content"], **row))
        print("SCORED", key, flush=True)
    methods = [
        "clean_rgb",
        "clean_420",
        "compressed",
        "official",
        "trained",
        "hevc_vs_clean_720",
    ]
    summary = {}
    for method in methods:
        selected = [r for r in results if r["method"] == method]
        by_clip = {}
        for row in selected:
            by_clip.setdefault(row["clip"], []).append(row)
        metrics = ["psnr", "mse"] + (["ssim"] if "ssim" in selected[0] else [])
        summary[method] = {
            metric: float(
                np.mean(
                    [np.mean([r[metric] for r in group]) for group in by_clip.values()]
                )
            )
            for metric in metrics
        }
    chosen = []
    for content in ("code", "web"):
        clips = {c["clip"] for c in cases if c["content"] == content}
        if not clips:
            continue
        ranked = []
        for clip in clips:
            gain = np.mean(
                [
                    next(r["psnr"] for r in rows if r["method"] == "trained")
                    - next(r["psnr"] for r in rows if r["method"] == "compressed")
                    for (c, _), (case, im, rows) in stored.items()
                    if c == clip
                ]
            )
            ranked.append((gain, clip))
        clip = sorted(ranked)[len(ranked) // 2][1]
        entry = sorted(
            [(frame, entry) for (c, frame), entry in stored.items() if c == clip]
        )[1][1]
        case, images, rows = entry
        chosen.append(figure(root, case, images, {r["method"]: r for r in rows}))
    atomic_json(
        root / "RESULTS.json",
        dict(protocol=protocol, summary=summary, rows=results, figures=chosen),
    )
    lines = [
        "# Reference-gap diagnostic",
        "",
        f"Trained EDSR checkpoint: epoch{payload['epoch']}.72 matched frames /24 Mac clips.",
        "",
        "| Method | RGB PSNR (dB) | SSIM |",
        "| --- | ---: | ---: |",
    ]
    for method, metrics in summary.items():
        lines.append(
            f"| {method} | {metrics['psnr']:.4f} | {metrics.get('ssim', 'n/a')} |"
        )
    lines += [
        "",
        "The uncompressed720p paths are diagnostic reference points, not SR ceilings. Original vs itself has zero MSE and infinite PSNR. No human visibility threshold is inferred from these scores.",
    ]
    (root / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
