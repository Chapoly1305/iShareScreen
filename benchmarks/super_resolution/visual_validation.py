"""Frozen validation-only base/finetune scene and local-region visual audit."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from benchmarks.super_resolution.legacy_validation_compare import adapter, display
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import sha256

MODELS = ("edsr", "imdn", "safmn", "quicksrnet", "efrlfn")


def read(path):
    return json.loads(Path(path).read_text())


def psnr(reference, image):
    mse = np.mean((reference.astype(np.float64) - image) ** 2)
    return 10 * math.log10(65025 / mse) if mse else None


def tiles(reference):
    """Fixed grid with reference-only saliency; excludes mostly flat tiles."""
    gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    edge = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    result = []
    for y in (120, 280, 440, 600, 760, 900):
        for x in (32, 352, 672, 992, 1312, 1568):
            roi = edge[y : y + 160, x : x + 320]
            energy = float(roi.mean())
            if energy >= 2 and float((roi > 10).mean()) >= 0.025:
                result.append(
                    dict(box=[x, y, x + 320, y + 160], reference_edge_energy=energy)
                )
    return result


def card(root, model, clip, frame, box, name, title):
    folder = root / "renders" / model / clip
    paths = [
        root / "references" / clip / "reference.png",
        folder / "base.png",
        folder / "trained.png",
    ]
    images = [Image.open(p).convert("RGB").crop(box) for p in paths]
    row_height = max(images[0].height * 2 + 24, 280)
    canvas = Image.new("RGB", (1320, 3 * row_height + 70), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    draw.text((15, 10), title, fill="black", font=font)
    draw.text(
        (15, 37),
        f"{model} | {clip} | frame{frame} | crop {box} | exact same pixels; left1x / right2x nearest",
        fill="black",
        font=small,
    )
    for i, (label, image) in enumerate(
        zip(["Reference", "Official base", "Finetuned"], images)
    ):
        y = 70 + i * row_height
        draw.text((15, y + 8), label, fill="black", font=font)
        value = psnr(np.asarray(images[0]), np.asarray(image))
        draw.text(
            (15, y + 40),
            f"Crop PSNR: {value:.2f}dB" if value is not None else "Crop PSNR: infinity",
            fill="black",
            font=small,
        )
        canvas.paste(image, (250, y))
        canvas.paste(
            image.resize((image.width * 2, image.height * 2), Image.Resampling.NEAREST),
            (660, y),
        )
    path = root / "cards" / f"{name}.png"
    path.parent.mkdir(exist_ok=True)
    canvas.save(path)
    return str(path.relative_to(root))


def analyze(root, campaign, base):
    records = []
    frames = []
    summaries = []
    contents = {
        r["clip"]: r["content"]
        for r in read(base / "INPUTS.json")["runs"]
        if r["config"] == "720p-600k-420"
    }
    for model in MODELS:
        folder = campaign / "models" / model
        done = read(folder / "COMPLETED.json")
        prior = read(folder / "validation-000.json")
        trained = read(folder / f"validation-{done['best_epoch']:03d}.json")
        lookup = {(r["clip"], r["frame"]): r for r in prior["mac_frames"]}
        for row in trained["mac_frames"]:
            old = lookup[row["clip"], row["frame"]]
            entry = dict(
                model=model,
                clip=row["clip"],
                frame=row["frame"],
                content=contents[row["clip"]],
                base_psnr=old["psnr"],
                trained_psnr=row["psnr"],
                delta_psnr=row["psnr"] - old["psnr"],
                delta_ssim=row["ssim"] - old["ssim"],
                delta_lpips=row["lpips"] - old["lpips"],
            )
            frames.append(entry)
        for clip in sorted(contents):
            selected = [r for r in frames if r["model"] == model and r["clip"] == clip]
            entry = dict(
                model=model,
                clip=clip,
                content=contents[clip],
                frames=len(selected),
                **{
                    k: float(np.mean([r[k] for r in selected]))
                    for k in [
                        "base_psnr",
                        "trained_psnr",
                        "delta_psnr",
                        "delta_ssim",
                        "delta_lpips",
                    ]
                },
            )
            entry["numeric_bucket"] = (
                "increase"
                if entry["delta_psnr"] > 0.1
                else ("decrease" if entry["delta_psnr"] < -0.1 else "small_change")
            )
            records.append(entry)
        for content in sorted(set(contents.values())):
            selected = [
                r for r in records if r["model"] == model and r["content"] == content
            ]
            summaries.append(
                dict(
                    model=model,
                    content=content,
                    clips=len(selected),
                    **{
                        k: float(np.mean([r[k] for r in selected]))
                        for k in [
                            "base_psnr",
                            "trained_psnr",
                            "delta_psnr",
                            "delta_ssim",
                            "delta_lpips",
                        ]
                    },
                )
            )
    atomic_json(
        root / "SCENES.json",
        dict(
            clips=records,
            frames=frames,
            content_summary=summaries,
            bucket_limit="PSNR ±0.1dB is a numerical screening convention, not a human visibility/significance threshold. Local regions can regress despite positive full-frame metrics.",
        ),
    )
    for name, rows in [
        ("scene-metrics.csv", records),
        ("frame-metrics.csv", frames),
        ("content-summary.csv", summaries),
    ]:
        with (root / name).open("w") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def render(args):
    root = args.root.resolve()
    base = args.base.resolve()
    campaign = args.campaign.resolve()
    root.mkdir(exist_ok=True, parents=True)
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    analyze(root, campaign, base)
    cases = [
        dict(clip=r["clip"], content=r["content"], **r["samples"][1])
        for r in read(base / "INPUTS.json")["runs"]
        if r["config"] == "720p-600k-420"
    ]
    provenance = {}
    regions = []
    for model in MODELS:
        checkpoint = root / "weights" / f"{model}.pt"
        done = read(campaign / "models" / model / "COMPLETED.json")
        assert sha256(checkpoint) == done["best_sha256"]
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert payload["epoch"] == done["best_epoch"]
        wrapper = adapter(model, "cuda:0")
        provenance[model] = dict(
            base=wrapper.provenance,
            trained_sha256=sha256(checkpoint),
            epoch=payload["epoch"],
        )
        for phase in ["base", "trained"]:
            if phase == "trained":
                torch.nn.Module.load_state_dict(
                    wrapper.model, payload["state_dict"], strict=True
                )
            for i, case in enumerate(cases):
                lr = base / case["lr"]
                hr = base / case["hr"]
                assert (
                    sha256(lr) == case["lr_sha256"] and sha256(hr) == case["hr_sha256"]
                )
                folder = root / "renders" / model / case["clip"]
                folder.mkdir(parents=True, exist_ok=True)
                target = folder / (phase + ".png")
                if not target.exists():
                    image = np.asarray(Image.open(lr).convert("RGB"))
                    out = display(wrapper, image)
                    Image.fromarray(out).save(target)
                if phase == "base":
                    refroot = root / "references" / case["clip"]
                    refroot.mkdir(parents=True, exist_ok=True)
                    if not (refroot / "reference.png").exists():
                        Image.open(hr).save(refroot / "reference.png")
                        image = np.asarray(Image.open(lr).convert("RGB"))
                        Image.fromarray(
                            cv2.resize(
                                image, (1920, 1080), interpolation=cv2.INTER_CUBIC
                            )
                        ).save(refroot / "compressed-bicubic.png")
                if phase == "trained":
                    reference = np.asarray(Image.open(hr).convert("RGB"))
                    old = np.asarray(Image.open(folder / "base.png").convert("RGB"))
                    new = np.asarray(Image.open(target).convert("RGB"))
                    for index, tile in enumerate(tiles(reference)):
                        x, y, xx, yy = tile["box"]
                        ref = reference[y:yy, x:xx]
                        pb = psnr(ref, old[y:yy, x:xx])
                        pt = psnr(ref, new[y:yy, x:xx])
                        regions.append(
                            dict(
                                model=model,
                                clip=case["clip"],
                                frame=case["frame"],
                                content=case["content"],
                                region=index,
                                **tile,
                                base_psnr=pb,
                                trained_psnr=pt,
                                delta_psnr=pt - pb,
                            )
                        )
                if i % 8 == 7:
                    print("RENDERED", model, phase, i + 1, "/24", flush=True)
        del wrapper
        torch.cuda.empty_cache()
    atomic_json(root / "REGIONS.json", regions)
    atomic_json(
        root / "PROTOCOL.json",
        dict(
            models=provenance,
            validation_clips=24,
            scored_frames_per_model=72,
            rendered_frames_per_model=24,
            render_selection="Middle of three frozen timestamps per clip; render both versions FP32 on the same H100 GPU.",
            roi_policy="Fixed320x160 grid, reference-only Laplacian saliency, no ROI learned from outputs; rank differences only to screen examples for qualitative inspection.",
            human_test=False,
            scope="Qualitative assistant inspection and paired image metrics; no participant study, visibility significance, OCR correctness or temporal-flicker claims.",
            content_limit="Labels describe captured scene, not independent datasets. Some Mac clips show controlled synthetic desktop pages;24 recordings share a capture session.",
            gpu=torch.cuda.get_device_name(),
            host=platform.node(),
            torch=str(torch.__version__),
            script_sha256=sha256(Path(__file__)),
        ),
    )
    print("DONE", len(regions), "local region comparisons", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--campaign", type=Path, required=True)
    render(p.parse_args())
