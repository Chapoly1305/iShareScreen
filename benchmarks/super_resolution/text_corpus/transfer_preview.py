"""Actual model inference on the existing same-page HEVC decoded frames."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from benchmarks.super_resolution.text_corpus.train import initialize, state_identity
from benchmarks.super_resolution.training_state import atomic_json, fingerprint
from isharescreen.enhancement import candidates
from isharescreen.enhancement.models import project_home, sha256
from benchmarks.super_resolution.text_corpus.runtime_paths import font


def read(path):
    return json.loads(path.read_text())


def run(args):
    root = args.root.resolve()
    experiment = root / "official-transfer-v1"
    source = root / "screenshots-same-source-v1"
    out = experiment / ("previews" if args.results else "official-preview")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("Preview already completed")
    old = read(source / "manifest.json")
    roi = old["geometry"]["roi"]
    x, y, w, h = roi
    box = (x, y, x + w, y + h)
    reference = Image.open(source / "reference-fhd.png").convert("RGB")
    original = root / old["source_case"]["hr"]
    assert sha256(original) == old["source_hr_sha256"]
    assert np.array_equal(
        np.array(reference), np.array(Image.open(original).convert("RGB"))
    )
    torch.set_num_threads(4)
    models = {}
    identities = {}
    records = []
    models["Official pretrained"], spec = candidates.load_network(
        "quicksrnet", project_home()
    )
    identities["Official pretrained"] = dict(
        checkpoint_sha256=spec["files"]["weights.pth"]["sha256"],
        tensor_sha256=state_identity(models["Official pretrained"]),
        epoch=0,
    )
    if args.results:
        for name, label in [
            ("official-finetuned", "Official + text fine-tuning"),
            ("scratch-control", "Scratch, matched control"),
        ]:
            folder = args.results / "runs" / name
            policy = read(folder / "protocol.json")
            done = read(folder / "COMPLETED.json")
            assert (
                fingerprint(policy) == done["protocol_binding"]
                and sha256(folder / "best.pt") == done["best_sha256"]
            )
            model, _ = initialize(project_home(), 682, "scratch")
            payload = torch.load(
                folder / "best.pt", map_location="cpu", weights_only=True
            )
            assert payload["protocol_binding"] == done["protocol_binding"]
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval().requires_grad_(False)
            models[label] = model
            identities[label] = dict(
                checkpoint_sha256=sha256(folder / "best.pt"),
                epoch=payload["epoch"],
                tensor_sha256=state_identity(model),
                training_protocol_sha256=sha256(folder / "protocol.json"),
            )
    preview_font = font()
    font_large = ImageFont.truetype(preview_font, 18)
    small = ImageFont.truetype(preview_font, 14)
    for kbps in [600, 1000]:
        for second in [1, 3]:
            input_path = source / f"{kbps}k/{second}s-lr-native.png"
            rgb = np.array(Image.open(input_path).convert("RGB"))
            tensor = torch.from_numpy(rgb.transpose(2, 0, 1).copy())[None].float() / 255
            images = {}
            for label, model in models.items():
                with torch.inference_mode():
                    prediction = model(tensor)
                assert (
                    prediction.shape == (1, 3, 1080, 1920)
                    and torch.isfinite(prediction).all()
                )
                image = Image.fromarray(
                    prediction[0]
                    .mul(255)
                    .round()
                    .clamp(0, 255)
                    .byte()
                    .permute(1, 2, 0)
                    .numpy()
                )
                slug = (
                    label.lower().replace(" + ", "-").replace(",", "").replace(" ", "-")
                )
                destination = out / f"{kbps}k-{second}s-{slug}-fhd.png"
                image.save(destination)
                images[label] = image
                records.append(
                    dict(
                        requested_kbps=kbps,
                        second=second,
                        method=label,
                        input_sha256=sha256(input_path),
                        prediction=destination.name,
                        prediction_sha256=sha256(destination),
                    )
                )
            if args.results:
                panels = [
                    ("Clean FHD reference", reference),
                    ("Official pretrained", images["Official pretrained"]),
                    ("Scratch, matched control", images["Scratch, matched control"]),
                    (
                        "Official + text fine-tuning",
                        images["Official + text fine-tuning"],
                    ),
                ]
            else:
                panels = [
                    (
                        "Decoded 540p -> FHD (bilinear)",
                        Image.open(source / f"{kbps}k/{second}s-decoded-fhd.png"),
                    ),
                    ("Official pretrained", images["Official pretrained"]),
                ]
            rows = len(panels) // 2
            canvas = Image.new(
                "RGB", (2 * w + 36, 55 + rows * (h + 40) + 35), "#f1f4f6"
            )
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 10),
                f"{kbps} Kbps | page held {second} s | identical decoded input, native pixel crops",
                font=font_large,
                fill="#15232a",
            )
            for i, (label, img) in enumerate(panels):
                left = 12 + (i % 2) * (w + 12)
                top = 55 + (i // 2) * (h + 40)
                epoch = identities.get(label, {}).get("epoch")
                draw.text(
                    (left, top),
                    label + (f" / epoch {epoch}" if epoch is not None else ""),
                    font=small,
                    fill="#15232a",
                )
                canvas.paste(img.crop(box), (left, top + 25))
            draw.text(
                (12, canvas.height - 25),
                "Offline HEVC 420 / 30 fps / 540p x2 -> FHD; no sharpening or synthetic pixels added to panels.",
                font=small,
                fill="#35434a",
            )
            canvas.save(out / f"{kbps}k-{second}s-comparison.png")
    atomic_json(
        out / "manifest.json",
        dict(
            source_manifest_sha256=sha256(source / "manifest.json"),
            source_case=old["source_case"],
            roi=roi,
            models=identities,
            records=records,
            device="CPU FP32",
            script_sha256=sha256(Path(__file__)),
            finished=time.time(),
        ),
    )
    print("Actual official model previews saved:", out, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--results", type=Path)
    run(p.parse_args())
