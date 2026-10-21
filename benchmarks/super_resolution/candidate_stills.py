"""Render selected recording frames and append labeled still-image comparisons."""
import argparse
from fractions import Fraction
import html
import json
from pathlib import Path
import shutil
import tempfile

from PIL import Image

from benchmarks.super_resolution.gallery import crop_strip
from benchmarks.super_resolution.render_candidate_media import MODELS, frame_at, label, load_model, read, require
from benchmarks.super_resolution.training_state import fingerprint
from isharescreen.enhancement.models import sha256


def render(args):
    import cv2
    import numpy as np
    import torch
    from benchmarks.super_resolution.candidate_training import predict

    require(not args.output.exists(), "Use a fresh still-render directory")
    source, spec, folds = read(args.manifest), read(args.spec), read(args.folds)
    require(spec["folds_sha256"] == sha256(args.folds), "Scene plan uses different folds")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True)
    records = []
    for scene in spec["scenes"]:
        clip = next(clip for clip in source["clips"] if clip["id"] == scene["capture"])
        fold = next(clip["fold"] for clip in folds["clips"] if clip["id"] == scene["capture"])
        require(Fraction(clip["fps"]) == 60 and 0 <= scene["frame"] < clip["frames"], "Invalid selected frame")
        folder = args.output / scene["id"]
        folder.mkdir()
        evidence, files = [], {}
        selected = dict(scene, fold=fold, time_s=scene["frame"] / 60)
        for role, name in (("reference", "reference"), ("low", "input")):
            path = args.manifest.parent / clip[role]
            require(sha256(path) == clip[f"{role}_sha256"], "Source pixels changed")
            image = frame_at(path, scene["frame"])
            image.save(folder / f"{name}.png")
            files[name] = dict(path=f"{scene['id']}/{name}.png", sha256=sha256(folder / f"{name}.png"))
        with Image.open(folder / "input.png") as image:
            low = np.array(image.convert("RGB"))
        require(low.shape == (540, 960, 3), "Expected 540p model input")
        Image.fromarray(cv2.resize(low, (1920, 1080), interpolation=cv2.INTER_CUBIC)).save(folder / "bicubic.png")
        files["bicubic"] = dict(path=f"{scene['id']}/bicubic.png", sha256=sha256(folder / "bicubic.png"))
        for key in MODELS:
            run = args.runs / f"{key}-fold-{fold}"
            done, policy = read(run / "COMPLETED.json"), read(run / "protocol.json")
            checkpoint = run / "best.pt"
            require(done["model"] == key and done["fold"] == fold and done["epochs"] == 1500
                    and policy["folds_sha256"] == spec["folds_sha256"], "Selected model/fold differs")
            require(done["protocol_binding"] == fingerprint(policy)
                    and sha256(checkpoint) == done["checkpoint_sha256"], "Changed selected checkpoint")
            for variant in ("pretrained", "fine-tuned"):
                model, initialization = load_model(key, variant, checkpoint, args.device, args.home)
                tensor = torch.from_numpy(low.copy()).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(args.device)
                with torch.inference_mode():
                    result = predict(model, key, tensor)
                require(result.shape == (1, 3, 1080, 1920) and torch.isfinite(result).all().item(),
                        "Invalid reconstructed pixels")
                rgb = result[0].mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                name = f"{key}-{variant}"
                target = folder / f"{name}.png"
                Image.fromarray(rgb).save(target)
                files[name] = dict(path=f"{scene['id']}/{name}.png", sha256=sha256(target))
                evidence.append(dict(model=key, variant=variant, fold=fold, initialization=initialization,
                    checkpoint_sha256=done["checkpoint_sha256"] if variant == "fine-tuned" else initialization["checkpoint_sha256"],
                    selected_epoch=done["best_epoch"] if variant == "fine-tuned" else 0,
                    protocol_binding=done["protocol_binding"]))
                del model, tensor, result
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
        records.append(dict(scene=selected, files=files, evidence=evidence))
        (args.output / "progress.json").write_text(json.dumps(records, indent=2) + "\n")
        print(f"RENDERED {scene['id']} at frame {scene['frame']} fold {fold}", flush=True)
    (args.output / "COMPLETED.json").write_text(json.dumps(dict(records=records,
        source_manifest_sha256=sha256(args.manifest), folds_sha256=sha256(args.folds),
        spec_sha256=sha256(args.spec), precision="fp32", inference_tf32=False,
        generator_sha256=sha256(Path(__file__))), indent=2) + "\n")


def write_gallery(output, manifest):
    scene_titles = {}
    for picture in manifest["pictures"]:
        scene_titles.setdefault(picture["scene"], picture.get("scene_title", picture["scene"].title()))
    page = ['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1"><title>Model Comparison</title>',
        '<style>body{font-family:system-ui;max-width:1920px;margin:32px auto;padding:0 24px;background:#f7f8fa;color:#111827}'
        'img{display:block;width:100%;height:auto;border:1px solid #94a3b8}section{margin:32px 0}p{line-height:1.6}'
        'nav{display:flex;gap:10px;flex-wrap:wrap}nav a{padding:8px 14px;background:#e2e8f0;border-radius:6px;color:#1e293b;text-decoration:none}'
        'html{scroll-behavior:smooth}</style><body><h1>Model Comparison</h1>',
        '<p>Top: pretrained. Bottom: fine-tuned. Model columns stay aligned. '
        'The first column shows Original above Bicubic; the second shows the same unenhanced 540p input '
        'with nearest-neighbor display scaling. Click any screenshot to view it at full resolution.</p><nav>']
    for scene, title in scene_titles.items():
        page.append(f'<a href="#scene-{html.escape(scene, quote=True)}">{html.escape(title)}</a>')
    page.append('</nav>')
    seen = set()
    for picture in manifest["pictures"]:
        scene = picture["scene"]
        anchor = f' id="scene-{html.escape(scene, quote=True)}"' if scene not in seen else ''
        seen.add(scene)
        name = html.escape(picture["file"], quote=True)
        page.append(f'<section{anchor}><h2>{html.escape(picture["title"])}</h2><a href="{name}">'
                    f'<img src="{name}?v={picture["sha256"][:12]}" alt="{html.escape(picture["title"], quote=True)}" loading="lazy"></a></section>')
    page.append('<p><a href="../../recording-midpoint-review/index.html">All recording previews</a></p></body></html>')
    (output / "index.html").write_text("\n".join(page) + "\n")


def compose(args):
    require(not args.output.exists(), "Use a fresh gallery directory")
    done = read(args.run / "COMPLETED.json")
    manifest = read(args.existing / "manifest.json")
    presentation = read(args.presentation_spec) if args.presentation_spec else None
    require(done["folds_sha256"] == args.folds_sha256, "Unexpected fold mapping")
    for picture in manifest["pictures"]:
        require(sha256(args.existing / picture["file"]) == picture["sha256"], "Changed existing screenshot")
    shutil.copytree(args.existing, args.output)
    labels = ("Original", "Input 540p - Unenhanced", *(f"{label(key)} - Pretrained" for key in MODELS),
              "Bicubic", "Input 540p - Unenhanced", *(f"{label(key)} - Fine-tuned" for key in MODELS))
    used = {picture["scene"] for picture in manifest["pictures"]}
    for record in done["records"]:
        scene, files = dict(record["scene"]), record["files"]
        if presentation:
            selected = next(row for row in presentation["scenes"] if row["id"] == scene["id"])
            require((selected["capture"], selected["frame"]) == (scene["capture"], scene["frame"]),
                    "Presentation adjustment must preserve rendered-frame identity")
            scene["crops"] = selected["crops"]
        require(scene["id"] not in used, "Scene already exists")
        used.add(scene["id"])
        for entry in files.values():
            require(sha256(args.run / entry["path"]) == entry["sha256"], "Changed rendered screenshot")
        roles = ["reference", "input", *(f"{key}-pretrained" for key in MODELS),
                 "bicubic", "input", *(f"{key}-fine-tuned" for key in MODELS)]
        panels = []
        for role in roles:
            with Image.open(args.run / files[role]["path"]) as image:
                panel = image.convert("RGB")
            if role == "input":
                panel = panel.resize((1920, 1080), Image.Resampling.NEAREST)
            panels.append(panel)
        for crop in scene["crops"]:
            target = args.output / "images" / f"{scene['id']}-{crop['id']}.png"
            crop_strip(panels, crop["xywh"], labels=labels, columns=6).save(target)
            manifest["pictures"].append(dict(file=str(target.relative_to(args.output)), sha256=sha256(target),
                title=f"{scene['title']} — {crop['title']}", scene=scene["id"], scene_title=scene["title"],
                source_frame=scene["frame"], time_s=scene["time_s"], fold=scene["fold"], xywh=crop["xywh"]))
        input_target = args.output / "inputs" / f"{scene['id']}-540p.png"
        shutil.copyfile(args.run / files["input"]["path"], input_target)
        manifest["raw_inputs"].append(dict(scene=scene["id"], title=scene["title"],
            image=str(input_target.relative_to(args.output)), image_sha256=sha256(input_target), width=960, height=540,
            source_frame=scene["frame"]))
    evidence = args.output / "selected-stills.json"
    evidence.write_text(json.dumps(done, indent=2) + "\n")
    manifest.update(presentation="screenshots", still_run_sha256=sha256(evidence),
                    still_generator_sha256=sha256(Path(__file__)))
    if presentation:
        shutil.copyfile(args.presentation_spec, args.output / "selected-scenes.json")
        manifest["presentation_spec_sha256"] = sha256(args.presentation_spec)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_gallery(args.output, manifest)
    print(f"Composed {len(manifest['pictures'])} screenshots across {len(used)} scenes", flush=True)


def refresh(args):
    """Keep only enlarged detail comparisons in an existing screenshot gallery."""
    root = args.output
    manifest = read(root / "manifest.json")
    removed = []
    kept = []
    for picture in manifest["pictures"]:
        path = root / picture["file"]
        require(path.resolve().is_relative_to(root.resolve()) and sha256(path) == picture["sha256"],
                "Changed or unsafe gallery image")
        if picture.get("kind") == "overview":
            removed.append(path)
        else:
            kept.append(picture)
    require(bool(kept), "Gallery must retain detail comparisons")
    manifest["pictures"] = kept
    # A screenshot gallery must not advertise superseded video deliverables.
    for key in ("video", "video_sha256", "chapters", "commands"):
        manifest.pop(key, None)
    manifest["presentation"] = "screenshots"
    manifest["still_generator_sha256"] = sha256(Path(__file__))
    with tempfile.TemporaryDirectory(prefix=".gallery-", dir=root) as temporary:
        staged = Path(temporary)
        (staged / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        write_gallery(staged, manifest)
        for name in ("manifest.json", "index.html"):
            (staged / name).replace(root / name)
    for path in removed:
        path.unlink()
    print(f"Removed {len(removed)} overview images; kept {len(kept)} enlarged comparisons", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    render_parser = sub.add_parser("render")
    for name in ("manifest", "spec", "folds", "runs", "home", "output"):
        render_parser.add_argument(f"--{name}", type=Path, required=True)
    render_parser.add_argument("--device", default="cuda")
    compose_parser = sub.add_parser("compose")
    for name in ("run", "existing", "output"):
        compose_parser.add_argument(f"--{name}", type=Path, required=True)
    compose_parser.add_argument("--folds-sha256", required=True)
    compose_parser.add_argument("--presentation-spec", type=Path)
    refresh_parser = sub.add_parser("refresh")
    refresh_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "render":
        render(args)
    elif args.mode == "compose":
        compose(args)
    else:
        refresh(args)


if __name__ == "__main__":
    main()
