"""Render auditable pretrained-versus-fine-tuned candidate comparison media."""
import argparse
from fractions import Fraction
import html
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution import candidate_training, training_state
from benchmarks.super_resolution.gallery import IMAGE_SEPARATOR, panel_grid, write_gallery, crop_strip
from benchmarks.super_resolution.prepare_video import probe
from isharescreen.enhancement import candidates
from isharescreen.enhancement.models import sha256


# Only these candidates have an official native x2 starting point.  The x4-only
# candidates have a deterministic x4-to-x2 initialization, but calling that a
# pretrained x2 comparison would be misleading.
MODELS = ("plainusr", "quicksrnet", "rt4ksr", "efrlfn")


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def label(key):
    return candidates.specification(key)["label"].replace(" - Pretrained", "")


def comparison_labels(key):
    return ("Original", "Input 540p", "Bicubic",
            f"{label(key)} - Pretrained", f"{label(key)} - Fine-tuned")


def comparison_grid(strips):
    """Reuse labeled single-model cells in the shared two-row comparison order."""
    require(len(strips) == len(MODELS), "Missing model rows")
    require(len({image.size for image in strips}) == 1, "Row dimensions differ")
    width, height = strips[0].size
    require((width - 4 * IMAGE_SEPARATOR) % 5 == 0, "Unknown strip width")
    cell_width = (width - 4 * IMAGE_SEPARATOR) // 5

    def cell(row, column):
        x = column * (cell_width + IMAGE_SEPARATOR)
        return strips[row].crop((x, 0, x + cell_width, height))

    panels = [cell(0, 0), cell(0, 1), *(cell(row, 3) for row in range(len(MODELS))),
              cell(0, 2), cell(0, 1), *(cell(row, 4) for row in range(len(MODELS)))]
    return panel_grid(panels, 6)


def refresh_still_headers(root):
    """Restore model labels in existing galleries while preserving crop pixels."""
    root = Path(root)
    # Stage and verify all source images before replacing any asset.
    with tempfile.TemporaryDirectory(prefix="candidate-headers-") as temporary:
        staged = Path(temporary)
        updates = []
        for key in MODELS:
            folder = root / key
            manifest = read(folder / "manifest.json")
            require(sha256(folder / manifest["video"]) == manifest["video_sha256"], "Changed viewing video")
            for picture in manifest["pictures"]:
                path = folder / picture["file"]
                require(sha256(path) == picture["sha256"], "Changed comparison PNG")
                with Image.open(path) as source:
                    width, height = (2 * value for value in picture["xywh"][2:])
                    expected_width = 5 * width + 4 * IMAGE_SEPARATOR
                    header = manifest.get("image_layout", {}).get("panel_header_px", 0)
                    require(source.size == (expected_width, height + header), "Unknown screenshot layout")
                    panels = []
                    for column, name in enumerate(comparison_labels(key)):
                        x = column * (width + IMAGE_SEPARATOR)
                        content = source.crop((x, header, x + width, header + height))
                        panel = Image.new("RGB", (width, height + 40), "#111827")
                        ImageDraw.Draw(panel).text((14, 8), name, font=ImageFont.load_default(size=24), fill="white")
                        panel.paste(content, (0, 40))
                        require(panel.crop((0, 40, width, height + 40)).tobytes() == content.tobytes(),
                                "Crop pixels changed")
                        panels.append(panel)
                    result = panel_grid(panels, 5)
                target = staged / key / picture["file"]
                target.parent.mkdir(parents=True, exist_ok=True)
                result.save(target)
                picture["sha256"] = sha256(target)
                updates.append((target, path))
            manifest["image_layout"].update(panel_header_px=40)
            manifest["generator_sha256"] = sha256(Path(__file__))
            (staged / key / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            write_gallery(staged / key, manifest)
            updates.extend((staged / key / name, folder / name) for name in ("manifest.json", "index.html"))
        matrix_folder = root / "comparison-v2"
        manifest = read(matrix_folder / "manifest.json")
        require(sha256(matrix_folder / manifest["video"]) == manifest["video_sha256"], "Changed matrix video")
        for picture in manifest["pictures"]:
            require(sha256(matrix_folder / picture["file"]) == picture["sha256"], "Changed matrix PNG")
            strips = []
            for key in MODELS:
                with Image.open(staged / key / picture["file"]) as strip:
                    strips.append(strip.convert("RGB"))
            image = comparison_grid(strips)
            target = staged / "comparison-v2" / picture["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            picture["sha256"] = sha256(target)
            updates.append((target, matrix_folder / picture["file"]))
        manifest["image_layout"] = dict(columns=6, separator_px=IMAGE_SEPARATOR,
                                       panel_header_px=40, embedded_titles=False)
        (staged / "comparison-v2" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        page = (matrix_folder / "index.html").read_text()
        import re
        for picture in manifest["pictures"]:
            filename = html.escape(picture["file"], quote=True)
            page = re.sub(r'(src=[\"\'])' + re.escape(filename) + r'(?:\?v=[a-f0-9]+)?([\"\'])',
                          lambda match: f'{match[1]}{filename}?v={picture["sha256"][:12]}{match[2]}', page)
        (staged / "comparison-v2" / "index.html").write_text(page)
        updates.extend((staged / "comparison-v2" / name, matrix_folder / name)
                       for name in ("manifest.json", "index.html"))
        for source, destination in updates:
            destination.write_bytes(source.read_bytes())
    print("Updated 30 screenshots and five galleries; comparison pixels verified", flush=True)


def frame_at(path, index, fps=60):
    import av
    with av.open(str(path)) as video:
        for number, frame in enumerate(video.decode(video=0)):
            if number == index:
                require(frame.pts is not None and abs(float(frame.pts * frame.time_base) - index / fps) < .0011,
                        "Poster-frame timestamp differs")
                return Image.fromarray(frame.to_ndarray(format="rgb24"))
    raise ValueError("Poster frame not found")


def load_model(key, variant, checkpoint, device, home):
    import torch
    model, initialization = candidate_training.load(key, home)
    if variant == "fine-tuned":
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        require(state["model"] == key and "state_dict" in state and "protocol_binding" in state,
                "Invalid fine-tuned checkpoint")
        torch.nn.Module.load_state_dict(model, state["state_dict"], strict=True)
    else:
        require(initialization["kind"] == "strict_official_x2", "Pretrained candidate is not native x2")
    model.to(device).eval().requires_grad_(False)
    return model, initialization


def infer(args):
    import av
    import cv2
    import torch
    from benchmarks.super_resolution.video_benchmark import LosslessWriter, verify_lossless

    data = read(args.prepared)
    require(data["schema"] == "sr_trained_media_inputs_v1" and data["fps"] == 60, "Incorrect prepared media")
    require(args.model in MODELS, "Model does not have official native x2 weights")
    require(args.variant in ("pretrained", "fine-tuned"), "Unknown variant")
    require(not args.output.exists(), "Use a fresh output directory")
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    outputs, evidence = [], []
    args.output.mkdir(parents=True)
    try:
        for scene in data["scenes"]:
            run = args.runs / f"{args.model}-fold-{scene['fold']}"
            done, protocol = read(run / "COMPLETED.json"), read(run / "protocol.json")
            require(done["model"] == args.model and done["fold"] == scene["fold"] and done["epochs"] == 1500,
                    "Wrong completed fold")
            require(done["protocol_binding"] == training_state.fingerprint(protocol), "Unbound checkpoint receipt")
            checkpoint = run / "best.pt"
            require(checkpoint.is_file() and sha256(checkpoint) == done["checkpoint_sha256"], "Changed selected checkpoint")
            model, initialization = load_model(args.model, args.variant, checkpoint, args.device, args.home)
            low = args.prepared.parent / scene["files"]["low"]["path"]
            require(sha256(low) == scene["files"]["low"]["sha256"], "Prepared input changed")
            output = args.output / f"{scene['id']}.mkv"
            sink = LosslessWriter(output, 1920, 1080, Fraction(60))
            started = time.monotonic()
            try:
                with av.open(str(low)) as video, torch.inference_mode():
                    for index, frame in enumerate(video.decode(video=0)):
                        require(frame.pts is not None and abs(float(frame.pts * frame.time_base) - index / 60) < .0011,
                                "Input timestamp differs")
                        rgb = frame.to_ndarray(format="rgb24")
                        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float().div_(255).to(args.device)
                        prediction = candidate_training.predict(model, args.model, tensor)
                        require(prediction.shape == (1, 3, 1080, 1920) and torch.isfinite(prediction).all().item(),
                                "Candidate output geometry differs")
                        restored = prediction[0].mul(255).round().clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
                        sink.write(restored)
                        if index == scene["frames"] // 2:
                            Image.fromarray(restored).save(args.output / f"{scene['id']}.png")
                        if (index + 1) % 60 == 0:
                            print(f"RENDER {args.model} {args.variant} {scene['id']} {index + 1}/{scene['frames']}", flush=True)
            finally:
                sink.close()
            require(sink.count == scene["frames"], "Truncated reconstruction")
            verify_lossless(output, sink.count, sink.digest.hexdigest())
            outputs.append(dict(scene=scene["id"], output=output.name, output_sha256=sha256(output),
                                decoded_rgb_sequence_sha256=sink.digest.hexdigest(), frames=sink.count,
                                lossless_roundtrip_verified=True, elapsed_s=time.monotonic() - started))
            evidence.append(dict(scene=scene["id"], fold=scene["fold"], checkpoint=str(checkpoint),
                                 checkpoint_sha256=sha256(checkpoint), best_epoch=done["best_epoch"],
                                 protocol_binding=done["protocol_binding"], initialization=initialization))
            del model
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        (args.output / "COMPLETED.json").write_text(json.dumps(dict(model=args.model, variant=args.variant,
            prepared_sha256=sha256(args.prepared), outputs=outputs, evidence=evidence, finished=time.time()), indent=2) + "\n")
    except Exception as error:
        (args.output / "FAILED.json").write_text(json.dumps(dict(error=repr(error), time=time.time()), indent=2) + "\n")
        raise


def compose(args):
    data = read(args.prepared)
    require(not args.output.exists() and args.model in MODELS, "Use a new output and a native-x2 model")
    variants = {name: read(args.predictions / args.model / name / "COMPLETED.json")
                for name in ("pretrained", "fine-tuned")}
    require(all(value["prepared_sha256"] == sha256(args.prepared) for value in variants.values()), "Mismatched inputs")
    args.output.mkdir(parents=True)
    images = args.output / "images"
    images.mkdir()
    title = label(args.model)
    panels = comparison_labels(args.model)
    parts, pictures, commands = [], [], []
    for scene in data["scenes"]:
        base = args.prepared.parent / scene["files"]["reference"]["path"]
        low = args.prepared.parent / scene["files"]["low"]["path"]
        bicubic = args.prepared.parent / scene["files"]["bicubic"]["path"]
        output_paths = []
        for variant, record in variants.items():
            row = next(row for row in record["outputs"] if row["scene"] == scene["id"])
            path = args.predictions / args.model / variant / row["output"]
            require(row["lossless_roundtrip_verified"] and sha256(path) == row["output_sha256"], "Changed reconstruction")
            output_paths.append(path)
        all_paths = (base, low, bicubic, *output_paths)
        starts = (0, 0, 0, 0, 0)
        source = scene["frames"] // 2
        images_at = [frame_at(path, source, data["fps"]) for path in all_paths]
        images_at[1] = images_at[1].resize((1920, 1080), Image.Resampling.NEAREST)
        # Preserve model/variant headers; descriptive scene titles live in HTML.
        for crop in scene["crops"]:
            x, y, width, height = crop["xywh"]
            picture = images / f"{scene['id']}-{crop['id']}.png"
            crop_strip(images_at, crop['xywh'], labels=panels, columns=5).save(picture)
            pictures.append(dict(file=str(picture.relative_to(args.output)), sha256=sha256(picture), scene=scene["id"],
                                 source_frame=scene["poster_source_frame"], fold=scene["fold"], xywh=crop["xywh"],
                                 display_scale=2, display_interpolation="nearest", title=f"{title}: {crop['title']}"))
        for chapter in ("overview", "detail"):
            overlay = args.output / f"{chapter}-{scene['id']}-labels.png"
            canvas = Image.new("RGBA", (3200, 440))
            draw, font = ImageDraw.Draw(canvas), ImageFont.load_default(size=24)
            draw.rectangle((0, 0, 3199, 39), fill=(12, 16, 24, 255))
            draw.text((16, 8), f"{title} - {scene['title']} - {chapter.title()}", font=font, fill="white")
            for index, text in enumerate(panels):
                x = index * 640
                draw.rectangle((x, 40, x + 639, 79), fill=(12, 16, 24, 255))
                draw.text((x + 16, 48), text, font=font, fill="white")
            canvas.save(overlay)
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-filter_complex_threads", "4"]
            for path in all_paths:
                command += ["-threads", "1", "-i", str(path)]
            command += ["-loop", "1", "-framerate", "30", "-i", str(overlay)]
            filters = []
            for index in range(5):
                x, y, width, height = scene["detail_xywh"]
                resize = "scale=640:360:flags=lanczos" if chapter == "overview" else f"crop={width}:{height}:{x}:{y},scale=640:360:flags=neighbor"
                low_scale = "scale=1920:1080:flags=neighbor," if index == 1 else ""
                filters.append(f"[{index}:v]trim=end_frame={scene['frames']},setpts=PTS-STARTPTS,fps=30,format=rgb24,{low_scale}{resize},setsar=1,pad=640:400:0:40:color=black[p{index}]")
            filters += ["".join(f"[p{index}]" for index in range(5)) + "xstack=inputs=5:layout=0_0|640_0|1280_0|1920_0|2560_0:shortest=1[stack]",
                        "[stack]pad=3200:440:0:40:color=black[grid]",
                        "[grid][5:v]overlay=shortest=1,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]"]
            output = args.output / f"{chapter}-{scene['id']}.mp4"
            command += ["-filter_complex", ";".join(filters), "-map", "[v]", "-frames:v", str(scene["frames"] // 2), "-an",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-threads", "4", "-movflags", "+faststart", str(output)]
            subprocess.run(command, check=True, timeout=1800)
            stream = probe(output)
            require((stream["width"], stream["height"]) == (3200, 440) and int(stream["nb_read_frames"]) == scene["frames"] // 2,
                    "Comparison-video geometry differs")
            parts.append(dict(file=output.name, frames=scene["frames"] // 2, title=f"{title} - {scene['title']} - {chapter.title()}"))
            commands.append(command)
    chapters = args.output / "chapters.txt"
    chapters.write_text("".join(f"file '{part['file']}'\n" for part in parts))
    video = args.output / f"{args.model}-pretrained-vs-finetuned.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "concat", "-safe", "0", "-i", str(chapters),
                    "-c", "copy", "-movflags", "+faststart", str(video)], check=True, timeout=1800)
    manifest = dict(epoch=1500, model=args.model, panel_labels=panels, prepared_sha256=sha256(args.prepared),
                    video=video.name, video_sha256=sha256(video), pictures=pictures, chapters=parts, commands=commands,
                    image_layout=dict(columns=5, separator_px=IMAGE_SEPARATOR, embedded_titles=False, panel_header_px=40), viewing_copy=True,
                    quality_metrics_recomputed=False, generator_sha256=sha256(Path(__file__)))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_gallery(args.output, manifest)


def compose_matrix(args):
    """Compare model columns with pretrained above fine-tuned on one timeline."""
    data = read(args.prepared)
    require(not args.output.exists(), "Use a new output directory")
    prepared_sha = sha256(args.prepared)
    variants = {
        model: {variant: read(args.predictions / model / variant / "COMPLETED.json")
                for variant in ("pretrained", "fine-tuned")}
        for model in MODELS
    }
    for model, entries in variants.items():
        for variant, record in entries.items():
            require(record["model"] == model and record["variant"] == variant
                    and record["prepared_sha256"] == prepared_sha, "Mismatched reconstruction inputs")
    args.output.mkdir(parents=True)
    images = args.output / "images"
    images.mkdir()
    labels = ("Original", "Input 540p - Unenhanced", *(f"{label(model)} - Pretrained" for model in MODELS),
              "Bicubic", "Input 540p - Unenhanced", *(f"{label(model)} - Fine-tuned" for model in MODELS))
    columns, width, height, cell_width, cell_height = 6, 3840, 840, 640, 400
    pictures, parts, commands, inputs, raw_inputs = [], [], [], {}, []
    raw_folder = args.output / "inputs"
    raw_folder.mkdir()
    for scene in data["scenes"]:
        paths, evidence = [], []
        for variant, role in (("pretrained", "reference"), ("fine-tuned", "bicubic")):
            path = args.prepared.parent / scene["files"][role]["path"]
            require(sha256(path) == scene["files"][role]["sha256"], "Changed reference video")
            paths.append(path)
            low = args.prepared.parent / scene["files"]["low"]["path"]
            require(sha256(low) == scene["files"]["low"]["sha256"], "Changed model input")
            paths.append(low)
            for model in MODELS:
                record = variants[model][variant]
                require(len(record["outputs"]) == len(data["scenes"]), "Missing reconstructed scene")
                row = next(item for item in record["outputs"] if item["scene"] == scene["id"])
                path = args.predictions / model / variant / row["output"]
                require(row["lossless_roundtrip_verified"] and row["frames"] == scene["frames"]
                        and sha256(path) == row["output_sha256"], "Changed reconstruction")
                paths.append(path)
                evidence.append(dict(model=model, variant=variant, **row))
        inputs[scene["id"]] = paths
        stills = [frame_at(path, scene["frames"] // 2, data["fps"]) for path in paths]
        input_png = raw_folder / f"{scene['id']}-540p.png"
        input_video = raw_folder / f"{scene['id']}-540p.mkv"
        stills[1].save(input_png)
        shutil.copyfile(low, input_video)
        require(sha256(input_video) == scene["files"]["low"]["sha256"], "Input-copy hash differs")
        raw_inputs.append(dict(scene=scene["id"], title=scene["title"],
            image=str(input_png.relative_to(args.output)), image_sha256=sha256(input_png),
            video=str(input_video.relative_to(args.output)), video_sha256=sha256(input_video),
            width=960, height=540, source_frame=scene["poster_source_frame"]))
        for index in (1, 7):
            require(stills[index].size == (960, 540), "Unexpected input size")
            stills[index] = stills[index].resize((1920, 1080), Image.Resampling.NEAREST)
        for crop in scene["crops"]:
            target = images / f"{scene['id']}-{crop['id']}.png"
            crop_strip(stills, crop["xywh"], labels=labels, columns=columns).save(target)
            picture_title = crop['title'] if crop['title'].startswith(scene['title']) else f"{scene['title']} — {crop['title']}"
            pictures.append(dict(file=str(target.relative_to(args.output)), sha256=sha256(target),
                title=picture_title, scene=scene["id"],
                source_frame=scene["poster_source_frame"], fold=scene["fold"], xywh=crop["xywh"]))
    # Show all overviews first, followed by all enlarged detail excerpts.
    for chapter in ("overview", "detail"):
        for scene in data["scenes"]:
            paths = inputs[scene["id"]]
            overlay = args.output / f"{chapter}-{scene['id']}-labels.png"
            canvas = Image.new("RGBA", (width, height))
            draw, font = ImageDraw.Draw(canvas), ImageFont.load_default(size=25)
            draw.rectangle((0, 0, width - 1, 39), fill="#0c1018")
            draw.text((16, 7), f"Model Comparison · {scene['title']} · {chapter.title()}", font=font, fill="white")
            for index, name in enumerate(labels):
                x = (index % columns) * cell_width
                y = 40 + (index // columns) * cell_height
                draw.rectangle((x, y, x + cell_width - 1, y + 39), fill="#111827")
                draw.text((x + 14, y + 7), name, font=font, fill="white")
            # Visible dividers extend through both panel headers and content.
            for column in range(1, columns):
                x = column * cell_width
                draw.rectangle((x - 2, 40, x + 1, height - 1), fill="#94a3b8")
            draw.rectangle((0, 438, width - 1, 441), fill="#94a3b8")
            canvas.save(overlay)
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
                       "-filter_complex_threads", "4"]
            for path in paths:
                command += ["-threads", "1", "-i", str(path)]
            command += ["-loop", "1", "-framerate", "30", "-i", str(overlay)]
            filters = []
            for index in range(len(labels)):
                x, y, crop_width, crop_height = scene["detail_xywh"]
                resize = "scale=640:360:flags=lanczos" if chapter == "overview" else (
                    f"crop={crop_width}:{crop_height}:{x}:{y},scale=640:360:flags=neighbor")
                if index in (1, 7):
                    resize = "scale=640:360:flags=neighbor" if chapter == "overview" else (
                        f"scale=1920:1080:flags=neighbor,{resize}")
                filters.append(f"[{index}:v]trim=end_frame={scene['frames']},setpts=PTS-STARTPTS,"
                    f"fps=30,format=rgb24,{resize},setsar=1,pad=640:400:0:40:color=black[p{index}]")
            layout = "|".join(f"{(index % columns)*640}_{(index // columns)*400}" for index in range(len(labels)))
            filters += ["".join(f"[p{index}]" for index in range(len(labels))) +
                        f"xstack=inputs={len(labels)}:layout={layout}:shortest=1,pad={width}:{height}:0:40:color=black[grid]",
                        f"[grid][{len(labels)}:v]overlay=shortest=1,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]"]
            output = args.output / f"{chapter}-{scene['id']}.mp4"
            count = scene["frames"] // 2
            command += ["-filter_complex", ";".join(filters), "-map", "[v]", "-frames:v", str(count),
                        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16", "-threads", "4",
                        "-movflags", "+faststart", str(output)]
            print(f"COMPOSE {chapter} {scene['id']}", flush=True)
            subprocess.run(command, check=True, timeout=1800)
            stream = probe(output)
            require((stream["width"], stream["height"], int(stream["nb_read_frames"])) == (width, height, count),
                    "Comparison-video geometry differs")
            parts.append(dict(file=output.name, frames=count, title=f"{scene['title']} — {chapter.title()}"))
            commands.append(command)
    chapters = args.output / "chapters.txt"
    chapters.write_text("".join(f"file '{part['file']}'\n" for part in parts))
    metadata, elapsed = ";FFMETADATA1\n", 0
    for part in parts:
        end = elapsed + part["frames"] * 1000 // 30
        metadata += f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={elapsed}\nEND={end}\nTITLE={part['title']}\n"
        elapsed = end
    meta = args.output / "chapters.ffmetadata"
    meta.write_text(metadata)
    video = args.output / "candidate-models-pretrained-vs-finetuned.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-f", "concat", "-safe", "0",
        "-i", str(chapters), "-i", str(meta), "-map", "0:v:0", "-map_metadata", "1", "-map_chapters", "1",
        "-c", "copy", "-movflags", "+faststart", str(video)], check=True, timeout=1800)
    stream = probe(video)
    require((stream["width"], stream["height"], int(stream["nb_read_frames"])) ==
            (width, height, sum(part["frames"] for part in parts)), "Incomplete combined video")
    manifest = dict(epoch=1500, models=list(MODELS), panel_labels=labels, panel_columns=columns,
        prepared_sha256=prepared_sha, video=video.name, video_sha256=sha256(video), pictures=pictures,
        chapters=parts, commands=commands, raw_inputs=raw_inputs,
        input_display="nearest-neighbor only; no interpolation smoothing or model enhancement",
        image_layout=dict(columns=columns, separator_px=IMAGE_SEPARATOR,
            panel_header_px=40, embedded_titles=False), viewing_copy=True, quality_metrics_recomputed=False,
        generator_sha256=sha256(Path(__file__)))
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_matrix_gallery(args.output, manifest)


def write_matrix_gallery(output, manifest):
    """Use the same panel order for the video and every labeled screenshot."""
    page = ['<!doctype html><html lang="en"><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width,initial-scale=1">',
        '<title>Model Comparison</title>',
        '<style>body{font-family:system-ui;max-width:1920px;margin:32px auto;padding:0 24px;background:#f7f8fa;color:#111827}'
        'video,img{width:100%;height:auto;background:#fff}img{display:block;border:1px solid #94a3b8}'
        'section{margin:32px 0}button{padding:8px 14px;margin:0 8px 8px 0;cursor:pointer}</style>',
        '<body><h1>Model Comparison</h1>',
        '<p>Each model stays in the same column. Top: pretrained. Bottom: fine-tuned. '
        'The first column shows Original above Bicubic. The second column shows the same unenhanced 540p input '
        'in both rows, enlarged with nearest-neighbor for comparison.</p>',
        f'<video id="comparison" controls preload="metadata" src="{html.escape(manifest["video"], quote=True)}"></video>',
        '<nav aria-label="Video chapters">']
    elapsed = 0
    for part in manifest["chapters"]:
        page.append(f'<button type="button" onclick="document.getElementById(\'comparison\').currentTime={elapsed}">'
                    f'{html.escape(part["title"])}</button>')
        elapsed += part["frames"] / 30
    page.append('</nav>')
    for picture in manifest["pictures"]:
        name = html.escape(picture["file"], quote=True)
        page.append(f'<section><h2>{html.escape(picture["title"])}</h2><a href="{name}">'
                    f'<img src="{name}?v={picture["sha256"][:12]}" alt="{html.escape(picture["title"], quote=True)}" loading="lazy"></a></section>')
    page.append('</body></html>\n')
    (Path(output) / "index.html").write_text("\n".join(page))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    render = sub.add_parser("infer")
    render.add_argument("--prepared", type=Path, required=True)
    render.add_argument("--runs", type=Path, required=True)
    render.add_argument("--home", type=Path, required=True)
    render.add_argument("--model", choices=MODELS, required=True)
    render.add_argument("--variant", choices=("pretrained", "fine-tuned"), required=True)
    render.add_argument("--output", type=Path, required=True)
    render.add_argument("--device", default="cuda")
    compose_parser = sub.add_parser("compose")
    compose_parser.add_argument("--prepared", type=Path, required=True)
    compose_parser.add_argument("--predictions", type=Path, required=True)
    compose_parser.add_argument("--model", choices=MODELS, required=True)
    compose_parser.add_argument("--output", type=Path, required=True)
    matrix = sub.add_parser("compose-matrix")
    matrix.add_argument("--prepared", type=Path, required=True)
    matrix.add_argument("--predictions", type=Path, required=True)
    matrix.add_argument("--output", type=Path, required=True)
    refresh = sub.add_parser("refresh-headers")
    refresh.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "infer":
        infer(args)
    elif args.mode == "compose":
        compose(args)
    elif args.mode == "compose-matrix":
        compose_matrix(args)
    else:
        refresh_still_headers(args.root)


if __name__ == "__main__":
    main()
