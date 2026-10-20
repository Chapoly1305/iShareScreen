"""Create synchronized comparison media from verified epoch reconstructions."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

from benchmarks.super_resolution.prepare_video import probe
from benchmarks.super_resolution.gallery import (
    IMAGE_SEPARATOR, crop_strip, panel_grid, refresh_gallery, write_gallery,
)
from isharescreen.enhancement.models import sha256

MODELS = ("edsr", "imdn", "safmn")
LABELS = ("Original", "Input 540p", "Bicubic", "EDSR", "IMDN", "SAFMN")
TRAINING_LABELS = ("Original", "EDSR - Pretrained", "IMDN - Pretrained", "SAFMN - Pretrained",
                   "Bicubic", "EDSR - Fine-tuned", "IMDN - Fine-tuned", "SAFMN - Fine-tuned")
def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def chapter_title(scene, chapter):
    return f"{scene['title']} - {chapter.title()}"


def video_filename(epoch, pretrained=False):
    if type(epoch) is not int or epoch < 1:
        raise ValueError("Epoch must be a positive integer")
    return f"pretrained-vs-finetuned-epoch-{epoch}.mp4" if pretrained else f"epoch-{epoch}-comparison.mp4"


def pretrained_outputs(root, data):
    """Reuse official lossless outputs only for the identical source manifest."""
    root = Path(root)
    done, environment, rows = (read(root/name) for name in ("COMPLETED.json", "environment.json", "clips.json"))
    require(not (root/"FAILED.json").exists() and len(rows) == done["videos"], "Incomplete pretrained run")
    require(environment["manifest_sha256"] == data["source_manifest_sha256"], "Pretrained inputs differ")
    require(environment["precision"] == "fp32", "Pretrained comparison requires FP32 outputs")
    identities = [(r["model"],r["clip"]) for r in rows]
    require(len(identities) == len(set(identities)), "Duplicate pretrained result")
    lock = read(Path(__file__).with_name("models.lock.json"))
    paths, evidence = {}, []
    for model in MODELS:
        state = done["status"][model]
        require(state["state"] == "completed" and state["checkpoint_sha256"] == lock[model]["sha256"]
                and state["source_sha"] == lock[model]["revision"], "Pretrained weights or architecture changed")
    for scene in data["scenes"]:
        paths[scene["id"]] = []
        for model in MODELS:
            matches = [r for r in rows if (r["model"],r["clip"]) == (model,scene["capture"])]
            require(len(matches) == 1, "Missing pretrained capture")
            row = matches[0]
            require(row["lossless_roundtrip_verified"] and Fraction(row["fps"]) == data["fps"]
                    and 0 <= scene["start_frame"] < scene["start_frame"]+scene["frames"] <= row["frames"],
                    "Pretrained range, cadence or lossless identity differs")
            path = root/row["output"]
            require(path.resolve().is_relative_to(root.resolve()) and sha256(path) == row["sha256"],
                    "Pretrained video changed or escapes run root")
            paths[scene["id"]].append(path)
            evidence.append(dict(row, scene=scene["id"], start_frame=scene["start_frame"],
                excerpt_frames=scene["frames"], checkpoint_sha256=done["status"][model]["checkpoint_sha256"],
                source_revision=done["status"][model]["source_sha"]))
    return paths, evidence


def frame_at(path, index, fps=60):
    import av
    with av.open(str(path)) as video:
        for i, frame in enumerate(video.decode(video=0)):
            if i == index:
                require(frame.pts is not None and abs(float(frame.pts*frame.time_base)-i/fps) < .0011,
                        "Poster frame timestamp differs")
                return Image.fromarray(frame.to_ndarray(format="rgb24"))
    raise ValueError("Poster frame not found")


def compose(args):
    with_pretrained = getattr(args, "pretrained", None) is not None
    filename = video_filename(args.epoch, with_pretrained)
    panel_labels = TRAINING_LABELS if with_pretrained else LABELS
    columns, count = (4,8) if with_pretrained else (3,6)
    width, height = columns*640, 840
    require(not args.output.exists(), "Use a fresh comparison directory")
    data = read(args.prepared)
    require(data["schema"] == "sr_trained_media_inputs_v1" and data["fps"] == 60, "Incorrect prepared protocol")
    official, official_evidence = pretrained_outputs(args.pretrained, data) if with_pretrained else ({}, [])
    completed = {m:read(args.predictions/m/"COMPLETED.json") for m in MODELS}
    for model, done in completed.items():
        require(not (args.predictions/model/"FAILED.json").exists()
                and done["prepared_sha256"] == sha256(args.prepared), "Incomplete or mismatched reconstruction")
        require(len(done["records"]) == len(data["scenes"]), "Missing reconstructed scene")
    inputs, evidence = {}, []
    for scene in data["scenes"]:
        paths = [args.prepared.parent/scene["files"][r]["path"] for r in ("reference","low","bicubic")]
        for r,path in zip(("reference","low","bicubic"),paths):
            require(sha256(path) == scene["files"][r]["sha256"], "Prepared video changed")
        for model in MODELS:
            record = next(r for r in completed[model]["records"] if r["scene"] == scene["id"])
            require(record["epoch"] == args.epoch and record["checkpoint_role"] == "CURRENT"
                    and record["fold"] == scene["fold"] and record["capture"] == scene["capture"]
                    and record["frames"] == scene["frames"] and record["fps"] == data["fps"]
                    and record["lossless_roundtrip_verified"], "Reconstruction identity or geometry differs")
            path = args.predictions/model/record["output"]
            require(sha256(path) == record["output_sha256"], "Reconstruction changed")
            paths.append(path)
            evidence.append(record)
        for i,path in enumerate(paths):
            s = probe(path)
            require((s["width"],s["height"]) == ((960,540) if i == 1 else (1920,1080))
                    and int(s["nb_read_frames"]) == scene["frames"] and Fraction(s["avg_frame_rate"]) == 60,
                    "Video geometry, cadence or frame count differs")
        starts = [0]*len(paths)
        if with_pretrained:
            # Same-model columns: official weights above fine-tuned weights.
            for path in official[scene["id"]]:
                s = probe(path)
                row = next(r for r in official_evidence if r["scene"] == scene["id"] and args.pretrained/r["output"] == path)
                require((s["width"],s["height"]) == (1920,1080) and Fraction(s["avg_frame_rate"]) == 60
                        and int(s["nb_read_frames"]) == row["frames"], "Pretrained video geometry or count differs")
            paths = [paths[0], *official[scene["id"]], paths[2], *paths[3:]]
            starts = [0,scene["start_frame"],scene["start_frame"],scene["start_frame"],0,0,0,0]
        inputs[scene["id"]] = (paths, starts)
    args.output.mkdir(parents=True, exist_ok=False)
    images_folder = args.output/"images"
    images_folder.mkdir()
    pictures, commands, parts = [], [], []
    for scene in data["scenes"]:
        paths, starts = inputs[scene["id"]]
        # Decode the exact same frame from the verified lossless videos, rather
        # than relying on an approximate seek or an unverified screenshot.
        picture_indices = range(8) if with_pretrained else (0,2,3,4,5)
        originals = [frame_at(paths[i],starts[i]+scene["frames"]//2) for i in picture_indices]
        overview = originals[0].copy()
        draw = ImageDraw.Draw(overview)
        for number, crop in enumerate(scene["crops"], 1):
            x,y,w,h = crop["xywh"]
            draw.rectangle((x,y,x+w-1,y+h-1),outline=(230,120,0),width=3)
            draw.text((x+4,y+4),str(number),fill=(230,120,0),font=ImageFont.load_default(size=24))
            image_path = images_folder/f"{scene['id']}-{crop['id']}.png"
            crop_strip(originals,crop["xywh"],
                       TRAINING_LABELS if with_pretrained else None, 4 if with_pretrained else None).save(image_path)
            pictures.append(dict(file=str(image_path.relative_to(args.output)),sha256=sha256(image_path),
                scene=scene["id"],source_frame=scene["poster_source_frame"],fold=scene["fold"],
                xywh=crop["xywh"],display_scale=2,display_interpolation="nearest",title=crop["title"]))
        overview.save(images_folder/f"{scene['id']}-regions.png")

    for chapter in ("overview","detail"):
        for scene in data["scenes"]:
            paths, starts = inputs[scene["id"]]
            labels = Image.new("RGBA",(width,height))
            draw = ImageDraw.Draw(labels)
            font = ImageFont.load_default(size=24)
            draw.rectangle((0,0,width-1,39),fill=(12,16,24,255))
            draw.text((16,8),chapter_title(scene,chapter),font=font,fill="white")
            draw.text((width-180,8),f"Epoch {args.epoch}",font=font,fill="white")
            for i,label in enumerate(panel_labels):
                x,y = (i%columns)*640,40+(i//columns)*400
                draw.rectangle((x,y,x+639,y+39),fill=(12,16,24,255))
                draw.text((x+16,y+8),label,font=font,fill="white")
            overlay = args.output/f"{chapter}-{scene['id']}-labels.png"
            labels.save(overlay)
            command = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-filter_complex_threads","4"]
            for path in paths:
                command += ["-threads","1","-i",str(path)]
            command += ["-loop","1","-framerate","30","-i",str(overlay)]
            filters = []
            for i in range(count):
                low = "scale=1920:1080:flags=neighbor," if not with_pretrained and i == 1 else ""
                x,y,w,h = scene["detail_xywh"]
                require((w,h) == (320,180), "Video detail crop must be exactly 320x180 for 2x viewing")
                resize = "scale=640:360:flags=lanczos" if chapter == "overview" else f"crop={w}:{h}:{x}:{y},scale=640:360:flags=neighbor"
                trim = f"trim=start_frame={starts[i]}:end_frame={starts[i]+scene['frames']},"
                filters.append(f"[{i}:v]{trim}setpts=PTS-STARTPTS,fps=30,format=rgb24,{low}{resize},setsar=1,pad=640:400:0:40:color=black[p{i}]")
            layout = "|".join(f"{(i%columns)*640}_{(i//columns)*400}" for i in range(count))
            filters += ["".join(f"[p{i}]" for i in range(count))+f"xstack=inputs={count}:layout={layout}:shortest=1,pad={width}:{height}:0:40:color=black[grid]",
                        f"[grid][{count}:v]overlay=shortest=1,scale=out_color_matrix=bt709:out_range=tv,format=yuv420p[v]"]
            path = args.output/f"{chapter}-{scene['id']}.mp4"
            frames = scene["frames"]//2
            command += ["-filter_complex",";".join(filters),"-map","[v]","-frames:v",str(frames),"-an",
                "-c:v","libx264","-preset","fast","-crf","16","-threads","4","-color_range","tv","-colorspace","bt709",
                "-color_primaries","bt709","-color_trc","bt709","-movflags","+faststart",str(path)]
            commands.append(command)
            print(f"COMPOSE {chapter} {scene['id']}",flush=True)
            subprocess.run(command,check=True,timeout=1800)
            stream = probe(path)
            require(int(stream["nb_read_frames"]) == frames and Fraction(stream["avg_frame_rate"]) == 30, "Viewing part frame count differs")
            parts.append(dict(file=path.name,frames=frames,title=chapter_title(scene,chapter)))
    concat = args.output/"chapters.txt"
    concat.write_text("".join(f"file '{p['file']}'\n" for p in parts))
    metadata = ";FFMETADATA1\n"
    elapsed = 0
    for part in parts:
        end = elapsed+part["frames"]*1000//30
        metadata += f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={elapsed}\nEND={end}\nTITLE={part['title']}\n"
        elapsed = end
    meta = args.output/"chapters.ffmetadata"
    meta.write_text(metadata)
    output = args.output/filename
    command = ["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-f","concat","-safe","0","-i",str(concat),
        "-i",str(meta),"-map","0:v:0","-map_metadata","1","-map_chapters","1","-c","copy","-movflags","+faststart",str(output)]
    subprocess.run(command,check=True,timeout=1800)
    commands.append(command)
    stream = probe(output)
    require(int(stream["nb_read_frames"]) == sum(p["frames"] for p in parts)
            and Fraction(stream["avg_frame_rate"]) == 30 and (stream["width"],stream["height"]) == (width,height), "Final comparison video differs")
    manifest = dict(epoch=args.epoch,checkpoint_role="CURRENT",prepared_sha256=sha256(args.prepared),
        prepared=data,model_outputs=evidence,video=output.name,video_sha256=sha256(output),stream=stream,
        pictures=pictures,chapters=parts,commands=commands,generator_sha256=sha256(Path(__file__)),
        gallery_generator_sha256=sha256(Path(__file__).with_name("gallery.py")),
        viewing_copy=True,viewing_fps=30,reconstruction_fps=60,quality_metrics_recomputed=False,
        image_layout=dict(columns=4 if with_pretrained else 5,separator_px=IMAGE_SEPARATOR,embedded_titles=False))
    if with_pretrained:
        manifest.update(pretrained_outputs=official_evidence, panel_labels=panel_labels, panel_columns=columns,
            pretrained_environment_sha256=sha256(args.pretrained/"environment.json"),
            pretrained_results_sha256=sha256(args.pretrained/"clips.json"))
    (args.output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    write_gallery(args.output,manifest)
    print(f"COMPLETED {output}",flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("prepared","predictions"):
        p.add_argument(f"--{name}",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--epoch",type=int,default=1500)
    p.add_argument("--pretrained",type=Path,help="Completed official-pretrained run on the same source manifest")
    p.add_argument("--refresh-gallery",action="store_true",help="Update existing PNG layout and HTML without rendering video")
    args = p.parse_args()
    if args.refresh_gallery:
        refresh_gallery(args.output)
    else:
        if args.prepared is None or args.predictions is None:
            p.error("prepared and predictions are required for video composition")
        compose(args)


if __name__ == "__main__":
    main()
