"""Prepare aligned excerpts and render held-out CURRENT epoch checkpoints."""
import argparse
from fractions import Fraction
from itertools import islice
import json
from pathlib import Path
import re
import time

from benchmarks.super_resolution import train_epochs, training_state
from benchmarks.super_resolution.folds import validate
from benchmarks.super_resolution.training_state import atomic_json
from isharescreen.enhancement.models import Upscaler, sha256


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare(args):
    import cv2
    from PIL import Image
    from benchmarks.super_resolution.video_benchmark import aligned_frames, LosslessWriter, verify_lossless
    cv2.setNumThreads(1)
    source, spec, folds = read(args.manifest), read(args.spec), validate(read(args.folds))
    require(not args.output.exists(), "Use a fresh prepared directory")
    for scene in spec["scenes"]:
        require(re.fullmatch(r"[a-z][a-z0-9-]*", scene["id"]), "Unsafe scene ID")
        clip = next(c for c in source["clips"] if c["id"] == scene["capture"])
        require(Fraction(clip["fps"]) == 60 and scene["frames"] > 0 and scene["frames"] % 2 == 0
                and 0 <= scene["start_frame"] < scene["start_frame"]+scene["frames"] <= clip["frames"],
                "Invalid aligned 60fps excerpt")
        for role in ("reference", "low"):
            require(sha256(args.manifest.parent / clip[role]) == clip[f"{role}_sha256"], "Video input changed")
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema="sr_trained_media_inputs_v1", scenes=[], source_manifest_sha256=sha256(args.manifest),
        folds_sha256=sha256(args.folds), spec_sha256=sha256(args.spec), source_manifest=str(args.manifest),
        reference_kind=source["reference"], fps=60, generator_sha256=sha256(Path(__file__)))
    for scene in spec["scenes"]:
        clip = next(c for c in source["clips"] if c["id"] == scene["capture"])
        fold = next(c["fold"] for c in folds["clips"] if c["id"] == scene["capture"])
        folder = args.output / scene["id"]
        folder.mkdir()
        writers = {role: LosslessWriter(folder / f"{role}.mkv", w, h, Fraction(60))
                   for role,w,h in (("reference",1920,1080),("low",960,540),("bicubic",1920,1080))}
        count = 0
        try:
            frames = aligned_frames(args.manifest.parent/clip["reference"], args.manifest.parent/clip["low"],
                                    clip["frames"], Fraction(clip["fps"]))
            for i, high, low in islice(frames, scene["start_frame"], scene["start_frame"]+scene["frames"]):
                values = dict(reference=high, low=low, bicubic=cv2.resize(low,(1920,1080),interpolation=cv2.INTER_CUBIC))
                for role, rgb in values.items():
                    writers[role].write(rgb)
                    if count == scene["frames"]//2:
                        Image.fromarray(rgb).save(folder / f"{role}.png")
                count += 1
        finally:
            for writer in writers.values():
                writer.close()
        require(count == scene["frames"], "Truncated excerpt")
        files = {}
        for role, writer in writers.items():
            path = folder / f"{role}.mkv"
            verify_lossless(path, count, writer.digest.hexdigest())
            files[role] = dict(path=str(path.relative_to(args.output)), sha256=sha256(path),
                               rgb_sequence_sha256=writer.digest.hexdigest())
        result["scenes"].append(dict(scene, fold=fold, files=files, poster_source_frame=scene["start_frame"]+count//2,
                                     source_reference_sha256=clip["reference_sha256"], source_low_sha256=clip["low_sha256"]))
        atomic_json(args.output / "manifest.json", result)
        print(f"PREPARED {scene['id']} fold={fold} frames={count}", flush=True)
    atomic_json(args.output / "COMPLETED.json", dict(scenes=len(result["scenes"]), manifest_sha256=sha256(args.output/"manifest.json")))


def checkpoint_identity(folder, key, scene, epoch, data, audit):
    """Reject training-fold leakage, selected-best substitution and changed states."""
    policy, done = read(folder/"protocol.json"), read(folder/"COMPLETED.json")
    job = f"{key}-fold-{scene['fold']}"
    require(audit["complete_cv"] and audit["folds_sha256"] == data["folds_sha256"] == policy["folds_sha256"],
            "Fold identity differs from audited training")
    require(policy["schema"] == "sr_true_epoch_protocol_v1" and train_epochs.recorded_sources_match(policy["sources"]),
            "Training source identity changed")
    require(policy["schedule"]["job"]["model"] == key and policy["schedule"]["job"]["fold"] == scene["fold"]
            and scene["capture"] in policy["schedule"]["job"]["validation_captures"]
            and scene["capture"] not in policy["schedule"]["job"]["train_captures"], "Capture is not held out for this checkpoint")
    require(policy["schedule"]["epochs"] == done["epochs"] == epoch, "Wrong epoch ceiling")
    require(sha256(folder/"last.pt") == audit["evidence"][job]["last.pt"], "CURRENT checkpoint differs from completed audit")
    require(training_state.fingerprint(policy) == done["protocol_binding"], "Checkpoint policy binding differs")
    return policy


def infer(args):
    import av
    import cv2
    import torch
    from PIL import Image
    from benchmarks.super_resolution.setup_models import verify_source
    from benchmarks.super_resolution.video_benchmark import LosslessWriter, verify_lossless
    data, audit = read(args.prepared), read(args.audit)
    require(not args.output.exists(), "Use a fresh model-output directory; no automatic retry")
    require(data["schema"] == "sr_trained_media_inputs_v1", "Not prepared comparison inputs")
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = read(Path(__file__).with_name("models.lock.json"))[args.model]
    verify_source(args.home/"vendor"/spec["directory"], args.model, spec["revision"])
    require(sha256(args.home/"checkpoints"/spec["checkpoint"]) == spec["sha256"], "Official architecture initializer changed")
    policies = {}
    for scene in data["scenes"]:
        folder = args.runs / f"{args.model}-fold-{scene['fold']}"
        policies[scene["id"]] = checkpoint_identity(folder, args.model, scene, args.epoch, data, audit)
        low = args.prepared.parent / scene["files"]["low"]["path"]
        require(sha256(low) == scene["files"]["low"]["sha256"], "Prepared input changed")
    args.output.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for scene in data["scenes"]:
            folder = args.runs / f"{args.model}-fold-{scene['fold']}"
            policy = policies[scene["id"]]
            state = torch.load(folder/"last.pt", map_location="cpu", weights_only=True)
            md = state["metadata"]
            require(state["binding"] == training_state.fingerprint(policy)
                    and md["epoch"] == args.epoch and md["cursor"] == 0 and md["step"] == policy["schedule"]["total_updates"],
                    "CURRENT recovery state is not the requested complete epoch")
            require(all(torch.isfinite(v).all().item() for v in state["state_dict"].values()), "Nonfinite weights")
            adapter = Upscaler(args.model, device=args.device, home=args.home, precision="fp32")
            torch.nn.Module.load_state_dict(adapter.model, state["state_dict"], strict=True)
            adapter.model.eval().requires_grad_(False)
            path = args.output / f"{scene['id']}.mkv"
            sink = LosslessWriter(path, 1920, 1080, Fraction(data["fps"]))
            started = time.monotonic()
            try:
                with av.open(str(args.prepared.parent / scene["files"]["low"]["path"])) as video:
                    for i, frame in enumerate(video.decode(video=0)):
                        require(frame.pts is not None and abs(float(frame.pts*frame.time_base)-i/data["fps"]) < .0011,
                                "Input timestamp differs")
                        rgb, _ = adapter(frame.to_ndarray(format="rgb24"))
                        require(rgb.shape == (1080,1920,3), "Output geometry changed")
                        sink.write(rgb)
                        if i == scene["frames"]//2:
                            Image.fromarray(rgb).save(args.output / f"{scene['id']}.png")
                        if (i+1) % 60 == 0:
                            print(f"RENDER {args.model} {scene['id']} {i+1}/{scene['frames']}", flush=True)
            finally:
                sink.close()
            require(sink.count == scene["frames"], "Wrong output frame count")
            verify_lossless(path, sink.count, sink.digest.hexdigest())
            records.append(dict(model=args.model, scene=scene["id"], capture=scene["capture"], fold=scene["fold"], epoch=args.epoch,
                checkpoint=str(folder/"last.pt"), checkpoint_sha256=sha256(folder/"last.pt"), checkpoint_role="CURRENT",
                protocol_binding=state["binding"], precision="fp32", inference_tf32=False,
                frames=sink.count, fps=data["fps"], output=path.name, output_sha256=sha256(path),
                decoded_rgb_sequence_sha256=sink.digest.hexdigest(), lossless_roundtrip_verified=True,
                poster_source_frame=scene["poster_source_frame"], elapsed_s=time.monotonic()-started,
                device=adapter.device, gpu=torch.cuda.get_device_name() if adapter.device.startswith("cuda") else adapter.device))
            atomic_json(args.output/"results.json", records)
            del adapter, state
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
        atomic_json(args.output/"COMPLETED.json", dict(records=records, prepared_sha256=sha256(args.prepared),
            audit_sha256=sha256(args.audit), generator_sha256=sha256(Path(__file__)), finished=time.time()))
    except Exception as error:
        atomic_json(args.output/"FAILED.json", dict(error=repr(error), time=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    prep = sub.add_parser("prepare")
    for name in ("manifest", "folds", "spec", "output"):
        prep.add_argument(f"--{name}", type=Path, required=True)
    run = sub.add_parser("infer")
    for name in ("prepared", "audit", "runs", "home", "output"):
        run.add_argument(f"--{name}", type=Path, required=True)
    run.add_argument("--model", choices=("edsr","imdn","safmn"), required=True)
    run.add_argument("--epoch", type=int, default=1500)
    run.add_argument("--device", default="cuda")
    args = parser.parse_args()
    prepare(args) if args.mode == "prepare" else infer(args)


if __name__ == "__main__":
    main()
