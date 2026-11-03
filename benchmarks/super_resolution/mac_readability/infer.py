"""Replay official and selected trained models on identical real Mac inputs."""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import cv2
import numpy as np
from PIL import Image
import torch

from isharescreen.enhancement import models


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    if "GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a" not in gpu:
        raise RuntimeError("Unexpected GPU host")
    owners = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    if any(x.strip().isdigit() and int(x) != os.getpid() for x in owners.splitlines()):
        raise RuntimeError("GPU occupied")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    source = json.loads((args.inputs / "INPUTS.json").read_text())
    result = dict(
        source_sha256=source["source_sha256"],
        input_manifest_sha256=models.sha256(args.inputs / "INPUTS.json"),
        script_sha256=models.sha256(Path(__file__)),
        host=socket.gethostname(),
        gpu=gpu,
        precision="amp-fp16",
        torch=torch.__version__,
        started=time.time(),
        models=[],
        images=[],
        complete=False,
        scope="Actual Mac VideoToolbox 420 decoded inputs; x2 then area resize to 1080p. Inference times are diagnostic, not a warm performance ranking.",
    )
    for key in ["edsr", "imdn"]:
        for trained in [False, True]:
            method = key + ("-trained" if trained else "-official")
            adapter = models.Upscaler(key, device="cuda:0", precision="amp-fp16")
            identity = dict(adapter.provenance, method=method)
            if trained:
                checkpoint = args.checkpoints / key / "best.pt"
                done = json.loads(checkpoint.with_name("COMPLETED.json").read_text())
                assert models.sha256(checkpoint) == done["best_sha256"]
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                assert (
                    payload["binding"] == done["binding"]
                    and payload["epoch"] == done["best_epoch"]
                )
                torch.nn.Module.load_state_dict(
                    adapter.model, payload["state_dict"], strict=True
                )
                identity.update(
                    trained_sha256=done["best_sha256"],
                    epoch=payload["epoch"],
                    binding=payload["binding"],
                )
            adapter.model.eval().requires_grad_(False)
            result["models"].append(identity)
            for row in source["images"]:
                path = args.inputs / row["input_path"]
                assert models.sha256(path) == row["decoded_sha256"]
                image = np.asarray(Image.open(path).convert("RGB"))
                assert image.shape == (720, 1280, 3)
                enhanced, timing = adapter(image)
                assert enhanced.shape == (1440, 2560, 3)
                display = cv2.resize(
                    enhanced, (1920, 1080), interpolation=cv2.INTER_AREA
                )
                filename = row["id"] + "-" + method + ".png"
                out = args.output / filename
                Image.fromarray(display).save(out)
                result["images"].append(
                    {
                        k: row[k]
                        for k in [
                            "group",
                            "state_id",
                            "config",
                            "source_age_s",
                            "height",
                            "codec_frame_count_valid",
                        ]
                    }
                    | dict(
                        id=row["id"] + "-" + method,
                        method=method,
                        path=filename,
                        sha256=models.sha256(out),
                        input_sha256=row["decoded_sha256"],
                        timing=timing,
                    )
                )
            (args.output / "RESULTS.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(method, len(source["images"]), "frames", flush=True)
            del adapter
            torch.cuda.empty_cache()
    result.update(complete=True, finished=time.time())
    (args.output / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
