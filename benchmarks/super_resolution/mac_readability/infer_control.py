"""Official or frozen best-trained EDSR, on an explicitly bound GPU."""

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
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint-sha256")
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True,
    ).strip()
    assert a.gpu_uuid in gpu and os.environ.get("CUDA_VISIBLE_DEVICES") == a.gpu_uuid
    owners = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    )
    assert not any(
        x.strip().isdigit() and int(x) != os.getpid() for x in owners.splitlines()
    ), "GPU occupied"
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    source = json.loads((a.inputs / "INPUTS.json").read_text())
    model = models.Upscaler("edsr", device="cuda:0", precision="amp-fp16")
    method = "edsr-official"
    if a.checkpoint:
        assert (
            a.checkpoint_sha256 and models.sha256(a.checkpoint) == a.checkpoint_sha256
        )
        assert (
            a.checkpoint_sha256
            == "01d99513d182f669e1e1e92c88e0e17a9a053b6e2d046a8733119047e61db4cd"
        )
        payload = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        torch.nn.Module.load_state_dict(model.model, payload["state_dict"], strict=True)
        method = "steady_l1-step001000"
    model.model.eval().requires_grad_(False)
    result = dict(
        source_sha256=source["source_sha256"],
        input_manifest_sha256=models.sha256(a.inputs / "INPUTS.json"),
        script_sha256=models.sha256(Path(__file__)),
        models_script_sha256=models.sha256(Path(models.__file__)),
        host=socket.gethostname(),
        gpu=gpu,
        precision="amp-fp16",
        torch=str(torch.__version__),
        model=model.provenance,
        scope=__doc__
        + " Same x2, uint8 quantization, then OpenCV area resize to1080 as original baseline.",
        started=time.time(),
        complete=False,
        images=[],
    )
    result["checkpoint_sha256"] = a.checkpoint_sha256 if a.checkpoint else None
    for row in source["images"]:
        path = a.inputs / row["input_path"]
        assert models.sha256(path) == row["decoded_sha256"]
        rgb = np.asarray(Image.open(path).convert("RGB"))
        assert rgb.shape == (720, 1280, 3)
        enhanced, timing = model(rgb)
        display = cv2.resize(enhanced, (1920, 1080), interpolation=cv2.INTER_AREA)
        name = row["id"] + "-" + method
        out = a.output / (name + ".png")
        Image.fromarray(display).save(out)
        result["images"].append(
            {
                k: row[k]
                for k in [
                    "group",
                    "category",
                    "state_id",
                    "config",
                    "height",
                    "source_age_s",
                ]
            }
            | dict(
                id=name,
                method=method,
                path=out.name,
                sha256=models.sha256(out),
                input_sha256=row["decoded_sha256"],
                timing=timing,
            )
        )
        print(name, flush=True)
    result.update(complete=True, finished=time.time())
    (a.output / "RESULTS.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
