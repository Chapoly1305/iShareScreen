"""Small reference and evidence helpers; independent of training experiments."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stats(values):
    return dict(
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        mean_ms=float(np.mean(values)),
        samples_ms=values,
    )


def reference_resize(native, *, quantize_first=False):
    """Independent CPU 1440-to-1080 reference; input HWC float RGB in 0..255 units.

    Values may exceed that range. Preserve them until after resizing unless
    explicitly reproducing the historical RGB8-first output protocol.
    """
    import cv2

    if quantize_first:
        native = np.clip(np.rint(native), 0, 255).astype(np.uint8)
    resized = cv2.resize(native, (1920, 1080), interpolation=cv2.INTER_AREA)
    return np.clip(np.rint(resized), 0, 255).astype(np.uint8)


def guard():
    """Require the explicitly selected physical device and exclusive compute."""
    expected = os.environ.get("ISS_EXPECTED_GPU_UUID")
    if not expected:
        raise RuntimeError("Provide --expected-gpu-uuid before running GPU checks")
    devices = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"], text=True
    ).splitlines()
    if devices != [expected]:
        raise RuntimeError("This benchmark requires the selected single-GPU host")
    processes = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True,
    )
    if any(
        int(pid) != os.getpid()
        for pid in processes.splitlines()
        if pid.strip().isdigit()
    ):
        raise RuntimeError("Another GPU compute process is active")
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,temperature.gpu,power.draw,clocks.sm,memory.used",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()


class Engine:
    """Separate execution context for output equivalence, never used in timing."""

    def __init__(self, path):
        import torch
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        self.context = self.engine.create_execution_context()
        self.input = torch.empty(
            tuple(self.engine.get_tensor_shape("input")),
            device="cuda",
            dtype=torch.float32,
        )
        self.output = torch.empty(
            tuple(self.engine.get_tensor_shape("output")),
            device="cuda",
            dtype=torch.float32,
        )
        self.context.set_tensor_address("input", self.input.data_ptr())
        self.context.set_tensor_address("output", self.output.data_ptr())

    def run(self):
        import torch

        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT reference execution failed")
        return self.output


def reference_rgb(nv12):
    """Independent eager BT.709 full-range conversion and RGB8 truncation."""
    import torch

    y = nv12[:720].float()
    uv = (
        nv12[720:]
        .reshape(360, 640, 2)
        .float()
        .repeat_interleave(2, 0)
        .repeat_interleave(2, 1)
    )
    u, v = uv[:, :, 0] - 128, uv[:, :, 1] - 128
    kg = 1 - 0.2126 - 0.0722
    return (
        torch.stack(
            (
                y + 1.5748 * v,
                y
                - (2 * 0.0722 * (1 - 0.0722) / kg) * u
                - (2 * 0.2126 * (1 - 0.2126) / kg) * v,
                y + 1.8556 * u,
            )
        )
        .clamp(0, 255)
        .byte()
    )
