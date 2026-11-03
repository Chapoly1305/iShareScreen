"""Tensor-identical official weights across PyTorch checkpoint readers.

Some newer weights-only readers reject optimizer metadata in the vendor's
checkpoint. Normalize once with the existing verified official loader, retain
only tensors, and bind both original file identity and every resulting tensor.
No unrestricted pickle loading or changes to production loaders are needed.
"""

import argparse
import json
from pathlib import Path

import torch

from benchmarks.super_resolution.training_state import (
    atomic_json,
    atomic_weights,
    cpu_state,
)
from isharescreen.enhancement import candidates
from isharescreen.enhancement.candidates.quicksrnet import loader
from isharescreen.enhancement.models import project_home, sha256


def load(home):
    from benchmarks.super_resolution.text_corpus.train import state_identity

    path = home / "official-state.pt"
    receipt = home / "official-state.json"
    if not path.exists() and not receipt.exists():
        return candidates.load_network("quicksrnet", home)
    spec = candidates.verify_assets("quicksrnet", home)
    metadata = json.loads(receipt.read_text())
    if metadata["original_sha256"] != spec["files"]["weights.pth"][
        "sha256"
    ] or metadata["normalized_sha256"] != sha256(path):
        raise ValueError("Official tensor conversion identity differs")
    model = loader.build(candidates.asset_home("quicksrnet", home) / "source")
    model.load_state_dict(
        torch.load(path, map_location="cpu", weights_only=True), strict=True
    )
    if state_identity(model) != metadata["tensor_sha256"]:
        raise ValueError("Official parameters differ after conversion")
    return model.eval().requires_grad_(False), spec


def normalize(output):
    from benchmarks.super_resolution.text_corpus.train import state_identity

    output.mkdir(parents=True, exist_ok=True)
    if (output / "official-state.pt").exists():
        raise ValueError("Conversion already exists")
    model, spec = candidates.load_network("quicksrnet", project_home())
    atomic_weights(output / "official-state.pt", cpu_state(model))
    atomic_json(
        output / "official-state.json",
        dict(
            original_sha256=spec["files"]["weights.pth"]["sha256"],
            normalized_sha256=sha256(output / "official-state.pt"),
            tensor_sha256=state_identity(model),
            torch=str(torch.__version__),
            operation="Verified official loader; save only the unchanged strict model state_dict; discard optimizer metadata.",
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    normalize(p.parse_args().output)
