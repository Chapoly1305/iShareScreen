"""Construct QuickSRNet from its pinned architecture without reading weights."""

from pathlib import Path

from isharescreen.enhancement import candidates
from isharescreen.enhancement.candidates.quicksrnet import loader
from isharescreen.enhancement.models import sha256


def build(home: Path, seed: int):
    import torch

    # Verify the pinned definition files; a pretrained checkpoint is neither
    # required nor deserialized for a from-scratch experiment.
    spec = candidates.specification("quicksrnet")
    source = candidates.asset_home("quicksrnet", home)
    for name, asset in spec["files"].items():
        if not name.startswith("source/"):
            continue
        path = source / name
        if not path.is_file() or path.is_symlink() or sha256(path) != asset["sha256"]:
            raise ValueError("Missing or changed pinned scratch architecture: " + name)
    torch.manual_seed(seed)
    model = loader.build(source / "source").train().requires_grad_(True)
    return model, dict(
        kind="from_scratch",
        seed=seed,
        pretrained_weights_loaded=False,
        architecture="QuickSRNetSmall x2",
        initialization="Seeded PyTorch random parameters plus the official constructor identity initialization",
        source_revision=spec["revision"],
        source_hashes={
            n: v["sha256"] for n, v in spec["files"].items() if n.startswith("source/")
        },
    )
