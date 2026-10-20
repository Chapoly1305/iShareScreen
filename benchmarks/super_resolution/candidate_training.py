"""Training-only construction for the lightweight super-resolution candidates.

Inference loaders intentionally return deployed graphs.  This module keeps the
initialization decision explicit when a public checkpoint is native x4 but the
desktop data contract is x2.  It never relabels those x4 checkpoints as x2.
"""
from __future__ import annotations

from pathlib import Path

from isharescreen.enhancement import candidates
from isharescreen.enhancement.models import sha256


IDS = candidates.IDS


def source_hashes(key: str) -> dict:
    if key not in IDS:
        raise ValueError(f"Unknown candidate: {key}")
    root = Path(candidates.__file__).parent
    return {
        "candidate_training.py": sha256(Path(__file__)),
        "shared_train.py": sha256(Path(__file__).with_name("train.py")),
        "candidate_registry.py": sha256(root / "__init__.py"),
        "candidate_loader.py": sha256(root / key / "loader.py"),
        "candidate_spec.py": sha256(root / key / "model.json"),
    }


def _phase_groups(scale_from: int, scale_to: int, colors: int = 3):
    if scale_from % scale_to:
        raise ValueError("Scale conversion must partition source subpixels")
    ratio = scale_from // scale_to
    groups = []
    for color in range(colors):
        for row in range(scale_to):
            for col in range(scale_to):
                groups.append([color * scale_from**2 + (row*ratio + dy)*scale_from + col*ratio + dx
                               for dy in range(ratio) for dx in range(ratio)])
    return groups


def _reduce_output(weight, groups):
    import torch
    return torch.stack([weight[index].mean(0) for index in groups])


def _transfer_casr_x4_to_x2(model, official):
    """Transfer all compatible deployed CASR weights and average only x4 phases."""
    state, source, groups = model.state_dict(), official.state_dict(), _phase_groups(4, 2)
    copied = []
    for name, value in state.items():
        if name in source and source[name].shape == value.shape:
            state[name] = source[name].detach().clone(); copied.append(name); continue
        if name in ("input_conv2.conv3x3.weight", "input_conv2.conv3x3.bias"):
            state[name] = _reduce_output(source[name], groups); copied.append(name); continue
        raise ValueError(f"No deterministic x4-to-x2 CASR transfer for {name}")
    model.load_state_dict(state, strict=True)
    return {"kind": "x4_to_x2_phase_average", "source_scale": 4, "target_scale": 2,
            "copied_tensors": copied}


def _span_channel_groups():
    return _phase_groups(4, 2) + [[48 + i] for i in range(32)]


def _transfer_span_x4_to_x2(model, official):
    """Transfer SPANV2 backbone exactly and its scale-dependent paths by phase average."""
    import torch
    state, source, phases, channels = model.state_dict(), official.state_dict(), _phase_groups(4, 2), _span_channel_groups()
    copied = []
    for name, value in state.items():
        if name in source and source[name].shape == value.shape:
            state[name] = source[name].detach().clone(); copied.append(name); continue
        if name == "conv_near.weight":
            state[name] = _reduce_output(source[name], phases); copied.append(name); continue
        if name == "depthwise_conv.weight":
            state[name] = torch.stack([source[name][index].mean(0) for index in channels]); copied.append(name); continue
        if name == "pointwise_conv.weight":
            rows = []
            for outputs in phases:
                rows.append(torch.stack([source[name][outputs][:, inputs].mean((0, 1)) for inputs in channels]))
            state[name] = torch.stack(rows); copied.append(name); continue
        raise ValueError(f"No deterministic x4-to-x2 SPANV2 transfer for {name}")
    model.load_state_dict(state, strict=True)
    return {"kind": "x4_to_x2_phase_average", "source_scale": 4, "target_scale": 2,
            "copied_tensors": copied}


def load(key: str, home: Path):
    """Return a differentiable x2 model plus auditable initialization metadata."""
    import torch
    spec = candidates.verify_assets(key, home)
    root = candidates.asset_home(key, home)
    if spec["scale"] == 2:
        model, _ = candidates.load_network(key, home)
        return model.train().requires_grad_(True), dict(kind="strict_official_x2",
            source_scale=2, target_scale=2, checkpoint_sha256=spec["files"]["weights.pth"]["sha256"])
    loader = __import__(f"isharescreen.enhancement.candidates.{key}.loader", fromlist=["build"])
    target = loader.build(root / "source", scale=2)
    official, _ = candidates.load_network(key, home)
    if key == "casr":
        migration = _transfer_casr_x4_to_x2(target, official)
    elif key == "spanv2":
        migration = _transfer_span_x4_to_x2(target, official)
    else:
        raise ValueError(f"Unsupported native scale conversion: {key}")
    with torch.no_grad():
        output = target(torch.zeros(1, 3, 18, 24))
    if output.shape != (1, 3, 36, 48):
        raise ValueError("Converted candidate does not implement x2 geometry")
    migration["checkpoint_sha256"] = spec["files"]["weights.pth"]["sha256"]
    return target.train().requires_grad_(True), migration


def predict(model, key: str, tensor):
    """Differentiable [0,1] RGB forward pass, including PlainUSR's safe variant."""
    if key == "casr":
        # CASR's PixelUnshuffle(3) requires divisible spatial dimensions;
        # preserve the shared 64px crop recipe by reflect-padding only its
        # temporary boundary and cropping the corresponding x2 output.
        import torch.nn.functional as F
        height, width = tensor.shape[-2:]
        pad_h, pad_w = (-height) % 3, (-width) % 3
        output = model(F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect"))
        return output[..., :height*2, :width*2]
    if key != "plainusr":
        return model(tensor)
    # The public deployment forward mutates views in place.  It is correct for
    # inference but invalidates autograd saved tensors.  These concatenations
    # implement its same sequential feature replacement without mutation.
    features = model.head(tensor)
    features = __import__("torch").cat((model.block1(features[:, :model.im_feat[0]]),
                                         features[:, model.im_feat[0]:]), dim=1)
    features = __import__("torch").cat((model.blockm(features[:, :model.im_feat[1]]),
                                         features[:, model.im_feat[1]:]), dim=1)
    features = __import__("torch").cat((model.block2(features[:, :model.im_feat[0]]),
                                         features[:, model.im_feat[0]:]), dim=1)
    return model.tail(features)
