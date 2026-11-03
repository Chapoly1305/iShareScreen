"""Official initialization must retain weights, enable updates and isolate data."""

import numpy as np
import pytest
import torch

from benchmarks.super_resolution.text_corpus.train import (
    initialize,
    eligible_indices,
    state_identity,
)


def test_official_weights_are_retained_then_receive_actual_updates(
    monkeypatch, tmp_path
):
    from isharescreen.enhancement import candidates

    reference = torch.nn.Conv2d(3, 3, 1).eval().requires_grad_(False)
    identity = state_identity(reference)
    spec = dict(
        revision="pinned-official",
        files={"weights.pth": dict(sha256="official-checkpoint")},
    )
    monkeypatch.setattr(candidates, "load_network", lambda key, home: (reference, spec))
    model, initialization = initialize(tmp_path, 682, "official")
    assert (
        state_identity(model) == identity
        and initialization["pretrained_weights_loaded"]
    )
    assert model.training and all(p.requires_grad for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss = (model(torch.rand(1, 3, 8, 8)) - 0.5).square().mean()
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    optimizer.step()
    assert state_identity(model) != identity


def test_exclusion_applies_to_every_split_without_reassigning_sources():
    data = dict(
        rows=[
            dict(split=s, requested_bps=b)
            for s in ["train", "validation", "test"]
            for b in [128000, 600000, 1000000]
        ]
    )
    for split, expected in [
        ("train", [1, 2]),
        ("validation", [4, 5]),
        ("test", [7, 8]),
    ]:
        assert np.array_equal(eligible_indices(data, split, {128000}), expected)


def test_normalized_official_weights_match_and_tampering_is_rejected(
    monkeypatch, tmp_path
):
    from benchmarks.super_resolution.text_corpus import official
    from benchmarks.super_resolution.training_state import atomic_json
    from isharescreen.enhancement.models import sha256

    reference = torch.nn.Conv2d(3, 3, 1)
    torch.save(reference.state_dict(), tmp_path / "official-state.pt")
    spec = dict(files={"weights.pth": dict(sha256="pinned-original")})
    atomic_json(
        tmp_path / "official-state.json",
        dict(
            original_sha256="pinned-original",
            normalized_sha256=sha256(tmp_path / "official-state.pt"),
            tensor_sha256=state_identity(reference),
        ),
    )
    monkeypatch.setattr(official.candidates, "verify_assets", lambda key, home: spec)
    monkeypatch.setattr(official.candidates, "asset_home", lambda key, home: tmp_path)
    monkeypatch.setattr(official.loader, "build", lambda path: torch.nn.Conv2d(3, 3, 1))
    loaded, _ = official.load(tmp_path)
    assert state_identity(loaded) == state_identity(reference)
    assert not loaded.training and not any(p.requires_grad for p in loaded.parameters())
    with (tmp_path / "official-state.pt").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="identity differs"):
        official.load(tmp_path)
