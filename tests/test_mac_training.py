"""Meaningful geometry and data-role checks before the expensive GPU queue."""

from pathlib import Path

import cv2
import numpy as np
import torch

from benchmarks.super_resolution.mac_readability.train_ablations import (
    area_4_to_3,
    configure_parameters,
    PairCache,
)


def test_exact_display_area_geometry_and_gradients():
    rng = np.random.default_rng(17)
    pixels = rng.random((16, 24, 3)).astype(np.float32)
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())[None].requires_grad_()
    actual = area_4_to_3(tensor)
    expected = cv2.resize(pixels, (18, 12), interpolation=cv2.INTER_AREA)
    np.testing.assert_allclose(
        actual.detach().numpy()[0].transpose(1, 2, 0), expected, atol=2e-7
    )
    actual.square().mean().backward()
    assert torch.isfinite(tensor.grad).all() and tensor.grad.abs().sum() > 0


def test_conservative_arm_leaves_trunk_and_normalization_fixed():
    model = torch.nn.Module()
    for name in ["head", "body", "tail", "sub_mean", "add_mean"]:
        setattr(model, name, torch.nn.Linear(2, 2))
    chosen = configure_parameters(model, "glyph_conservative")
    assert {id(p) for p in chosen} == {id(p) for p in model.tail.parameters()}
    chosen = configure_parameters(model, "native_glyph_mix")
    assert all(
        p.requires_grad
        for n, p in model.named_parameters()
        if n.startswith(("head.", "body.", "tail."))
    )
    assert not any(
        p.requires_grad
        for n, p in model.named_parameters()
        if n.startswith(("sub_mean.", "add_mean."))
    )


def test_sampler_is_reproducible_and_supports_native_glyph_ids():
    cache = PairCache.__new__(PairCache)
    # RGB coordinate grids make any misalignment between LR and HR visible.
    low = np.zeros((720, 1280, 3), np.uint8)
    high = np.zeros((1080, 1920, 3), np.uint8)
    low[:, :, 0] = np.arange(1280)[None, :] % 128
    high[:, :, 0] = (np.arange(1920)[None, :] * 2 // 3) % 128
    low[:, :, 1] = np.arange(720)[:, None] % 128
    high[:, :, 1] = (np.arange(1080)[:, None] * 2 // 3) % 128
    cache.rows = [
        dict(
            id="train-glyph-000-12px",
            transform=dict(content_box=[100, 100, 1500, 900], line_boxes=[]),
        )
    ]
    cache.pairs = [(low, high)]
    first = cache.batch("native_glyph_mix", 41, size=2)
    second = cache.batch("native_glyph_mix", 41, size=2)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert first[2] == ["train-glyph-000-12px"] * 2
    assert torch.equal(first[0][:, :, 0, 0], first[1][:, :, 0, 0])
    assert first[0].shape == (2, 3, 128, 128) and first[1].shape == (2, 3, 192, 192)
