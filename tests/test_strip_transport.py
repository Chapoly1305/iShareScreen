import hashlib

import numpy as np
import pytest

from benchmarks.super_resolution.mac_readability.strip_transport import (
    StripReceiver,
    cuts_from_luma,
    frame_strips,
    planes_nv12,
)


def test_source_gaps_protect_character_rows_and_roundtrip():
    width, height = 1920, 1080
    y = np.full((height, width), 235, np.uint8)
    # Synthetic repeated glyph strokes with known horizontal blank gaps.
    for top in range(12, height - 20, 36):
        y[top : top + 18, 20:1800:3] = 16
    raw = y.tobytes() + bytes([128]) * (width * height // 2)
    cuts = cuts_from_luma(y)
    assert all(v % 2 == 0 for v in cuts)
    for cut in cuts[1:-1]:
        assert np.all(y[cut - 2 : cut + 2, :] == 235)
    r = StripReceiver(width, height)
    identity = hashlib.sha256(raw).digest()[:16]
    r.begin(1, identity)
    packets = frame_strips(raw, width, height, 1)
    for item in reversed(packets):
        assert r.accept(item["packet"])
    assert r.mask.all()
    for a, b in zip(r.planes, planes_nv12(raw, width, height)):
        np.testing.assert_array_equal(a, b)
    r.begin(2, identity)
    assert not r.mask.any() and not r.accept(packets[0]["packet"])


def test_random_420_strip_roundtrip_and_atomic_truncation():
    width, height = 520, 300
    raw = (
        np.random.default_rng(29)
        .integers(0, 256, width * height * 3 // 2, dtype=np.uint8)
        .tobytes()
    )
    packets = frame_strips(raw, width, height, 1)
    r = StripReceiver(width, height)
    r.begin(1, hashlib.sha256(raw).digest()[:16])
    with pytest.raises(ValueError):
        r.accept(packets[0]["packet"][:-1])
    assert not r.mask.any() and not r.planes[0].any()
    for item in packets:
        assert r.accept(item["packet"])
    for a, b in zip(r.planes, planes_nv12(raw, width, height)):
        np.testing.assert_array_equal(a, b)
