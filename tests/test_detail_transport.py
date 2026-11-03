import hashlib

import numpy as np
import pytest

from benchmarks.super_resolution.mac_readability.detail_transport import (
    Receiver,
    frame_tiles,
    planes_nv12,
)


def test_420_roundtrip_boundaries_and_cached_identical_frame():
    width, height = 520, 300
    raw = (
        np.random.default_rng(1)
        .integers(0, 256, width * height * 3 // 2, dtype=np.uint8)
        .tobytes()
    )
    receiver = Receiver(width, height)
    receiver.begin(1, hashlib.sha256(raw).digest()[:16])
    tiles = frame_tiles(raw, width, height, 1)
    assert len(tiles) == 6
    for item in reversed(tiles):
        assert receiver.accept(item["packet"])
    assert receiver.mask.all()
    for a, b in zip(receiver.planes, planes_nv12(raw, width, height)):
        np.testing.assert_array_equal(a, b)
    assert frame_tiles(raw, width, height, 2, previous_raw=raw) == []


def test_new_revision_invalidates_all_old_refinement_and_rejects_inflight():
    raw = bytes([83]) * (256 * 256 * 3 // 2)
    identity = hashlib.sha256(raw).digest()[:16]
    packet = frame_tiles(raw, 256, 256, 1)[0]["packet"]
    receiver = Receiver(256, 256)
    receiver.begin(1, identity)
    assert receiver.accept(packet) and receiver.mask.all()
    receiver.begin(2, identity)
    assert not receiver.mask.any() and not receiver.accept(packet)
    with pytest.raises(ValueError):
        receiver.begin(1, identity)


def test_corrupt_payload_cannot_partially_update_display():
    raw = bytes([83]) * (256 * 256 * 3 // 2)
    receiver = Receiver(256, 256)
    receiver.begin(1, hashlib.sha256(raw).digest()[:16])
    packet = frame_tiles(raw, 256, 256, 1)[0]["packet"]
    with pytest.raises(ValueError):
        receiver.accept(packet[:-3])
    assert not receiver.mask.any() and not receiver.planes[0].any()
