import numpy as np
import pytest

from benchmarks.super_resolution.mac_readability.component_codec import encode, decode


def test_repeated_patterns_deduplicate_only_identical_pixels():
    y = np.full((64, 128), 235, np.uint8)
    shape = np.array([[16, 235, 16], [16, 16, 16], [16, 235, 16]], np.uint8)
    for x in [4, 20, 40]:
        y[8:11, x : x + 3] = shape
    y[8:11, 60:63] = shape
    y[8, 60] = 17  # one-byte difference must not alias
    raw = y.tobytes() + bytes([128]) * (64 * 128 // 2)
    packet, info = encode(raw, 128, 64)
    assert (
        info["components"] == 4
        and info["unique_patterns"] == 2
        and info["exact_reused_instances"] == 2
    )
    assert decode(packet) == raw
    with pytest.raises(ValueError):
        decode(packet[:-1])


@pytest.mark.parametrize("constant", [False, True])
def test_arbitrary_color420_and_background_roundtrip(constant):
    raw = (
        bytes([0]) * (128 * 64 * 3 // 2)
        if constant
        else np.random.default_rng(2)
        .integers(0, 256, 128 * 64 * 3 // 2, dtype=np.uint8)
        .tobytes()
    )
    packet, _ = encode(raw, 128, 64)
    assert decode(packet) == raw
