import hashlib

from benchmarks.super_resolution.mac_readability.component_codec import encode
from benchmarks.super_resolution.mac_readability.component_receiver import (
    ComponentReceiver,
)


def test_stale_supplement_is_rejected_before_and_during_decode(monkeypatch):
    raw = bytes([128]) * (64 * 64 * 3 // 2)
    packet, _ = encode(raw, 64, 64)
    identity = hashlib.sha256(raw).digest()
    gate = ComponentReceiver()
    gate.begin(1, identity)
    assert gate.accept(1, packet) == raw
    gate.begin(2, identity)
    assert gate.latest is None and gate.accept(1, packet) is None

    def slow_decode(_):
        gate.begin(3, identity)
        return raw

    monkeypatch.setattr(
        "benchmarks.super_resolution.mac_readability.component_receiver.decode",
        slow_decode,
    )
    assert gate.accept(2, packet) is None and gate.latest is None
