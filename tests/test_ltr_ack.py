"""The LTR ack must name a frame the decoder really holds."""
from __future__ import annotations

from isharescreen.proxy.media.hevc import HevcDecoder


def _decoder() -> HevcDecoder:
    return HevcDecoder(4, prefer_hwaccel=False)


def test_clean_decoded_frame_becomes_the_ack():
    d = _decoder()

    d._store_frame(0, object(), False, 41)

    assert d.last_clean_donl[0] == 41


def test_concealed_frame_is_never_acked():
    d = _decoder()
    d._store_frame(0, object(), False, 41)

    d._store_frame(0, object(), True, 42)

    assert d.last_clean_donl[0] == 41


def test_frame_decoded_on_a_broken_tile_is_never_acked():
    d = _decoder()
    d._store_frame(0, object(), False, 41)
    d._gate.mark_decode_error(0)

    d._store_frame(0, object(), False, 42)

    assert d.last_clean_donl[0] == 41


def test_received_but_undecoded_frame_is_never_acked():
    d = _decoder()
    # A P-slice fed before any IDR is dropped by the keyframe gate.
    d._decode_one(b"\x02\x01\x80" + b"\x00" * 8, 0, 77)

    assert d.last_clean_donl[0] is None
    assert not d._pts_donl
