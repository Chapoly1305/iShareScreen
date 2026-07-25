"""AVC loss recovery must reseed an empty DPB on Apple's non-IDR I frame."""
from __future__ import annotations

import types

from isharescreen.proxy.media.avc import AvcDecoder


# Minimal type-1 slice headers:
#   first_mb_in_slice = ue(0)
#   slice_type = ue(0) for P, ue(2) for I
_P_SLICE = b"\x61\xc0"
_I_SLICE = b"\x61\xb0"


class _FakeCodec:
    def __init__(self) -> None:
        self.decode_calls = 0
        self.closed = False

    def decode(self, pkt):
        self.decode_calls += 1
        return [types.SimpleNamespace(pts=pkt.pts)]

    def close(self) -> None:
        self.closed = True


def test_reference_break_drops_deltas_then_rebuilds_on_intra():
    decoder = AvcDecoder(1, prefer_hwaccel=False)
    old_codec = _FakeCodec()
    made: list[_FakeCodec] = []
    decoder._codec = old_codec
    decoder._await_key = False
    decoder._dpb_ready = True
    decoder._gate.mark_decode_error(0)

    def ensure_codec():
        if decoder._codec is None:
            decoder._codec = _FakeCodec()
            made.append(decoder._codec)
        return decoder._codec

    decoder._ensure_codec_locked = ensure_codec
    decoder.mark_reference_chain_broken("reference picture missing during reorder")

    decoder.feed_nalu(_P_SLICE, 0)

    assert old_codec.decode_calls == 0
    assert decoder.recovery_diagnostics["reference_reset_pending"] is True
    assert decoder.recovery_diagnostics["await_key"] is True

    decoder.feed_nalu(_I_SLICE, 0)

    assert old_codec.closed is True
    assert len(made) == 1
    assert made[0].decode_calls == 1
    assert decoder.recovery_diagnostics["reference_reset_pending"] is False
    assert decoder.recovery_diagnostics["reference_resets"] == 1
    assert decoder.recovery_diagnostics["await_key"] is False
    # The codec-only reset must preserve sticky FIR/recovery state until the
    # newly seeded decoder's frame is consumed and confirmed clean.
    assert decoder._gate._keyframe_required == {0}


def test_break_reported_inside_decode_does_not_publish_concealed_frame():
    decoder = AvcDecoder(1, prefer_hwaccel=False)
    published: list[int] = []
    decoder._on_frame_published = published.append
    decoder._await_key = False
    decoder._dpb_ready = True

    class _BreakingCodec(_FakeCodec):
        def decode(self, pkt):
            self.decode_calls += 1
            decoder.mark_reference_chain_broken(
                "reference picture missing during reorder",
            )
            return [types.SimpleNamespace(pts=pkt.pts)]

    codec = _BreakingCodec()
    decoder._codec = codec
    decoder.feed_nalu(_P_SLICE, 0)

    assert codec.decode_calls == 1
    assert published == []
    assert decoder._tiles[0].good_count == 0
    assert decoder.recovery_diagnostics["reference_reset_pending"] is True
    assert decoder.recovery_diagnostics["await_key"] is True
