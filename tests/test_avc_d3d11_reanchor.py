"""Preventive D3D11VA re-anchors never rebuild the decoder context."""
from __future__ import annotations

import types

from isharescreen.proxy.session import (
    Session,
    _AVC_D3D11_REANCHOR_FRAMES,
)


def _session(decoder):
    session = Session.__new__(Session)
    session._connected = True
    session._video_codec = "avc"
    session._decoder = decoder
    session._observed_tile_count = 1
    session._last_avc_reanchor_t = 0.0
    return session


def test_d3d11va_requests_intra_without_decoder_reset_before_wrap():
    firs: list[tuple[int, bool, bool]] = []
    decoder = types.SimpleNamespace(
        hw_accel="d3d11va",
        recovery_diagnostics={
            "frames_since_keyframe": _AVC_D3D11_REANCHOR_FRAMES,
            "reference_reset_pending": False,
        },
    )
    session = _session(decoder)
    session._send_fir_for_tile = (
        lambda tile, log_per_tile, record_grayout:
        firs.append((tile, log_per_tile, record_grayout)) or True
    )

    session._maybe_reanchor_d3d11va_avc()

    assert firs == [(0, False, False)]
    assert decoder.recovery_diagnostics["reference_reset_pending"] is False


def test_reanchor_ignores_software_and_pre_threshold_hardware():
    for hw_accel, frames in (
        (None, _AVC_D3D11_REANCHOR_FRAMES),
        ("d3d11va", _AVC_D3D11_REANCHOR_FRAMES - 1),
    ):
        firs: list[int] = []
        decoder = types.SimpleNamespace(
            hw_accel=hw_accel,
            recovery_diagnostics={
                "frames_since_keyframe": frames,
                "reference_reset_pending": False,
            },
        )
        session = _session(decoder)
        session._send_fir_for_tile = lambda tile, **_kwargs: firs.append(tile)

        session._maybe_reanchor_d3d11va_avc()

        assert firs == []


def test_reanchor_does_not_fire_during_an_inflight_reset():
    firs: list[int] = []
    decoder = types.SimpleNamespace(
        hw_accel="d3d11va",
        recovery_diagnostics={
            "frames_since_keyframe": _AVC_D3D11_REANCHOR_FRAMES + 50,
            "reference_reset_pending": True,
        },
    )
    session = _session(decoder)
    session._send_fir_for_tile = lambda tile, **_kwargs: firs.append(tile)

    session._maybe_reanchor_d3d11va_avc()

    assert firs == []
