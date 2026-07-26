"""Preventive D3D11VA DPB resets stay narrow and frame-count driven."""
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
    return session


def test_d3d11va_reanchors_before_picture_order_wrap():
    marked: list[str] = []
    firs: list[object] = []
    decoder = types.SimpleNamespace(
        hw_accel="d3d11va",
        recovery_diagnostics={
            "frames_since_context_reset": _AVC_D3D11_REANCHOR_FRAMES,
            "reference_reset_pending": False,
        },
        mark_reference_chain_broken=marked.append,
    )
    session = _session(decoder)
    session.request_fir = firs.append

    session._maybe_reanchor_d3d11va_avc()

    assert len(marked) == 1
    assert "POC wrap" in marked[0]
    assert firs == [None]


def test_reanchor_ignores_software_and_pre_threshold_hardware():
    for hw_accel, frames in (
        (None, _AVC_D3D11_REANCHOR_FRAMES),
        ("d3d11va", _AVC_D3D11_REANCHOR_FRAMES - 1),
    ):
        marked: list[str] = []
        decoder = types.SimpleNamespace(
            hw_accel=hw_accel,
            recovery_diagnostics={
                "frames_since_context_reset": frames,
                "reference_reset_pending": False,
            },
            mark_reference_chain_broken=marked.append,
        )
        session = _session(decoder)
        session.request_fir = lambda _tile: None

        session._maybe_reanchor_d3d11va_avc()

        assert marked == []


def test_reanchor_does_not_duplicate_an_inflight_reset():
    marked: list[str] = []
    decoder = types.SimpleNamespace(
        hw_accel="d3d11va",
        recovery_diagnostics={
            "frames_since_context_reset": _AVC_D3D11_REANCHOR_FRAMES + 50,
            "reference_reset_pending": True,
        },
        mark_reference_chain_broken=marked.append,
    )
    session = _session(decoder)
    session.request_fir = lambda _tile: None

    session._maybe_reanchor_d3d11va_avc()

    assert marked == []
