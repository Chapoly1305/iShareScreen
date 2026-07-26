"""Truthful AVC hardware fallback and restart behavior."""
from __future__ import annotations

import types

from isharescreen.proxy.media.avc import AvcDecoder
from isharescreen.proxy.session import Session, _prefer_avc_hwaccel


def test_windows_avc_defaults_to_software():
    assert _prefer_avc_hwaccel("win32", None) is False


def test_windows_avc_hardware_requires_explicit_opt_in():
    assert _prefer_avc_hwaccel("win32", "1") is True
    assert _prefer_avc_hwaccel("win32", "true") is True
    assert _prefer_avc_hwaccel("win32", "0") is False


def test_non_windows_avc_retains_hardware_first_default():
    assert _prefer_avc_hwaccel("darwin", None) is True
    assert _prefer_avc_hwaccel("linux", None) is True
    assert _prefer_avc_hwaccel("linux", "off") is False


def test_explicit_hw_failure_relabels_decoder_as_software():
    decoder = AvcDecoder(1)
    decoder._hw_name = "d3d11va"

    decoder.mark_hwaccel_failed("Failed setup for format d3d11")

    assert decoder.hw_accel is None
    assert decoder._hw_failed is True


def test_known_failed_hwaccel_is_not_retried_on_context_rebuild(monkeypatch):
    decoder = AvcDecoder(1)
    decoder._sps = b"sps"
    decoder._pps = b"pps"
    decoder.mark_hwaccel_failed("Failed to get the decoder GUIDs")
    attempted: list[str] = []
    software = object()

    monkeypatch.setattr(
        decoder,
        "_try_hwaccel_locked",
        lambda name, _extra: attempted.append(name),
    )
    monkeypatch.setattr(
        decoder,
        "_make_sw_context",
        lambda _extra: software,
    )

    with decoder._codec_lock:
        codec = decoder._ensure_codec_locked()

    assert codec is software
    assert attempted == []
    assert decoder.hw_accel is None


def test_session_routes_late_hw_failure_to_active_decoder():
    failures: list[str] = []
    session = Session.__new__(Session)
    session._decoder = types.SimpleNamespace(
        mark_hwaccel_failed=failures.append,
    )

    session._on_libav_hwaccel_failure(
        "Failed setup for format d3d11: hwaccel initialisation returned error.",
    )

    assert failures == [
        "Failed setup for format d3d11: hwaccel initialisation returned error.",
    ]
