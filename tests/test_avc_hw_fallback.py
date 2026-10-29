"""Truthful AVC hardware fallback and restart behavior."""
from __future__ import annotations

import types

from isharescreen.proxy.media.avc import AvcDecoder
from isharescreen.proxy.session import Session


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


def test_d3d11_reference_failure_switches_only_the_recovery_context(monkeypatch):
    decoder = AvcDecoder(1)
    decoder._hw_name = "d3d11va"
    decoder._sps = b"sps"
    decoder._pps = b"pps"
    attempted: list[str] = []
    software = object()

    decoder.mark_hwaccel_reference_failure(
        "reference picture missing during reorder",
    )

    # Keep the event diagnostic truthful until the fresh-intra path closes the
    # failed context. Only the replacement context changes to software.
    assert decoder.hw_accel == "d3d11va"
    assert decoder._hw_failed is True

    monkeypatch.setattr(
        decoder,
        "_try_hwaccel_locked",
        lambda name, _extra: attempted.append(name),
    )
    monkeypatch.setattr(decoder, "_make_sw_context", lambda _extra: software)
    decoder._codec = None

    with decoder._codec_lock:
        codec = decoder._ensure_codec_locked()

    assert codec is software
    assert attempted == []
    assert decoder.hw_accel is None


def test_non_d3d11_reference_failure_does_not_disable_hardware():
    decoder = AvcDecoder(1)
    decoder._hw_name = "videotoolbox"

    decoder.mark_hwaccel_reference_failure("reference picture missing")

    assert decoder.hw_accel == "videotoolbox"
    assert decoder._hw_failed is False


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
