"""Automatic hardware HEVC 4:2:0 selection (concept from upstream PR #31).

Auto prefers hardware HEVC 4:4:4, then hardware HEVC 4:2:0, then H.264. The
4:2:0 rung is gated on a real hardware decode probe, never on software
fallback, and 4:2:0-only decoders never serve Apple's default 4:4:4 stream.
"""
from __future__ import annotations

import argparse

import pytest

from isharescreen.proxy.media import hwcaps as H, registry as R
from isharescreen.proxy.protocol.media_config import MediaOfferConfig, PixelFormat


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in ("ISS_DECODER", "ISS_VIDEO_CHROMA", "ISS_VIDEO_CODEC", "ISS_HEVC420", "ISS_HEVC444"):
        # setenv first so teardown restores the original (absent) state even
        # when code under test writes os.environ directly.
        monkeypatch.setenv(name, "")
        monkeypatch.delenv(name)
    monkeypatch.setattr(H, "_libav420_cache", {})
    monkeypatch.setattr(R.sys, "platform", "win32")
    monkeypatch.setattr(R, "_libav_hw", lambda: False)
    monkeypatch.setattr(R, "_qsv_hw", lambda: False)
    monkeypatch.setattr(R, "_libav_hw420", lambda: False)


def test_auto_prefers_444_then_420_then_avc(monkeypatch):
    monkeypatch.setattr(R, "_libav_hw", lambda: True)
    monkeypatch.setattr(R, "_libav_hw420", lambda: True)
    assert R.resolve_video_format("auto") == ("hevc", "444")

    monkeypatch.setattr(R, "_libav_hw", lambda: False)
    assert R.resolve_video_format("auto") == ("hevc", "420")
    assert R.resolve_codec("auto") == "hevc"

    monkeypatch.setattr(R, "_libav_hw420", lambda: False)
    assert R.resolve_video_format("auto") == ("avc", "420")


def test_explicit_chroma_restricts_auto(monkeypatch):
    monkeypatch.setattr(R, "_libav_hw", lambda: True)
    monkeypatch.setattr(R, "_libav_hw420", lambda: True)
    assert R.resolve_video_format("auto", chroma="420") == ("hevc", "420")
    monkeypatch.setenv("ISS_VIDEO_CHROMA", "420")
    assert R.resolve_video_format("auto") == ("hevc", "420")
    monkeypatch.setattr(R, "_libav_hw", lambda: False)
    assert R.resolve_video_format("auto", chroma="444") == ("avc", "420")


def test_explicit_420_decoder_determines_codec_and_chroma(monkeypatch):
    monkeypatch.setenv("ISS_DECODER", "libav-hevc420")
    assert R.resolve_video_format("auto") == ("hevc", "420")


def test_420_only_decoders_never_serve_the_444_stream(monkeypatch):
    monkeypatch.setattr(R, "_libav_hw420", lambda: True)
    names = [s.name for s in R.candidates("hevc")]
    assert "libav-hevc420" not in names and "libav-hevc420-sw" not in names
    assert R.select("hevc").name == "libav-hevc444-sw"


def test_420_ladder_order_and_software_floor(monkeypatch):
    monkeypatch.setattr(R, "_libav_hw420", lambda: True)
    names = [s.name for s in R.candidates("hevc", "420")]
    assert names == ["libav-hevc420", "libav-hevc420-sw"]   # AMD stays explicit-only
    assert R.select("hevc", chroma="420").name == "libav-hevc420"
    monkeypatch.setattr(R, "_libav_hw420", lambda: False)
    assert R.select("hevc", chroma="420").name == "libav-hevc420-sw"
    assert R.can_decode("hevc", "420", hardware_only=True) is False


def test_amd_override_still_honored(monkeypatch):
    assert R.select("hevc", override="amd-hevc420", chroma="420").name == "amd-hevc420"


def test_macos_videotoolbox_covers_420(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "darwin")
    monkeypatch.setattr(R, "_vt_available", lambda: True)
    assert R.select("hevc", chroma="420").name == "vt-hevc444"


def test_build_best_falls_back_to_software_420(monkeypatch):
    monkeypatch.setattr(R, "_libav_hw420", lambda: True)
    built = []

    def fail(*_a, **_k):
        raise RuntimeError("no surface")

    hw = next(s for s in R._REGISTRY if s.name == "libav-hevc420")
    sw = next(s for s in R._REGISTRY if s.name == "libav-hevc420-sw")
    monkeypatch.setattr(R, "_REGISTRY", [
        R.DecoderSpec(**{**hw.__dict__, "build": fail}),
        R.DecoderSpec(**{**sw.__dict__, "build": lambda n, **k: built.append(n) or "sw"}),
    ])
    spec, dec = R.build_best("hevc", override=None, num_tiles=4, chroma="420")
    assert (spec.name, dec, built) == ("libav-hevc420-sw", "sw", [4])


def test_probe_overrides_and_macos(monkeypatch):
    monkeypatch.setattr(H, "_probe_hevc420", lambda _h: pytest.fail("probe must not run"))
    monkeypatch.setenv("ISS_HEVC420", "0")
    assert H.libav_hevc420_hwdecode() is False
    monkeypatch.setattr(H, "_libav420_cache", {})
    monkeypatch.setenv("ISS_HEVC420", "1")
    assert H.libav_hevc420_hwdecode() is True
    monkeypatch.setattr(H, "_libav420_cache", {})
    monkeypatch.delenv("ISS_HEVC420")
    monkeypatch.setattr(H.sys, "platform", "darwin")
    assert H.libav_hevc420_hwdecode() is False


def test_probe_is_cached_and_tries_each_hwaccel(monkeypatch):
    monkeypatch.setattr(H.sys, "platform", "win32")
    monkeypatch.setattr(H, "available_hwaccels", lambda c: c)
    calls = []
    monkeypatch.setattr(H, "_probe_hevc420", lambda h: calls.append(h) or h == "d3d12va")
    assert H.libav_hevc420_hwdecode() is True
    assert H.libav_hevc420_hwdecode() is True
    assert calls == ["d3d11va", "d3d12va"]


def test_decoder_chroma_follows_offered_pixel_formats():
    config = MediaOfferConfig()
    assert config.video.decoder_chroma == "444"
    config.video.pixel_formats = PixelFormat.YUV420_8_FULL | PixelFormat.YUV420_8_VIDEO
    assert config.video.decoder_chroma == "420"
    config.video.codec = "avc"
    config.video.pixel_formats = PixelFormat.ALL
    assert config.video.decoder_chroma == "420"


def test_desktop_auto_offers_hevc420_when_only_420_hw(monkeypatch):
    import os
    import sys
    import types
    from isharescreen import cli

    monkeypatch.setattr(R, "_libav_hw420", lambda: True)
    # The real desktop app needs GLFW; only the codec/chroma decision matters.
    monkeypatch.setitem(sys.modules, "isharescreen.frontend.desktop.app",
                        types.SimpleNamespace(run=lambda *a, **k: 0))
    args = argparse.Namespace(frontend="desktop", codec=None, display=None, auto_quit_secs=None)
    assert cli._run_frontend(object(), args) == 0
    assert os.environ["ISS_VIDEO_CODEC"] == "hevc"
    assert os.environ["ISS_VIDEO_CHROMA"] == "420"
    assert MediaOfferConfig.from_environment().video.decoder_chroma == "420"
