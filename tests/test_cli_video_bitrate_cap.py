"""The opt-in bitrate trial reaches the media offer without changing defaults."""

from __future__ import annotations

import pytest
import sys
from types import SimpleNamespace

from isharescreen import cli
from isharescreen.proxy.protocol.media_config import BandwidthMode, MediaOfferConfig


def _config(monkeypatch, *flags):
    monkeypatch.setattr(cli, "_password_from_args", lambda _args: "test-only")
    args = cli._make_parser().parse_args(
        ["--host", "example.invalid", "-u", "user", "--no-keyring", *flags]
    )
    return cli._build_session_config(args)


def test_bitrate_cap_is_opt_in_and_preserves_environment(monkeypatch):
    monkeypatch.setenv("ISS_VIDEO_CODEC", "hevc")
    assert _config(monkeypatch).media_offer is None
    config = _config(monkeypatch, "--video-bitrate-cap-kbps", "1250")
    assert config.media_offer is not None
    assert config.media_offer.video.codec == "hevc"
    desktop = BandwidthMode.LOW_LATENCY_SCREEN_WIRED
    assert [
        entry.max_bandwidth_bps
        for entry in config.media_offer.bandwidth
        if (entry.configuration_extension or 0) & int(desktop)
    ] == [1_250_000]


@pytest.mark.parametrize("value", ["0", "-1", "4294968"])
def test_bitrate_cap_rejects_out_of_range(monkeypatch, value):
    with pytest.raises(SystemExit, match="video-bitrate-cap-kbps"):
        _config(monkeypatch, "--video-bitrate-cap-kbps", value)


@pytest.mark.parametrize("cap", [None, "600"])
def test_browser_codec_resolution_preserves_custom_offer_unless_cap_requested(
    monkeypatch, cap
):
    monkeypatch.delenv("ISS_VIDEO_CODEC", raising=False)
    flags = [] if cap is None else ["--video-bitrate-cap-kbps", cap]
    args = cli._make_parser().parse_args(["--frontend", "browser", *flags])
    original = MediaOfferConfig.from_environment()
    config = SimpleNamespace(media_offer=original)
    monkeypatch.setitem(
        sys.modules,
        "isharescreen.frontend.wt.server",
        SimpleNamespace(run=lambda *args, **kwargs: 0),
    )
    assert cli._run_frontend(config, args) == 0
    if cap is None:
        assert config.media_offer is original
    else:
        assert config.media_offer.video.codec == "avc"
        desktop = BandwidthMode.LOW_LATENCY_SCREEN_WIRED
        assert [
            entry.max_bandwidth_bps
            for entry in config.media_offer.bandwidth
            if (entry.configuration_extension or 0) & int(desktop)
        ] == [600_000]
