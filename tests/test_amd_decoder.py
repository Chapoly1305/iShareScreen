from types import SimpleNamespace
import pytest
from isharescreen import cli
from isharescreen.proxy.media import registry
from isharescreen.proxy.media.hevc import HevcDecoder
from isharescreen.proxy.protocol.media_config import MediaOfferConfig, PixelFormat


def test_amd_selection_requests_hevc420(monkeypatch):
    for key in ("ISS_DECODER", "ISS_VIDEO_CODEC", "ISS_VIDEO_CHROMA"):
        monkeypatch.setenv(key, "auto")
    args = cli._make_parser().parse_args(["--frontend", "desktop", "--decoder", "amd-hevc420"])
    cli._apply_video_overrides(args)
    offer = MediaOfferConfig.from_environment()
    assert offer.video.codec == "hevc"
    assert offer.video.pixel_formats == PixelFormat.YUV420_8_FULL | PixelFormat.YUV420_8_VIDEO


def test_explicit_amd_failure_does_not_select_other_gpu(monkeypatch):
    def fail(*a, **kw): raise RuntimeError("AMD failed")
    amd = SimpleNamespace(name="amd-hevc420", build=fail)
    other = SimpleNamespace(name="other", available=lambda: True,
                            build=lambda *a, **kw: pytest.fail("silent fallback"))
    monkeypatch.setattr(registry, "select", lambda *a, **kw: amd)
    monkeypatch.setattr(registry, "candidates", lambda *a: [other])
    assert registry.build_best("hevc", override="amd-hevc420", num_tiles=4) == (None, None)


def test_amd_decoder_disallows_internal_software_fallback():
    decoder = HevcDecoder(4, amd_only=True)
    with pytest.raises(RuntimeError, match="AMD hardware"):
        decoder._make_sw_context(b"")
