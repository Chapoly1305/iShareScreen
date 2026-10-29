import pytest

@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    for key in ("ISS_VIDEO_CODEC", "ISS_VIDEO_CHROMA", "ISS_DECODER"):
        monkeypatch.delenv(key, raising=False)

from isharescreen import cli
from isharescreen.proxy.protocol.media_config import MediaOfferConfig, PixelFormat


@pytest.mark.parametrize("codec,chroma,mask", [
    ("hevc", "444", PixelFormat.YUV444_8_FULL | PixelFormat.YUV444_8_VIDEO),
    ("hevc", "420", PixelFormat.YUV420_8_FULL | PixelFormat.YUV420_8_VIDEO),
    ("avc", "420", PixelFormat.YUV420_8_FULL | PixelFormat.YUV420_8_VIDEO),
])
def test_codec_and_chroma_reach_media_offer(monkeypatch, codec, chroma, mask):
    args = cli._make_parser().parse_args(["--frontend", "desktop", "--codec", codec,
                                         "--chroma", chroma, "--decoder", "auto"])
    cli._apply_video_overrides(args)
    offer = MediaOfferConfig.from_environment()
    offer.validate()
    assert offer.video.codec == codec
    assert offer.video.pixel_formats == mask
    assert offer.video.to_protobuf(1)


def test_auto_codec_with_444_chooses_hevc(monkeypatch):
    monkeypatch.delenv("ISS_VIDEO_CODEC", raising=False)
    args = cli._make_parser().parse_args(["--frontend", "desktop", "--codec", "auto",
                                         "--chroma", "444", "--decoder", "auto"])
    cli._apply_video_overrides(args)
    assert MediaOfferConfig.from_environment().video.codec == "hevc"


@pytest.mark.parametrize("flags", [
    ["--frontend", "desktop", "--codec", "avc"],
    ["--frontend", "browser", "--codec", "auto"],
    ["--frontend", "desktop", "--codec", "auto", "--decoder", "libav-avc420"],
])
def test_unsupported_444_combinations_rejected(monkeypatch, flags):
    monkeypatch.delenv("ISS_VIDEO_CHROMA", raising=False)
    args = cli._make_parser().parse_args([*flags, "--chroma", "444"])
    with pytest.raises(ValueError, match="4:4:4"):
        cli._apply_video_overrides(args)


def test_chroma_auto_preserves_default_offer(monkeypatch):
    monkeypatch.setenv("ISS_VIDEO_CHROMA", "444")
    args = cli._make_parser().parse_args(["--frontend", "desktop", "--codec", "auto",
                                         "--chroma", "auto", "--decoder", "auto"])
    cli._apply_video_overrides(args)
    assert MediaOfferConfig.from_environment().video.pixel_formats == PixelFormat.ALL
