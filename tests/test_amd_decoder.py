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


def test_amd_context_reserves_surfaces_without_cpu_worker_fanout(monkeypatch):
    from isharescreen.proxy.media import hevc, amd
    context = SimpleNamespace(open=lambda: None)
    monkeypatch.setattr(amd, "hardware_accel", lambda **kw: object())
    monkeypatch.setattr(hevc, "av", SimpleNamespace(CodecContext=SimpleNamespace(create=lambda *a, **kw: context)))
    decoder = HevcDecoder(4, amd_only=True)
    assert decoder._try_hwaccel("d3d11va", b"params") is context
    assert context.thread_count == 1
    assert context.options == {"extra_hw_frames": "16"}


def test_amd_hardware_frames_defer_download(monkeypatch):
    from isharescreen.proxy.media import hevc, amd
    requested = []
    monkeypatch.setattr(amd, "hardware_accel", lambda **kw: requested.append(kw) or object())
    context = SimpleNamespace(open=lambda: None)
    monkeypatch.setattr(hevc, "av", SimpleNamespace(CodecContext=SimpleNamespace(create=lambda *a, **kw: context)))
    HevcDecoder(4, amd_only=True)._try_hwaccel("d3d11va", b"params")
    assert requested == [{"defer_download": True}]


def test_gpu_download_preserves_420_sampling():
    import av
    from isharescreen.proxy.media.decode_common import _av_frame_to_tile
    gpu_frame = SimpleNamespace(format=SimpleNamespace(name="d3d11"),
                                sw_format=SimpleNamespace(name="nv12"), width=64, height=32)
    nv12 = av.VideoFrame(64, 32, "nv12")
    for plane in nv12.planes:
        plane.update(bytes([128]) * plane.buffer_size)
    requested = []
    converter = av.video.reformatter.VideoReformatter()

    def reformat(frame, *, format):
        requested.append(format)
        if frame is gpu_frame:
            assert format == "nv12"
            return nv12
        return converter.reformat(frame, format=format)

    tile, corrupt = _av_frame_to_tile(gpu_frame, [SimpleNamespace(reformat=reformat)], set())
    assert requested == ["nv12", "yuv420p"]
    assert not corrupt
    assert (tile.width, tile.height, tile.chroma_width, tile.chroma_height) == (64, 32, 32, 16)
