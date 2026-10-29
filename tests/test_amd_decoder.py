from types import SimpleNamespace
import pytest
from isharescreen import cli
from isharescreen.proxy.media import registry
from isharescreen.proxy.media.hevc import HevcDecoder
from isharescreen.proxy.protocol.media_config import MediaOfferConfig, PixelFormat


def _prepared_tile():
    from isharescreen.proxy.media.tiles import TileFrame
    return TileFrame(b"\x80" * 16, b"\x80" * 4, b"\x80" * 4,
                     4, 4, 4, 2, 2, 2)


def _publish_test_frame(decoder, pts, tile, corrupt=False):
    decoder._pts_to_tile[pts] = tile
    decoder._publish_frame(SimpleNamespace(pts=pts, flags=int(corrupt)))


def test_amd_batches_gpu_outputs_before_readback(monkeypatch):
    from isharescreen.proxy.media import hevc
    calls, notifications = [], []
    tile = _prepared_tile()
    monkeypatch.setattr(hevc, "_av_frame_to_tile",
                        lambda frame, *a: (calls.append(frame.pts) or tile, False))
    decoder = HevcDecoder(4, amd_only=True, enable_quality_gate=False,
                          on_frame_published=notifications.append)
    decoder._hw_verified = decoder._dpb_has_idr = True
    for i in range(3):
        _publish_test_frame(decoder, i, i)
    assert calls == notifications == []
    assert decoder.get_frame(0) is None
    _publish_test_frame(decoder, 3, 3)
    assert calls == notifications == [0, 1, 2, 3]
    assert not decoder._pending_downloads
    assert [decoder.get_frame(i) for i in range(4)] == [tile] * 4
    assert calls == [0, 1, 2, 3]  # no GPU work on the consumer thread
    assert decoder.get_frame(0) is None


def test_amd_quiet_partial_batch_is_published(monkeypatch):
    import threading
    from isharescreen.proxy.media import hevc
    ready = threading.Event()
    tile = _prepared_tile()
    monkeypatch.setattr(hevc, "_av_frame_to_tile", lambda *a: (tile, False))
    decoder = HevcDecoder(4, amd_only=True, enable_quality_gate=False,
                          on_frame_published=lambda _: ready.set())
    decoder._hw_verified = decoder._dpb_has_idr = True
    monkeypatch.setattr(decoder, "_decode_one",
                        lambda _, ti: _publish_test_frame(decoder, 0, ti))
    decoder._start_worker()
    try:
        decoder._queue.put((b"frame", 0))
        assert ready.wait(1), "partial batches must flush without another tile arriving"
        assert decoder.get_frame(0) is tile
    finally:
        decoder.close()
    assert not decoder._pending_downloads


def test_amd_display_does_not_wait_for_inflight_download(monkeypatch):
    import threading
    from isharescreen.proxy.media import hevc
    entered, release, consumed = threading.Event(), threading.Event(), threading.Event()
    tile = _prepared_tile()
    decoder = HevcDecoder(4, amd_only=True, enable_quality_gate=False)
    decoder._hw_verified = decoder._dpb_has_idr = True

    def convert(*_):
        entered.set()
        assert release.wait(2)
        return tile, False

    monkeypatch.setattr(hevc, "_av_frame_to_tile", convert)
    _publish_test_frame(decoder, 0, 0, corrupt=True)
    worker = threading.Thread(target=decoder._flush_downloads)
    worker.start()
    consumer = threading.Thread(target=lambda: (decoder.get_frame(0), consumed.set()))
    try:
        assert entered.wait(1)
        consumer.start()
        assert consumed.wait(.5), "the window must not wait on a GPU readback"
    finally:
        release.set()
        worker.join(2)
        if consumer.ident is not None:
            consumer.join(2)
    assert decoder._tiles[0].prepared_error
    assert decoder.get_frame(0) is tile


def test_amd_shutdown_waits_for_download_before_releasing_codec(monkeypatch):
    import threading
    from isharescreen.proxy.media import hevc
    entered, release, freed, closed = (threading.Event() for _ in range(4))
    decoder = HevcDecoder(4, amd_only=True, enable_quality_gate=False)
    decoder._hw_verified = decoder._dpb_has_idr = True
    decoder._codec = SimpleNamespace(decode=lambda _: [], flush_buffers=freed.set)

    def convert(*_):
        entered.set()
        assert release.wait(2)
        assert not freed.is_set()
        return _prepared_tile(), False

    monkeypatch.setattr(hevc, "_av_frame_to_tile", convert)
    monkeypatch.setattr(decoder, "_decode_one",
                        lambda _, ti: _publish_test_frame(decoder, 0, ti))
    decoder._start_worker()
    decoder._queue.put((b"frame", 0))
    closer = threading.Thread(target=lambda: (decoder.close(), closed.set()))
    try:
        assert entered.wait(1)
        closer.start()
        assert not closed.wait(.05)
        assert not freed.is_set()
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(2)
        else:
            decoder.close()
    assert closed.is_set() and freed.is_set()
    assert decoder._codec is None
    assert not decoder._pending_downloads
    assert all(slot.raw_frame is None for slot in decoder._tiles)


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
