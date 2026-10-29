"""Consumers never convert frames or wait for the producer's conversion."""
import threading
from types import SimpleNamespace

import av
import pytest

from isharescreen.proxy.media import avc, hevc
from isharescreen.proxy.media.tiles import TileFrame


def source_frame():
    frame = av.VideoFrame(8, 8, "nv12")
    for plane in frame.planes:
        plane.update(bytes([128]) * plane.buffer_size)
    return frame


def setup_decoder(backend):
    if backend == "hevc":
        decoder = hevc.HevcDecoder(4, prefer_hwaccel=False, enable_quality_gate=False)
        decoder._hw_verified = decoder._dpb_has_idr = True

        def feed():
            frame = source_frame()
            frame.pts = 1
            decoder._pts_to_tile[1] = 2
            decoder._publish_frame(frame)

        return decoder, hevc, feed
    decoder = avc.AvcDecoder(4, prefer_hwaccel=False, enable_quality_gate=False)
    decoder._await_key = False

    def decode(packet):
        frame = source_frame()
        frame.pts = packet.pts
        return [frame]

    decoder._codec = SimpleNamespace(decode=decode, close=lambda: None)
    return decoder, avc, lambda: decoder.feed_nalu(b"\x61\xc0", 2)


@pytest.mark.parametrize("backend", ["hevc", "avc"])
def test_immediate_preparation_before_notification(monkeypatch, backend):
    decoder, module, feed = setup_decoder(backend)
    notifications = []
    decoder._on_frame_published = lambda ti: notifications.append(decoder.get_frame(ti))
    feed()
    assert len(notifications) == 1  # no batching delay outside AMD
    tile = notifications[0]
    assert isinstance(tile, TileFrame)
    assert (tile.width, tile.height, tile.chroma_width) == (8, 8, 4)
    for row in range(tile.height):
        assert tile.y[row * tile.y_stride:row * tile.y_stride + tile.width] == bytes([128]) * tile.width
    assert decoder.get_frame(2) is None
    monkeypatch.setattr(module, "_av_frame_to_tile",
                        lambda *a: pytest.fail("consumer must not convert"))
    decoder._tiles[2].last_evaluated_count = 0
    assert decoder.get_frame(2) is tile


@pytest.mark.parametrize("backend", ["hevc", "avc"])
def test_conversion_does_not_block_consumer(monkeypatch, backend):
    decoder, module, feed = setup_decoder(backend)
    feed()  # Leave an older prepared frame available during the next conversion.
    entered, release, consumed = (threading.Event() for _ in range(3))
    converter = module._av_frame_to_tile

    def blocked(*args):
        entered.set()
        assert release.wait(2)
        return converter(*args)

    monkeypatch.setattr(module, "_av_frame_to_tile", blocked)
    producer = threading.Thread(target=feed)
    result = []
    consumer = threading.Thread(target=lambda: (result.append(decoder.get_frame(2)), consumed.set()))
    producer.start()
    try:
        assert entered.wait(1)
        consumer.start()
        assert consumed.wait(.5)
        assert isinstance(result[0], TileFrame)
    finally:
        release.set()
        producer.join(2)
        if consumer.ident is not None:
            consumer.join(2)
    assert isinstance(decoder.get_frame(2), TileFrame)


@pytest.mark.parametrize("backend", ["hevc", "avc"])
def test_conversion_error_flag_survives_handoff(monkeypatch, backend):
    decoder, module, feed = setup_decoder(backend)
    converter = module._av_frame_to_tile
    monkeypatch.setattr(module, "_av_frame_to_tile", lambda *a: (converter(*a)[0], True))
    marked = []
    monkeypatch.setattr(decoder._gate, "mark_decode_error", marked.append)
    feed()
    assert decoder._tiles[2].prepared_error
    assert isinstance(decoder.get_frame(2), TileFrame)
    assert marked == [2]


def test_avc_close_waits_for_conversion(monkeypatch):
    decoder, module, feed = setup_decoder("avc")
    entered, release, freed, closed = (threading.Event() for _ in range(4))
    decoder._codec.close = freed.set
    converter = module._av_frame_to_tile

    def blocked(*args):
        entered.set()
        assert release.wait(2)
        assert not freed.is_set()
        return converter(*args)

    monkeypatch.setattr(module, "_av_frame_to_tile", blocked)
    producer = threading.Thread(target=feed)
    closer = threading.Thread(target=lambda: (decoder.close(), closed.set()))
    producer.start()
    try:
        assert entered.wait(1)
        closer.start()
        assert not closed.wait(.05)
        assert not freed.is_set()
    finally:
        release.set()
        producer.join(2)
        if closer.ident is not None:
            closer.join(2)
    assert closed.is_set() and freed.is_set()
