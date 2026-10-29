"""AVC packet processing stays responsive while decode/download is busy."""
import threading
from types import SimpleNamespace

from isharescreen.proxy.media.avc import AvcDecoder

P = b'\x61\xc0'
I = b'\x61\xb0'


def decoder_with_codec(decode):
    d = AvcDecoder(1, prefer_hwaccel=False, enable_quality_gate=False)
    d._sps, d._pps = b'sps', b'pps'
    d._codec = SimpleNamespace(decode=decode, close=lambda: None)
    d._await_key = False
    return d


def test_worker_does_not_block_packet_thread_and_preserves_order():
    entered, release, finished = (threading.Event() for _ in range(3))
    calls = []
    def decode(pkt):
        calls.append(pkt.pts)
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        if len(calls) == 4:
            finished.set()
        return []
    d = decoder_with_codec(decode)
    d.start()
    try:
        d.feed_nalu(P, 0)
        assert entered.wait(1)
        for _ in range(3):
            d.feed_nalu(P, 0)
        assert d._queue.qsize() == 3
        assert d.decode_queue_depth >= 3
        release.set()
        assert finished.wait(1)
        assert calls == [0, 1, 2, 3]
    finally:
        release.set()
        d.close()


def test_overflow_bounds_queue_and_requires_reference_recovery():
    d = decoder_with_codec(lambda pkt: [])
    d._worker = object()  # Hold the consumer while testing bounded submission.
    for _ in range(9):
        d.feed_nalu(P, 0)
    assert d._queue.qsize() == 1
    assert d.decode_queue_drops == 8
    d._decode_nalu(*d._queue.get_nowait())
    assert d._reference_reset_pending and d._await_key
    assert d._gate.consume_fir_request() == {0}
    assert d._next_pts == 0  # No delta enters the now-invalid reference chain.
    d._worker = None
    d.close()


def test_restart_discards_pre_restart_work():
    calls = []
    d = decoder_with_codec(lambda pkt: calls.append(pkt))
    old_generation = d._generation
    d._sps = d._pps = b''
    d.restart()
    d._decode_nalu(I, 0, old_generation)
    assert not calls and d._codec is None
    d.close()


def test_deferred_download_publishes_one_per_submission_and_flushes_last():
    prepared = []
    def decode(pkt):
        return [SimpleNamespace(pts=pkt.pts, format=SimpleNamespace(name='d3d11'))]
    d = decoder_with_codec(decode)
    d._prepare_frame_locked = lambda frame, ti: (prepared.append(frame.pts) or True)
    notifications = []
    d._on_frame_published = notifications.append
    for n in range(4):
        d._decode_nalu(P, 0)
        assert len(prepared) == n
        assert len(d._pending_frames) == 1
    assert prepared == [0, 1, 2]
    d._flush_pending()
    assert prepared == [0, 1, 2, 3]
    assert notifications == [0] * 4
    d.close()


def test_worker_flushes_quiet_gpu_frame_and_close_joins():
    ready = threading.Event()
    d = decoder_with_codec(lambda pkt: [SimpleNamespace(pts=pkt.pts, format=SimpleNamespace(name='d3d11'))])
    d._prepare_frame_locked = lambda frame, ti: True
    d._on_frame_published = lambda ti: ready.set()
    d.start()
    worker = d._worker
    try:
        d.feed_nalu(P, 0)
        assert ready.wait(1)
    finally:
        d.close()
    assert not worker.is_alive()
    assert not d._pending_frames
    d.feed_nalu(I, 0)
    assert d._queue.empty()
