"""Regression tests for ownership and startup failure handling."""
from types import SimpleNamespace
import threading

import pytest

from isharescreen.frontend.wt.access import ViewerAccess
from isharescreen.frontend.wt.server import WebTransportBridge
from isharescreen.proxy.config import SessionConfig
from isharescreen.proxy.media import registry
from isharescreen.util import pcap_recorder


def test_viewer_access_requires_local_authority_and_launch_token():
    access = ViewerAccess(4433)
    assert access.allows("localhost:4433", "http://localhost:4433", access.token)
    assert not access.allows("localhost:4433", None, "")
    assert not access.allows("evil.example:4433", None, access.token)
    assert not access.allows("localhost:4433", "https://evil.example", access.token)
    assert access.allows_connect({b":authority": b"localhost:4433",
        b":path": ("/wt?token=" + access.token).encode()})


def test_join_replays_only_to_new_viewer_without_holding_lock():
    bridge = WebTransportBridge(SessionConfig("host", "user", "pass"))
    bridge._session = SimpleNamespace(request_fir=lambda *a: None)
    bridge._last_config_env = b"config"
    bridge._last_keyframe_env = {0: b"key"}
    received = []
    def send(frame):
        assert bridge._viewers_lock.acquire(blocking=False)
        bridge._viewers_lock.release()
        received.append(frame)
    bridge._loop = SimpleNamespace(call_soon_threadsafe=lambda fn, arg: fn(arg))
    bridge.viewers.append(SimpleNamespace(send_frame=lambda frame: pytest.fail("existing viewer reset")))
    bridge.add_viewer(SimpleNamespace(send_frame=send))
    assert received == [b"config", b"key"]


def test_disconnect_during_connect_discards_unpublished_session(monkeypatch):
    from isharescreen.frontend.wt import server
    bridge = WebTransportBridge(SessionConfig("host", "user", "pass", audio=False))
    bridge.viewers.append(object())
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()
    class Session:
        def __init__(self, config): pass
        def connect(self):
            entered.set()
            assert release.wait(2)
        def set_video_au_callback(self, cb): pass
        def set_cursor_callback(self, cb): pass
        def set_audio_callback(self, cb): pass
        def close(self): closed.set()
    monkeypatch.setattr(server, "Session", Session)
    worker = threading.Thread(target=bridge._start_session_thread, args=(0,))
    worker.start()
    assert entered.wait(2)
    bridge.remove_viewer(bridge.viewers[0])
    release.set()
    worker.join(2)
    assert not worker.is_alive()
    assert closed.is_set()
    assert bridge._session is None


def test_decoder_initialization_failure_closes_before_fallback(monkeypatch):
    events = []
    def spec(name):
        return SimpleNamespace(name=name, note="", available=lambda: True,
            build=lambda *a, **kw: SimpleNamespace(name=name, close=lambda: events.append("close " + name)))
    first, second = spec("hardware"), spec("software")
    monkeypatch.setattr(registry, "select", lambda *a, **kw: first)
    monkeypatch.setattr(registry, "candidates", lambda *a: [first, second])
    def initialize(decoder):
        events.append("start " + decoder.name)
        if decoder.name == "hardware": raise RuntimeError("device failed")
    selected, decoder = registry.build_best("avc", override=None, num_tiles=1, initialize=initialize)
    assert selected is second
    assert events == ["start hardware", "close hardware", "start software"]


def test_recorder_disk_failure_stops_queueing(monkeypatch, tmp_path):
    closed = threading.Event()
    class BrokenFile:
        packets = bytes = 0
        def __init__(self, path): self.path = path
        def write(self, *args): raise OSError("disk full")
        def close(self): closed.set()
    monkeypatch.setattr(pcap_recorder, "_PcapFile", BrokenFile)
    recorder = pcap_recorder.PcapRecorder(str(tmp_path / "capture.pcap"), "127.0.0.1", "127.0.0.2")
    recorder.record_udp(True, 1000, 2000, b"data")
    assert closed.wait(2)
    recorder.close()
    recorder.record_udp(True, 1000, 2000, b"more")
    assert recorder.write_error == "disk full"
    assert not recorder._pending


def test_recorder_queue_has_byte_limit(tmp_path):
    recorder = pcap_recorder.PcapRecorder(str(tmp_path / "capture.pcap"), "127.0.0.1", "127.0.0.2")
    with recorder._cond:
        recorder._max_pending_bytes = 3
        recorder._enqueue_locked(0, b"123")
        recorder._enqueue_locked(0, b"4")
        assert recorder._pending_bytes == 3
        assert recorder.dropped_packets == 1
    recorder.close()


def test_desktop_failure_releases_acquired_resources(monkeypatch):
    from isharescreen.frontend.desktop import app
    closed = []
    def failing_viewer(config, *, resources, **options):
        resources.callback(closed.append, "session")
        resources.callback(closed.append, "audio")
        raise RuntimeError("window creation failed")
    monkeypatch.setattr(app, "_run_viewer", failing_viewer)
    with pytest.raises(RuntimeError, match="window creation failed"):
        app.run(SessionConfig("host", "user", "pass"))
    assert closed == ["audio", "session"]


def test_session_cleanup_survives_callback_failure():
    closed = []
    def broken_callback(value): raise RuntimeError("already stopped")
    session = SimpleNamespace(set_video_au_callback=broken_callback,
                              close=lambda: closed.append("session"))
    audio = SimpleNamespace(close=lambda: closed.append("audio"))
    WebTransportBridge._close_resources(session, audio, None, None)
    assert closed == ["session", "audio"]
