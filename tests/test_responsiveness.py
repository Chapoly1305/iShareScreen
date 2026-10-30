"""Latency-sensitive ordering, wakeups, and retained-frame ownership."""
import re
import threading
import time
from types import SimpleNamespace

import pytest

from isharescreen.proxy.input import AsyncInputController
from isharescreen.proxy.control import ControlServer
from isharescreen.proxy.session import Session, _PacketGroup


def test_motion_coalescing_preserves_press_position_and_key_order():
    entered, release = threading.Event(), threading.Event()
    sent = []
    class Socket:
        def sendall(self, payload):
            entered.set()
            assert release.wait(2)
            sent.append(payload)
        def shutdown(self, how): pass
    controller = AsyncInputController(Socket(), None, server_width=100, server_height=100)
    controller.key_event(True, 65)
    try:
        assert entered.wait(1)
        controller.pointer_event(1, 10, 10)
        controller.pointer_event(1, 20, 20)
        controller.pointer_event(1, 30, 30)
        controller.key_event(False, 65)
        controller.pointer_event(0, 40, 40)
        with controller._ready:
            commands = list(controller._commands)
        assert [kind for kind, _ in commands] == ['transition', 'pointer', 'record', 'transition']
        assert commands[0][1] == (1, 10, 10)
        assert commands[1][1] == (1, 30, 30)
        assert controller.coalesced_moves == 1
    finally:
        release.set()
        controller.close()
    assert not controller._writer.is_alive()
    assert controller.tx_pkts == 5


def test_slow_dashboard_cannot_block_feedback_publisher(tmp_path):
    server = ControlServer(str(tmp_path/'control.sock'))
    entered, release = threading.Event(), threading.Event()
    class SlowClient:
        def sendall(self, payload):
            entered.set()
            assert release.wait(2)
        def close(self): pass
    server._clients.append(SlowClient())
    server._sender_thr = threading.Thread(target=server._send_loop)
    server._sender_thr.start()
    try:
        server.publish_snapshot({'frame': 1})
        assert entered.wait(1)
        # Publishing must not acquire the sender's socket or client-list lock.
        server.publish_snapshot({'frame': 2})
        server.publish_snapshot({'frame': 3})
        with server._outgoing_ready:
            assert len(server._outgoing) == 1
            assert server._outgoing[0]['data']['frame'] == 3
    finally:
        release.set()
        server.close()
    assert not server._sender_thr.is_alive()


def test_packet_group_duplicate_bookkeeping_does_not_scan_list():
    group = _PacketGroup([(65535, False, b'a')])
    group.add_packet(0, True, b'b')
    group.add_packet(0, True, b'duplicate')
    assert len(group) == 2
    assert group.sequences == {65535, 0}
    assert group.has_marker


def test_hevc_repair_holds_dependent_other_tile_until_reference_arrives():
    s = Session.__new__(Session)
    s._video_codec = 'hevc'
    s._pending_groups = {}; s._group_arrival = {}; s._recently_flushed = {}
    s._group_repair_deadline = {}; s._last_group_marker_seq = {1: 9}
    flushed = []
    def flush(key):
        flushed.append(key)
        grp = s._pending_groups.pop(key)
        s._group_arrival.pop(key, None)
        s._group_repair_deadline.pop(key, None)
        s._last_group_marker_seq[key[0]] = next(p[0] for p in grp if p[1])
    s._flush_group = flush
    s._queue_video_group_packet(1, 100, 10, False, b'a')
    s._queue_video_group_packet(1, 100, 12, True, b'c')
    s._queue_video_group_packet(2, 100, 30, True, b'dependent tile')
    assert not flushed
    s._queue_video_group_packet(1, 100, 11, False, b'repair')
    assert flushed == [(1, 100), (2, 100)]


def test_publish_wakes_native_event_loop_without_consuming_latest_frame():
    s = Session.__new__(Session)
    s._fresh_evt = threading.Event()
    s._send_ltr_ack = lambda ti: None
    wake = []
    s.set_frame_wakeup(lambda: wake.append(True))
    s._on_frame_published(0)
    assert wake == [True]
    assert s.wait_for_fresh_tile(timeout=0)
    s.set_frame_wakeup(None)
    s._on_frame_published(0)
    assert wake == [True]


def test_native_vt_plane_outlives_frame_and_decoder():
    import gc
    import os
    if os.environ.get("ISS_NATIVE_TESTS") != "1":
        pytest.skip("set ISS_NATIVE_TESTS=1 for native decode/plane ownership validation")
    from isharescreen.proxy.media import vtdecode
    if not vtdecode.available():
        pytest.skip('native VideoToolbox unavailable')
    from isharescreen.proxy.media.hwcaps import _HEVC444_SAMPLE
    params = {(n[0] >> 1) & 63: n for n in re.split(b'\x00\x00\x00?\x01', _HEVC444_SAMPLE) if n}
    d = vtdecode.VTHevcDecoder(1)
    try:
        d.set_params(params[32], params[33], {0: params[34]})
        d.start()
        d.feed_nalu(params[20], 0, 123)
        d._queue.join()
        frame = d.get_frame(0)
        assert frame is not None
        plane = frame.y
        assert isinstance(plane, memoryview)
        expected = bytes(plane)
        assert d.last_clean_donl == [123]
        d.close()
        del frame
        gc.collect()
        assert bytes(plane) == expected
    finally:
        d.close()


def test_cipher_batch_keeps_counter_and_record_order():
    from isharescreen.proxy.protocol.enc1103 import StreamCipher
    encoder = StreamCipher(bytes(36), ecb_key=bytes(16))
    decoder = StreamCipher(bytes(36), ecb_key=bytes(16))
    sent = []
    encoder.encrypt_and_send_many(SimpleNamespace(sendall=sent.append), [b'press', b'move', b'release'])
    assert len(sent) == 1
    messages, consumed = decoder.decrypt_stream(sent[0])
    assert messages == [b'press', b'move', b'release']
    assert consumed == len(sent[0])
    assert encoder._enc_ctr == 3


def test_startup_probe_requires_params_and_complete_reference(monkeypatch):
    import struct
    from isharescreen.proxy.protocol import burst
    class Decryptor:
        def decrypt(self, packet): return packet
        def get_primary_ssrc_group(self, **kwargs): return {1}
    monkeypatch.setattr(burst, 'reassemble_group', lambda payloads: payloads)
    probe = burst.BurstReadiness(Decryptor(), 'hevc', tile_group_size=1)
    def packet(seq, nalu, marker=True, ts=None):
        return struct.pack('>BBHII', 0x80, 0x80 if marker else 0, seq,
                           seq if ts is None else ts, 1), nalu
    for seq, nt in enumerate((32, 33, 34), 1):
        assert not probe(packet(seq, bytes([nt << 1, 1])))
    assert not probe(packet(4, b'\x28\x01', False, 4))
    # Marker with a missing middle fragment must not end collection.
    assert not probe(packet(6, b'\x28\x01', True, 4))
    assert probe(packet(5, b'\x28\x01', False, 4))


def test_browser_mailbox_bounds_callbacks_and_recovers_before_deltas():
    from isharescreen.frontend.wt.server import _FrameMailbox
    callbacks, received, recovery = [], [], []
    loop = SimpleNamespace(call_soon_threadsafe=lambda f, *a: callbacks.append((f, a)),
                           call_soon=lambda f, *a: callbacks.append((f, a)))
    mailbox = _FrameMailbox(loop, received.append, lambda: recovery.append(True))
    for _ in range(1000): mailbox.offer(b'\x00delta')
    assert len(callbacks) == 2  # one drain, one recovery; never 1000 AU callbacks
    while callbacks:
        f, a = callbacks.pop(0); f(*a)
    assert recovery == [True]
    assert not received
    mailbox.offer(b'\x01key')
    mailbox.offer(b'\x00delta')
    while callbacks:
        f, a = callbacks.pop(0); f(*a)
    assert received == [b'\x01key', b'\x00delta']


def test_browser_stale_mouse_datagrams_cannot_repress_buttons():
    from isharescreen.frontend.wt.server import _ViewerSession
    viewer = _ViewerSession(1, 1, None, None, None, None)
    assert viewer.accept_mouse({'buttons': 1, 'seq': 1})  # reliable press
    assert viewer.accept_mouse({'buttons': 1, 'seq': 2, 'motion': True})
    assert viewer.accept_mouse({'buttons': 0, 'seq': 3})  # reliable release
    assert not viewer.accept_mouse({'buttons': 1, 'seq': 2, 'motion': True})
    assert not viewer.accept_mouse({'buttons': 0, 'seq': 2, 'motion': True})
    assert viewer.accept_mouse({'buttons': 0, 'seq': 4, 'motion': True})


def test_browser_reorder_timeout_runs_without_another_packet():
    import pathlib
    import shutil
    import subprocess
    node = shutil.which('node')
    if not node: pytest.skip('node unavailable')
    viewer = pathlib.Path('src/isharescreen/frontend/wt/static/viewer.js').read_text()
    function = viewer[viewer.index('  function drainReorder()'):viewer.index('  function feedAU(au)')]
    script = '''
const assert = require('node:assert/strict');
let now = 100, scheduled = null, reorderTimer = null, nextSeq = 10, gapSince = 0;
const performance = {now: () => now};
const setTimeout = (f, d) => {scheduled = {f, d}; return 1;};
const clearTimeout = () => {};
const pending = new Map([[11, 'eleven']]), received = [];
const REORDER_GAP_MAX = 6, REORDER_GAP_MAX_MS = 45;
function feedAU(au) {received.push(au);}
''' + function + '''
drainReorder();
assert.equal(scheduled.d, 45);
assert.deepEqual(received, []);
now = 146; scheduled.f();
assert.deepEqual(received, ['eleven']);
assert.equal(nextSeq, 12);
received.length = 0; nextSeq = 0xfffffffe; gapSince = 0;
pending.set(0, 'zero'); pending.set(0xffffffff, 'last'); now = 200;
drainReorder(); now = 246; scheduled.f();
assert.deepEqual(received, ['last', 'zero']);
'''
    subprocess.run([node, '-'], input=script, text=True, check=True, capture_output=True)


def test_startup_probe_waits_for_every_negotiated_tile(monkeypatch):
    import struct
    from isharescreen.proxy.protocol import burst
    class Decryptor:
        def decrypt(self, packet): return packet
        def get_primary_ssrc_group(self, **kwargs): return {1, 2, 3, 4}
    monkeypatch.setattr(burst, 'reassemble_group', lambda payloads: payloads)
    probe = burst.BurstReadiness(Decryptor(), 'hevc', tile_group_size=4)
    def packet(ssrc, seq, nt):
        return struct.pack('>BBHII', 0x80, 0x80, seq, seq, ssrc), bytes([nt << 1, 1])
    for seq, nt in enumerate((32, 33, 34, 20), 1):
        assert not probe(packet(1, seq, nt))
    assert not probe(packet(2, 1, 1))
    assert not probe(packet(3, 1, 1))
    assert probe(packet(4, 1, 1))


def test_partial_tile_canvas_is_not_stretched_to_fill_window():
    from isharescreen.frontend.desktop.gpu import Renderer
    r = Renderer.__new__(Renderer)
    r._w, r._h = 3000, 2000
    r._content_w, r._content_h = 2736, 448
    r._dbg_tiles = {0: (2736, 448)}
    r.expect_tiles(4)
    assert r.content_dims() == (3000, 2000)
    r._dbg_tiles.update({1: (), 2: (), 3: ()})
    r._content_h = 1740
    assert r.content_dims() == (2736, 1740)  # preserve real fallback crop


def test_reconfigure_does_not_consume_old_decoded_frame():
    s = Session.__new__(Session)
    s._needs_param_harvest = True
    s._decoder = SimpleNamespace(get_frame=lambda ti: pytest.fail('old geometry was consumed'))
    assert s.get_frame(0) is None


def test_hevc_whole_missing_au_retransmission_precedes_later_arrival():
    s = Session.__new__(Session)
    s._video_codec = 'hevc'; s._pending_groups = {}; s._group_arrival = {}
    s._recently_flushed = {}; s._group_repair_deadline = {}
    s._last_group_marker_seq = {1: 9}
    flushed = []
    def flush(key):
        flushed.append(key)
        grp = s._pending_groups.pop(key)
        s._group_arrival.pop(key, None)
        s._group_repair_deadline.pop(key, None)
        s._last_group_marker_seq[key[0]] = next(p[0] for p in grp if p[1])
    s._flush_group = flush
    s._queue_video_group_packet(1, 200, 13, True, b'later')
    s._queue_video_group_packet(2, 200, 30, True, b'dependent sibling')
    assert not flushed
    s._queue_video_group_packet(1, 100, 10, False, b'retransmit first')
    s._queue_video_group_packet(1, 100, 11, False, b'middle')
    s._queue_video_group_packet(1, 100, 12, True, b'end')
    assert flushed == [(1, 100), (1, 200), (2, 200)]


def test_force_all_recovery_is_atomic_across_decoder_and_feedback_threads():
    from isharescreen.proxy.media.quality_gate import FrameQualityGate
    gate = FrameQualityGate(4)
    entered, release, read = threading.Event(), threading.Event(), threading.Event()
    original = gate.mark_decode_error
    def pause_first(ti):
        original(ti)
        if ti == 0:
            entered.set()
            assert release.wait(2)
    gate.mark_decode_error = pause_first
    force = threading.Thread(target=gate.force_keyframe_all)
    result = []
    reader = threading.Thread(target=lambda: (result.append(gate.bad_tiles), read.set()))
    force.start()
    try:
        assert entered.wait(1)
        reader.start()
        assert not read.wait(.05)
    finally:
        release.set(); force.join(2); reader.join(2)
    assert result == [{0, 1, 2, 3}]


def test_native_submit_error_wakes_feedback_without_waiting_for_watchdog(monkeypatch):
    from isharescreen.proxy.media import vtdecode
    if not vtdecode.available(): pytest.skip('native VideoToolbox unavailable')
    d = vtdecode.VTHevcDecoder(1)
    d._session = object()
    d._make_sample = lambda nalu: object()
    monkeypatch.setattr(vtdecode._VT, 'VTDecompressionSessionDecodeFrameWithOutputHandler',
                        lambda *args: (-12909, 0))
    wake = threading.Event()
    d.set_feedback_wakeup(wake.set)
    try:
        d.feed_nalu(b'\x02\x01\x80', 0, 456)
        assert wake.is_set()
        assert d.last_clean_donl == [None]
    finally:
        d._session = None  # Fake submit handle must not enter the native destructor.
        d.close()
