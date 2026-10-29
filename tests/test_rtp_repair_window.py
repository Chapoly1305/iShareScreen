"""AVC RTP loss waits briefly for NACK repair without reordering frames."""
from __future__ import annotations

import threading
import time
import types

from isharescreen.proxy.media.quality_gate import FrameQualityGate
from isharescreen.proxy.session import Session


def _group_session() -> Session:
    session = Session.__new__(Session)
    session._video_codec = "avc"
    session._pending_groups = {}
    session._group_arrival = {}
    session._recently_flushed = {}
    session._group_repair_deadline = {}
    session._last_group_marker_seq = {}
    return session


def test_sequence_sort_handles_65535_to_zero_wrap():
    packets = [
        (0, False, b"zero"),
        (1, True, b"one"),
        (65534, False, b"penultimate"),
        (65535, False, b"last"),
    ]

    ordered = Session._sorted_group_packets(packets)

    assert [p[0] for p in ordered] == [65534, 65535, 0, 1]


def test_missing_first_packet_of_next_group_waits_for_repair():
    session = _group_session()
    session._last_group_marker_seq[0x1000] = 10
    key = (0x1000, 200)

    session._queue_video_group_packet(*key, 12, True, b"end")

    assert key in session._group_repair_deadline
    assert key in session._pending_groups


def test_incomplete_marker_waits_and_late_packet_unblocks_in_order():
    session = _group_session()
    flushed: list[tuple[int, int]] = []

    def flush(key):
        flushed.append(key)
        grp = session._pending_groups[key]
        markers = [seq for seq, marker, _payload in grp if marker]
        if markers:
            session._last_group_marker_seq[key[0]] = markers[-1]
        session._pending_groups.pop(key, None)
        session._group_arrival.pop(key, None)
        session._group_repair_deadline.pop(key, None)
        session._recently_flushed[key] = time.monotonic()

    session._flush_group = flush
    first = (0x1000, 100)
    later = (0x1000, 200)

    session._queue_video_group_packet(*first, 10, False, b"a")
    session._queue_video_group_packet(*first, 12, True, b"c")
    session._queue_video_group_packet(*later, 13, True, b"next")

    assert first in session._group_repair_deadline
    assert flushed == []

    # Retransmitted seq=11 completes the blocked AU. It must flush before the
    # already-complete later timestamp.
    session._queue_video_group_packet(*first, 11, False, b"b")

    assert flushed == [first, later]


def test_missing_marker_blocks_later_group_until_timeout():
    session = _group_session()
    session._last_group_marker_seq[0x1000] = 9
    session._decoder = types.SimpleNamespace(
        _gate=FrameQualityGate(1),
        mark_reference_chain_broken=lambda _trigger: None,
    )
    session._ssrc_to_tile = {0x1000: 0}
    session._tx_wakeup = threading.Event()
    flushed: list[tuple[int, int]] = []

    def flush(key):
        flushed.append(key)
        grp = session._pending_groups.pop(key)
        session._group_arrival.pop(key, None)
        session._group_repair_deadline.pop(key, None)
        markers = [seq for seq, marker, _payload in grp if marker]
        if markers:
            session._last_group_marker_seq[key[0]] = markers[-1]

    session._flush_group = flush
    missing_marker = (0x1000, 100)
    later = (0x1000, 200)
    session._queue_video_group_packet(*missing_marker, 10, False, b"tail")
    session._queue_video_group_packet(*later, 12, True, b"next")

    assert missing_marker in session._group_repair_deadline
    assert flushed == []

    session._group_repair_deadline[missing_marker] = time.monotonic() - 0.001
    session._evict_stale_groups()

    assert flushed == [later]


def test_wholly_missing_au_retransmission_restores_decode_order():
    session = _group_session()
    session._last_group_marker_seq[0x1000] = 9
    flushed: list[tuple[int, int]] = []

    def flush(key):
        flushed.append(key)
        grp = session._pending_groups.pop(key)
        session._group_arrival.pop(key, None)
        session._group_repair_deadline.pop(key, None)
        markers = [seq for seq, marker, _payload in grp if marker]
        if markers:
            session._last_group_marker_seq[key[0]] = markers[-1]

    session._flush_group = flush
    later = (0x1000, 200)
    retransmitted = (0x1000, 100)

    # AU 100 (seq 10-12) was wholly absent, so AU 200 exposes the gap first.
    session._queue_video_group_packet(*later, 13, True, b"later")
    assert later in session._group_repair_deadline

    # Its NACK retransmission arrives later in wall-clock time but must decode
    # first based on sequence distance from the last consumed marker.
    session._queue_video_group_packet(*retransmitted, 10, False, b"a")
    session._queue_video_group_packet(*retransmitted, 11, False, b"b")
    session._queue_video_group_packet(*retransmitted, 12, True, b"c")

    assert flushed == [retransmitted, later]


def test_expired_repair_drops_frame_and_arms_reference_recovery():
    session = _group_session()
    gate = FrameQualityGate(1)
    broken: list[str] = []
    session._decoder = types.SimpleNamespace(
        _gate=gate,
        mark_reference_chain_broken=broken.append,
    )
    session._ssrc_to_tile = {0x1000: 0}
    session._tx_wakeup = threading.Event()
    session._flush_ready_groups = lambda _ssrc: None
    key = (0x1000, 100)

    session._queue_video_group_packet(*key, 10, False, b"a")
    session._queue_video_group_packet(*key, 12, True, b"c")
    session._group_repair_deadline[key] = time.monotonic() - 0.001

    session._evict_stale_groups()

    assert key not in session._pending_groups
    assert gate._keyframe_required == {0}
    assert len(broken) == 1
    assert "NACK/reorder timeout" in broken[0]
    assert session._tx_wakeup.is_set()


def test_sequence_gap_wakes_nack_without_immediate_decoder_failure():
    session = Session.__new__(Session)
    gate = FrameQualityGate(1)
    session._decoder = types.SimpleNamespace(_gate=gate)
    session._ssrc_to_tile = {0x1000: 0}
    session._max_seq = {0x1000: 10}
    session._roc = {0x1000: 0}
    session._nack_pending = __import__("collections").defaultdict(set)
    session._received_pkts = 0
    session._lost_pkts = 0
    session._lost_pkts_per_tile = [0]
    session._last_video_pkt_t = 0.0
    session._tx_wakeup = threading.Event()

    session._track_seq(0x1000, 12)

    assert session._nack_pending[0x1000] == {11}
    assert gate._keyframe_required == set()
    assert session._tx_wakeup.is_set()
