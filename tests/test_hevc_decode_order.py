"""HEVC tiles share one POC sequence, so they must be fed in (ts, tile) order."""
from __future__ import annotations

import threading
import time
import types

from isharescreen.proxy.media.quality_gate import FrameQualityGate
from isharescreen.proxy.session import Session, _HEVC_RTP_REPAIR_WAIT_S

_TILES = 4


def _session() -> tuple[Session, list[tuple[int, int]]]:
    s = Session.__new__(Session)
    s._video_codec = "hevc"
    s._reset_group_state()
    s._ssrc_to_tile = {0x1000 + t: t for t in range(_TILES)}
    s._last_group_marker_seq = {0x1000 + t: 99 for t in range(_TILES)}
    s._decoder = types.SimpleNamespace(_gate=FrameQualityGate(_TILES))
    s._tx_wakeup = threading.Event()
    flushed: list[tuple[int, int]] = []

    def flush(key):
        grp = s._pending_groups.pop(key)
        s._group_arrival.pop(key, None)
        s._group_repair_deadline.pop(key, None)
        s._recently_flushed[key] = time.monotonic()
        s._last_group_marker_seq[key[0]] = max(p[0] for p in grp if p[1])
        flushed.append((key[1], s._ssrc_to_tile[key[0]]))

    s._flush_group = flush
    return s, flushed


def _send(s: Session, tile: int, ts: int, seq: int, marker: bool = True) -> None:
    ssrc = 0x1000 + tile
    newest = s._max_seq.get(ssrc)
    if newest is None or 0 < ((seq - newest) & 0xFFFF) < 0x8000:
        s._max_seq[ssrc] = seq
    s._queue_video_group_packet(ssrc, ts, seq, marker, b"x")


def _age(s: Session, seconds: float) -> None:
    for key in s._group_arrival:
        s._group_arrival[key] -= seconds


def test_sibling_tile_waits_for_mid_au_nack_repair():
    # Live trace: tile 3 (POC 3) was fed before tile 2 (POC 2) while tile 2
    # waited on a retransmission, producing "Duplicate POC in a sequence: 2".
    s, flushed = _session()
    _send(s, 0, 100, 100)
    _send(s, 1, 100, 100)
    _send(s, 2, 100, 100, marker=False)
    _send(s, 2, 100, 102)          # seq 101 lost
    _send(s, 3, 100, 100)

    assert flushed == [(100, 0), (100, 1)]

    _send(s, 2, 100, 101, marker=False)  # NACK retransmission

    assert flushed == [(100, 0), (100, 1), (100, 2), (100, 3)]


def test_next_frame_waits_for_tile_still_receiving():
    # Live trace: tile 0 of the next frame (POC 87) was fed while tile 3
    # (POC 86) had no marker yet, producing "Duplicate POC ... 86".
    s, flushed = _session()
    for tile in range(3):
        _send(s, tile, 100, 100)
    _send(s, 3, 100, 100, marker=False)
    _send(s, 0, 200, 101)

    assert flushed == [(100, 0), (100, 1), (100, 2)]

    _send(s, 3, 100, 101)

    assert flushed == [(100, 0), (100, 1), (100, 2), (100, 3), (200, 0)]


def test_idle_tile_does_not_hold_later_tiles():
    # Apple emits no picture for an unchanged tile, so tile 1 being absent
    # from frame 100 is normal and must not delay tiles 2 and 3.
    s, flushed = _session()
    _send(s, 0, 100, 100)
    _send(s, 2, 100, 100)
    _send(s, 3, 100, 100)

    assert flushed == [(100, 0), (100, 2), (100, 3)]
    assert not s._decoder._gate.bad_tiles


def test_wholly_missing_tile_with_seq_gap_waits_for_retransmission():
    s, flushed = _session()
    _send(s, 0, 100, 100)
    _send(s, 1, 200, 101)          # seq 100 (tile 1, frame 100) lost: gap seen
    _send(s, 2, 100, 100)
    _send(s, 3, 100, 100)

    assert flushed == [(100, 0)]

    _send(s, 1, 100, 100)          # NACK retransmission

    assert flushed == [(100, 0), (100, 1), (100, 2), (100, 3), (200, 1)]


def test_unfilled_gap_stops_holding_after_reorder_window():
    s, flushed = _session()
    _send(s, 0, 100, 100)
    _send(s, 1, 200, 101)
    _send(s, 2, 100, 100)

    _age(s, _HEVC_RTP_REPAIR_WAIT_S)
    s._flush_hevc_ready_groups()

    assert flushed == [(100, 0), (100, 2)]
    # The retransmission now lands after its successor and is never fed.
    _send(s, 1, 100, 100)
    assert (0x1001, 100) not in s._pending_groups
    assert 1 in s._decoder._gate.bad_tiles


def test_group_older_than_cursor_is_discarded_not_fed():
    s, flushed = _session()
    for tile in range(_TILES):
        _send(s, tile, 200, 100)

    _send(s, 2, 100, 99)           # an older frame arriving after frame 200

    assert flushed == [(200, t) for t in range(_TILES)]
    assert (0x1002, 100) not in s._pending_groups
    assert 2 in s._decoder._gate.bad_tiles


def test_timestamp_wrap_keeps_decode_order():
    s, flushed = _session()
    for tile in range(_TILES):
        _send(s, tile, 0xFFFFFF00, 100)
    _send(s, 0, 0x100, 101)
    _send(s, 1, 0x100, 101)

    # Post-wrap timestamps are newer, not late.
    assert flushed[-2:] == [(0x100, 0), (0x100, 1)]
    assert not s._decoder._gate.bad_tiles


def test_held_tile_does_not_drop_its_own_next_frame():
    # Live regression: while tile 0 / frame 100 waited for tile 3's repair,
    # tile 0 / frame 200 failed the sequence anchor check, hit its repair
    # deadline and was dropped, cascading into thousands of drops.
    s, flushed = _session()
    for tile in range(3):
        _send(s, tile, 100, 100)
    _send(s, 3, 100, 100, marker=False)
    _send(s, 0, 200, 101)
    _send(s, 1, 200, 101)

    assert (0x1000, 200) not in s._group_repair_deadline
    for key in s._group_repair_deadline:
        s._group_repair_deadline[key] = time.monotonic() - 0.001
    s._evict_stale_groups()
    _send(s, 3, 100, 101)

    assert (200, 0) in flushed and (200, 1) in flushed


def _poc_session(pocs: dict[tuple[int, int], int]):
    s, flushed = _session()
    s._decoder._rps_tracker = types.SimpleNamespace(
        sps=types.SimpleNamespace(log2_max_pic_order_cnt_lsb=11))
    s._hevc_group_poc = lambda key: (
        (pocs[(key[1], key[0] - 0x1000)], False)
        if (key[1], key[0] - 0x1000) in pocs else None)
    return s, flushed


def test_consecutive_poc_feeds_past_idle_tile():
    # Tile 1 sent nothing for frame 100 (unchanged), so POCs stay consecutive.
    s, flushed = _poc_session({(100, 0): 10, (100, 2): 11, (100, 3): 12})
    _send(s, 0, 100, 100)
    _send(s, 2, 100, 100)
    _send(s, 3, 100, 100)

    assert flushed == [(100, 0), (100, 2), (100, 3)]
    assert not s._nack_pending


def test_poc_gap_holds_until_reordered_picture_arrives():
    # Tile 1's single-packet AU (POC 11) is delayed behind tile 2's and tile 1
    # is otherwise idle: no sequence gap is visible, only the POC jump 10 -> 12.
    s, flushed = _poc_session({(100, 0): 10, (100, 1): 11, (100, 2): 12})
    _send(s, 0, 100, 100)
    _send(s, 2, 100, 100)

    assert flushed == [(100, 0)]

    _send(s, 1, 100, 100)                      # arrives out of order

    assert flushed == [(100, 0), (100, 1), (100, 2)]


def test_poc_gap_gives_up_after_reorder_window():
    s, flushed = _poc_session({(100, 0): 10, (100, 2): 12})
    _send(s, 0, 100, 100)
    _send(s, 2, 100, 100)

    _age(s, _HEVC_RTP_REPAIR_WAIT_S)
    s._flush_hevc_ready_groups()

    assert flushed == [(100, 0), (100, 2)]


def test_dropped_picture_does_not_cause_a_second_wait():
    s, flushed = _poc_session({(100, 0): 10, (100, 2): 12})
    _send(s, 0, 100, 100)
    s._hevc_poc_slack = 1                      # tile 1's AU was dropped

    _send(s, 2, 100, 100)

    assert flushed == [(100, 0), (100, 2)]


def test_pictures_after_a_lost_keyframe_are_held_back_until_the_next_keyframe():
    # The keyframe that restarted the POC count was lost, so POC falls from 900
    # to 1. Those pictures reference it; feeding them collides with the old
    # sequence still in the decoder ("Duplicate POC in a sequence: 1").
    pocs = {(100, 0): 900, (200, 1): 1, (200, 2): 2}
    s, flushed = _poc_session(pocs)
    _send(s, 0, 100, 100)
    _send(s, 1, 200, 100)
    _send(s, 2, 200, 100)

    assert flushed == [(100, 0)]
    assert not s._pending_groups
    assert 1 in s._decoder._gate.bad_tiles

    # The next keyframe (tile 0) and what follows it decode normally.
    s._hevc_group_poc = lambda key: (0, True) if key[1] == 300 else (1, False)
    _send(s, 0, 300, 101)
    _send(s, 1, 400, 101)

    assert flushed == [(100, 0), (300, 0), (400, 1)]

