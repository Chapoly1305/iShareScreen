"""Unit tests for HEVC decode-error recovery (hevc.py).

These are pure-logic tests: no real decode, no network, no host. They build a
minimal `HevcDecoder` via `__new__` and drive `_handle_decode_error` /
`_try_recovery` with synthetic exceptions, asserting the EAGAIN-backpressure
vs genuine-error policy and the wedge recovery (one FIR + queue drain).
"""
from __future__ import annotations

import errno
import queue as _queue

from isharescreen.proxy.media.hevc import (
    HevcDecoder, _TileSlot, _HWACCEL_SILENT_NALU_LIMIT,
)
from isharescreen.proxy.media.quality_gate import FrameQualityGate


def _make_decoder(num_tiles: int = 4) -> HevcDecoder:
    d = HevcDecoder.__new__(HevcDecoder)
    d._gate = FrameQualityGate(num_tiles)
    d._tiles = [_TileSlot() for _ in range(num_tiles)]
    d._queue = None
    d._eagain_streak = 0
    d._recovery_in_progress = False
    d._dpb_has_idr = True
    d._hw_name = "videotoolbox"
    return d


def _exc(errno_val: int) -> OSError:
    e = OSError("avcodec_send_packet()")
    e.errno = errno_val
    return e


_SLICE = b"\x02\x00"  # nal_unit_type 1 (TRAIL) — a non-IRAP slice


def test_eagain_does_not_fir_or_bump_bad_streak():
    """errno=35 (EAGAIN) is backpressure, not corruption: no FIR, no bad_streak."""
    d = _make_decoder()
    d._handle_decode_error(0, _SLICE, _exc(errno.EAGAIN))
    assert d._gate._states[0].bad_streak == 0
    assert 0 not in d._gate._keyframe_required
    assert d._eagain_streak == 1  # still counted for wedge detection


def test_genuine_error_fires_fir():
    """A real decode error (-12909 / -17694) re-roots via FIR."""
    d = _make_decoder()
    d._handle_decode_error(0, _SLICE, _exc(-12909))
    assert d._gate._states[0].bad_streak == 1
    assert 0 in d._gate._keyframe_required


def test_eagain_streak_resets_on_publish_path():
    """A successful publish (simulated) clears the streak so slow-but-producing
    never reaches the wedge threshold."""
    d = _make_decoder()
    for _ in range(10):
        d._handle_decode_error(0, _SLICE, _exc(errno.EAGAIN))
    assert d._eagain_streak == 10
    d._eagain_streak = 0  # what _decode_one does on a successful publish
    d._handle_decode_error(0, _SLICE, _exc(errno.EAGAIN))
    assert d._eagain_streak == 1


def test_sustained_eagain_wedge_drains_queue_and_fires_one_fir():
    """SILENT_NALU_LIMIT consecutive EAGAINs with zero output = genuine VT
    wedge → no-flush recovery: drain the stale backlog, drop to wait-for-IDR,
    and request exactly one FIR."""
    d = _make_decoder()
    d._queue = _queue.Queue()
    for _ in range(120):
        d._queue.put_nowait((b"x", 0))
    for _ in range(_HWACCEL_SILENT_NALU_LIMIT):
        d._handle_decode_error(0, _SLICE, _exc(errno.EAGAIN))
    assert d._recovery_in_progress is True
    assert d._dpb_has_idr is False          # _try_recovery engaged
    assert d._queue.qsize() == 0            # stale backlog drained
    assert 0 in d._gate._keyframe_required  # one wedge-entry FIR
    assert d._eagain_streak == 0            # reset by _try_recovery


def test_wedge_only_escalates_once():
    """recovery_in_progress guards re-entry until a publish resets it."""
    d = _make_decoder()
    d._queue = _queue.Queue()
    for _ in range(_HWACCEL_SILENT_NALU_LIMIT + 30):
        d._handle_decode_error(0, _SLICE, _exc(errno.EAGAIN))
    # one escalation; the extra EAGAINs after recovery_in_progress don't re-drain
    assert d._recovery_in_progress is True
