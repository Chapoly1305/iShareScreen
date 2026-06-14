"""Unit tests for the decoder-stall watchdog (Session._check_stall).

Pure-logic: a minimal `Session` is built via `__new__` with a fake decoder, and
`_check_stall` is driven with synthetic (gap, loss-growth, packet-flow,
bad_streak) state. Asserts the loss-aware discriminator:

  * lossless stall + packets flowing  → genuine VT saturation wedge → restart
  * stall WITH growing loss           → broken-ref → FIR only, never restart
  * apple-idle (no packets)           → neither (host isn't encoding)
"""
from __future__ import annotations

import time
import types

from isharescreen.proxy.session import Session


class _FakeDecoder:
    def __init__(self, bad_streaks):
        self._gate = types.SimpleNamespace(
            _states=[types.SimpleNamespace(bad_streak=b) for b in bad_streaks],
            _keyframe_required=set(),
        )
        self.restart_calls = 0

    def restart(self):
        self.restart_calls += 1


def _make_session(*, gap, lost_pkts=0, loss_growing=False,
                  video_pkt_age=0.1, bad_streaks=(0, 0, 0, 0), num_tiles=4):
    now = time.monotonic()
    s = Session.__new__(Session)
    s._connected = True
    s._last_publish_t = now - gap
    s._lost_pkts = lost_pkts
    s._last_video_pkt_t = now - video_pkt_age
    s._observed_tile_count = num_tiles  # backs the read-only `num_tiles` property
    s._decoder = _FakeDecoder(bad_streaks)
    # loss tracker: prev < cur ⇒ "loss growing now" ⇒ recent_loss True
    s._loss_at_prev_stall_check = lost_pkts - (10 if loss_growing else 0)
    s._last_loss_growth_t = 0.0
    s._last_decoder_restart_t = 0.0
    s._last_stuck_tile_fir_t = 0.0
    s._last_stall_fir_t = 0.0
    s.fir_calls = 0
    s.request_fir = lambda tile_idx=None: setattr(s, "fir_calls", s.fir_calls + 1)
    return s


def test_lossless_saturation_wedge_restarts():
    s = _make_session(gap=3.0, lost_pkts=500, loss_growing=False, video_pkt_age=0.1)
    s._check_stall()
    assert s._decoder.restart_calls == 1
    assert s.fir_calls >= 1


def test_loss_driven_stall_fir_only_no_restart():
    s = _make_session(gap=4.0, lost_pkts=500, loss_growing=True, video_pkt_age=0.1)
    s._check_stall()
    assert s._decoder.restart_calls == 0      # broken-ref must NOT flush
    assert s.fir_calls >= 1


def test_apple_idle_no_packets_does_nothing():
    # gap is long but no video packets for >1.5s → host isn't encoding.
    s = _make_session(gap=5.0, loss_growing=False, video_pkt_age=2.0)
    s._check_stall()
    assert s._decoder.restart_calls == 0
    assert s.fir_calls == 0


def test_pathB_stuck_tile_with_loss_fir_only():
    # one tile stuck (bad_streak high) but small gap so only Path B runs; loss
    # active ⇒ FIR storm, no flush.
    s = _make_session(gap=1.0, lost_pkts=500, loss_growing=True,
                      bad_streaks=(33, 0, 0, 0))
    s._check_stall()
    assert s._decoder.restart_calls == 0
    assert s.fir_calls >= 1


def test_pathB_stuck_tile_no_loss_restarts():
    # stuck tile, no loss ⇒ saturation wedge ⇒ restart.
    s = _make_session(gap=1.0, lost_pkts=500, loss_growing=False,
                      bad_streaks=(33, 0, 0, 0))
    s._check_stall()
    assert s._decoder.restart_calls == 1


def test_healthy_session_no_action():
    s = _make_session(gap=0.4, loss_growing=False, bad_streaks=(0, 0, 0, 0))
    s._check_stall()
    assert s._decoder.restart_calls == 0
    assert s.fir_calls == 0
