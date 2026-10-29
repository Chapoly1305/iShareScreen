"""Regression tests for browser-health FIR suppression."""
from __future__ import annotations

import time
import types

from isharescreen.proxy.session import Session


class _FakeDecoder:
    def __init__(self, *, attempts: int = 0) -> None:
        self._gate = types.SimpleNamespace(
            _keyframe_required={0},
            _fir_attempts=[attempts],
        )
        self.consume_calls = 0

    def consume_fir_request(self) -> set[int]:
        self.consume_calls += 1
        self._gate._fir_attempts[0] += 1
        return {0}


def _make_session(*, attempts: int = 0) -> Session:
    now = time.monotonic()
    session = Session.__new__(Session)
    session._decoder = _FakeDecoder(attempts=attempts)
    session._observed_tile_count = 1
    session._last_video_pkt_t = now
    session._browser_healthy_t = now
    session._dpb_forceall_pending = False
    session._last_dpb_error_t = 0.0
    session._grayout_window_tiles = set()
    session._grayout_window_t = 0.0
    session.sent_tiles = []

    def send_fir(tile_idx: int, log_per_tile: bool = True) -> bool:
        session.sent_tiles.append(tile_idx)
        return True

    session._send_fir_for_tile = send_fir
    return session


def test_browser_healthy_allows_first_recovery_fir_only() -> None:
    """A coarse browser "ok" must not suppress the first trusted recovery.

    Once that FIR is sent, the existing browser-health guard still suppresses
    sticky local-decoder retries while the browser continues to look healthy.
    """
    session = _make_session()

    session._drain_pending_fir()
    session._drain_pending_fir()

    assert session.sent_tiles == [0]
    assert session._decoder.consume_calls == 1


def test_browser_healthy_does_not_block_armed_force_all() -> None:
    """A normal-FPS smear can look "ok" but must not block AVC escalation."""
    session = _make_session(attempts=1)
    session._dpb_forceall_pending = True
    session._last_dpb_error_t = time.monotonic() - 3.0
    session.force_all_calls = []

    def request_fir(tile_idx=None) -> None:
        session.force_all_calls.append(tile_idx)

    session.request_fir = request_fir

    session._drain_pending_fir()

    assert session.force_all_calls == [None]
    assert session._dpb_forceall_pending is False
    assert session._decoder.consume_calls == 0
