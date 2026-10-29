"""Decoder lifecycle rules for fresh RTP SSRC generations."""
from __future__ import annotations

import collections
import time
import types

from isharescreen.proxy.session import (
    Session,
    _DYNAMIC_SSRC_PACKET_THRESHOLD,
    _SSRC_ADOPT_STALL_S,
)


class _Decoder:
    def __init__(self) -> None:
        self.restart_calls = 0

    def restart(self) -> None:
        self.restart_calls += 1


def _session(codec: str) -> Session:
    now = time.monotonic()
    session = Session.__new__(Session)
    session._video_codec = codec
    session._ssrc_to_tile = {0x1000: 0}
    session._ssrc_blacklist = set()
    session._video_decryptor = types.SimpleNamespace(
        ssrc_counts=collections.Counter({
            0x1000: 100,
            0x2000: _DYNAMIC_SSRC_PACKET_THRESHOLD,
        }),
    )
    session._last_publish_t = now - _SSRC_ADOPT_STALL_S - 0.1
    session._last_ssrc_adopt_ts = 0.0
    session._last_decoder_restart_t = now - 1.0
    session._needs_param_harvest = False
    session._dpb_error_window = collections.deque([now])
    session._decoder = _Decoder()
    session._observed_tile_count = 1
    session.fir_calls = 0
    session.request_fir = lambda tile=None: setattr(
        session, "fir_calls", session.fir_calls + 1,
    )
    return session


def test_avc_fresh_ssrc_always_resets_dpb_inside_restart_guard(monkeypatch):
    monkeypatch.setenv("ISS_TILES_PER_FRAME", "1")
    session = _session("avc")

    session._note_unknown_ssrc(0x2000)

    assert session._ssrc_to_tile == {0x2000: 0}
    assert session._decoder.restart_calls == 1
    assert session.fir_calls == 1


def test_hevc_fresh_ssrc_keeps_rapid_restart_guard(monkeypatch):
    monkeypatch.setenv("ISS_TILES_PER_FRAME", "1")
    session = _session("hevc")

    session._note_unknown_ssrc(0x2000)

    assert session._ssrc_to_tile == {0x2000: 0}
    assert session._decoder.restart_calls == 0
    assert session.fir_calls == 1
