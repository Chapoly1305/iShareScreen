"""Decoder lifecycle rules for fresh RTP SSRC generations."""
from __future__ import annotations

import collections
import time
import threading
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


def _session(codec: str, num_tiles: int = 1) -> Session:
    now = time.monotonic()
    session = Session.__new__(Session)
    session._video_codec = codec
    session._ssrc_to_tile = {
        0x1000 + i: i for i in range(num_tiles)
    }
    session._ssrc_blacklist = set()
    session._video_decryptor = types.SimpleNamespace(
        ssrc_counts=collections.Counter({
            **{0x1000 + i: 100 for i in range(num_tiles)},
            **{0x2000 + i: _DYNAMIC_SSRC_PACKET_THRESHOLD
               for i in range(num_tiles)},
        }),
    )
    session._last_publish_t = now - _SSRC_ADOPT_STALL_S - 0.1
    session._last_ssrc_adopt_ts = 0.0
    session._last_decoder_restart_t = now - 1.0
    session._restart_guard = threading.Lock()
    session._needs_param_harvest = False
    session._dpb_error_window = collections.deque([now])
    session._decoder = _Decoder()
    session._observed_tile_count = num_tiles
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


def test_hevc_first_live_ssrc_resets_burst_dpb_inside_guard(monkeypatch):
    monkeypatch.setenv("ISS_TILES_PER_FRAME", "4")
    session = _session("hevc", num_tiles=4)

    session._note_unknown_ssrc(0x2000)

    assert session._ssrc_to_tile == {0x2000 + i: i for i in range(4)}
    assert session._decoder.restart_calls == 1
    assert session.fir_calls == 1


def test_hevc_later_ssrc_adoption_keeps_rapid_restart_guard(monkeypatch):
    monkeypatch.setenv("ISS_TILES_PER_FRAME", "4")
    session = _session("hevc", num_tiles=4)
    session._note_unknown_ssrc(0x2000)

    for i in range(4):
        session._video_decryptor.ssrc_counts[0x3000 + i] = (
            _DYNAMIC_SSRC_PACKET_THRESHOLD
        )
    session._last_publish_t = time.monotonic() - _SSRC_ADOPT_STALL_S - 0.1
    session._note_unknown_ssrc(0x3000)

    assert session._ssrc_to_tile == {0x3000 + i: i for i in range(4)}
    assert session._decoder.restart_calls == 1
    assert session.fir_calls == 2


def test_adoption_picks_the_group_that_is_sending_not_the_lowest_ssrc(monkeypatch):
    # Live regression: four stale groups from the first second of the session
    # sorted below the live one and were adopted first, 2 s apart, for a 10 s
    # black start.
    monkeypatch.setenv("ISS_TILES_PER_FRAME", "4")
    session = _session("hevc", num_tiles=4)
    for i in range(4):
        session._video_decryptor.ssrc_counts[0x5000 + i] = 1000

    session._note_unknown_ssrc(0x5002)          # a packet from the live group

    assert session._ssrc_to_tile == {0x5000 + i: i for i in range(4)}


def test_single_picture_hevc420_later_ssrc_resets_inside_restart_guard(monkeypatch):
    from isharescreen.proxy.protocol.media_config import MediaOfferConfig, PixelFormat

    monkeypatch.setenv("ISS_TILES_PER_FRAME", "1")
    session = _session("hevc")
    session._media_offer_config = MediaOfferConfig()
    session._media_offer_config.video.pixel_formats = (
        PixelFormat.YUV420_8_FULL | PixelFormat.YUV420_8_VIDEO)
    session._media_offer_config.video.tiles_per_frame = 1
    session._note_unknown_ssrc(0x2000)

    session._video_decryptor.ssrc_counts[0x3000] = _DYNAMIC_SSRC_PACKET_THRESHOLD
    session._last_publish_t = time.monotonic() - _SSRC_ADOPT_STALL_S - 0.1
    session._note_unknown_ssrc(0x3000)

    assert session._ssrc_to_tile == {0x3000: 0}
    assert session._decoder.restart_calls == 2
    assert session.fir_calls == 2
