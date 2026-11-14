"""Media feedback: legacy keepalive by default, upstream PR #31 RR routing opt-in."""
from __future__ import annotations

import socket

from isharescreen.proxy.session import Session


class _Sock:
    def __init__(self) -> None:
        self.sent: list[tuple[bytes, tuple[str, int]]] = []

    def sendto(self, data, addr):
        self.sent.append((data, addr))


class _Enc:
    def __init__(self, tag: bytes) -> None:
        self.tag = tag

    def encrypt(self, payload, pt):
        return self.tag + bytes([pt]) + payload

    def protect(self, rtcp):
        return self.tag + rtcp


def _session(feedback_rr: bool) -> Session:
    s = Session.__new__(Session)
    s._feedback_rr = feedback_rr
    s._sock_ctrl, s._sock_video = _Sock(), _Sock()
    s._audio_encryptor = _Enc(b"A")
    s._audio_srtcp_enc = _Enc(b"R") if feedback_rr else None
    s._srtcp_enc = _Enc(b"V")
    s._our_audio_ssrc, s._our_video_ssrc = 0x1111, 0x2222
    s._dest_host = "192.0.2.1"
    s._config = type("C", (), {"udp_ctrl_port": 5900, "udp_video_port": 5901, "port": 5900})()
    s._ssrc_to_tile, s._max_seq, s._roc, s._server_sr = {}, {}, {}, {}
    s._tx_tick = 1
    return s


def test_default_keeps_pt101_keepalive_and_ctrl_rr():
    s = _session(False)
    s._send_heartbeat()
    s._send_rr_and_maybe_sr()
    assert [addr for _, addr in s._sock_ctrl.sent] == [("192.0.2.1", 5900)] * 2
    assert s._sock_ctrl.sent[0][0].startswith(b"A\x65")      # PT=101 audio RTP
    assert s._sock_ctrl.sent[1][0].startswith(b"V")
    assert s._sock_video.sent == []


def test_opt_in_sends_audio_rr_and_video_rr_on_video_leg():
    s = _session(True)
    s._send_heartbeat()
    s._send_rr_and_maybe_sr()
    (audio, audio_addr), = s._sock_ctrl.sent
    (video, video_addr), = s._sock_video.sent
    assert audio_addr == ("192.0.2.1", 5900) and audio.startswith(b"R")
    assert audio[1 + 1] == 201                                 # RTCP RR
    assert int.from_bytes(audio[5:9], "big") == 0x1111         # our audio SSRC
    assert video_addr == ("192.0.2.1", 5901) and video.startswith(b"V")


def test_burst_drain_ticks_feedback_while_waiting():
    a, b = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
    try:
        ticks: list[int] = []
        Session._drain_socket_into(a, [], max_seconds=0.7, ready=lambda _p: False,
                                   on_tick=lambda: ticks.append(1))
        assert len(ticks) == 2          # t=0 and t=0.5
    finally:
        a.close()
        b.close()
