"""AVC must never advertise HEVC's long-term-reference protocol."""
from __future__ import annotations

from isharescreen.proxy.protocol import offers
from isharescreen.proxy.session import Session, SessionConfig


def test_avc_offer_disables_ltrp_even_when_global_default_is_on(monkeypatch):
    monkeypatch.setenv("ISS_VIDEO_CODEC", "avc")
    monkeypatch.setenv("ISS_LTRP", "1")

    blob = offers._build_mediablob(7, session_id=1, timestamp=1)

    assert offers._ltrp_enabled_for_codec("avc") is False
    assert b"LTR;" not in blob


def test_default_offer_keeps_ltrp_for_hevc_capable_bank(monkeypatch):
    monkeypatch.setenv("ISS_VIDEO_CODEC", "both")
    monkeypatch.delenv("ISS_LTRP", raising=False)

    blob = offers._build_mediablob(7, session_id=1, timestamp=1)

    assert offers._ltrp_enabled_for_codec("both") is True
    assert b"LTR;" in blob


def test_avc_session_never_sends_ltr_ack(monkeypatch):
    monkeypatch.setenv("ISS_VIDEO_CODEC", "avc")
    monkeypatch.setenv("ISS_LTRP", "1")

    session = Session(SessionConfig(
        host="example.invalid", username="user", password="unused",
    ))

    assert session._video_codec == "avc"
    assert session._ltr_enabled is False
