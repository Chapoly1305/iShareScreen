"""Wire compatibility and configurable offer -> session integration."""
from __future__ import annotations

import hashlib
import json
import plistlib
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from isharescreen.proxy.protocol import offers
from isharescreen.proxy.protocol.media_config import (
    BandwidthMode, BandwidthSettings, HDRMode, MediaOfferConfig, PixelFormat,
    RuleOperation, ScreenFormat,
)
from isharescreen.proxy.protocol.srtp import SRTPDecryptor
from isharescreen.proxy import session as session_module
from isharescreen.proxy.session import Session, SessionConfig


_LEGACY = json.loads((Path(__file__).parent / "fixtures/media_offers_legacy.json").read_text())


def _fields(data):
    """Inspect the actual wire fields, independently of model attribute names."""
    result = {}
    pos = 0
    while pos < len(data):
        tag, pos = offers._read_varint(data, pos)
        number, wire_type = tag >> 3, tag & 7
        if wire_type == 0:
            value, pos = offers._read_varint(data, pos)
        elif wire_type == 2:
            size, pos = offers._read_varint(data, pos)
            value, pos = data[pos:pos + size], pos + size
        else:
            raise AssertionError(f"unexpected wire type {wire_type}")
        result.setdefault(number, []).append(value)
    return result


def _blob(offer):
    return _fields(zlib.decompress(plistlib.loads(offer)["avcMediaStreamNegotiatorMediaBlob"]))


def _desktop_caps(blob):
    return [f[2][0] for entry in blob[9]
            if (f := _fields(entry)).get(3, [0])[0] & 0x40000]


@pytest.mark.parametrize("case", _LEGACY["cases"])
def test_default_wire_bytes_match_pre_refactor_capture(monkeypatch, case):
    for name in ("ISS_VIDEO_CODEC", "ISS_LTRP", "ISS_TILES_PER_FRAME", "ISS_AUDIO_F4"):
        monkeypatch.delenv(name, raising=False)
    for name, value in case["env"].items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(offers, "__version__", _LEGACY["version"])
    for mode in (7, 8):
        data = offers._build_mediablob(
            mode, _LEGACY["session_id"], _LEGACY["timestamp"],
            audio_enabled=case["audio_enabled"],
        )
        assert hashlib.sha256(data).hexdigest() == case["sha256"][str(mode)]


def test_explicit_config_reaches_both_compressed_plists_and_overrides_environment(monkeypatch):
    monkeypatch.setenv("ISS_VIDEO_CODEC", "avc")
    monkeypatch.setenv("ISS_LTRP", "1")
    monkeypatch.setenv("ISS_TILES_PER_FRAME", "4")
    config = MediaOfferConfig().set_video_bitrate(30_000_000)
    config.video.codec = "hevc"
    config.video.tiles_per_frame = 1
    config.video.ltrp_enabled = False
    config.video.allow_rtcp_feedback = True
    config.video.pixel_formats = PixelFormat.YUV420_8_FULL
    config.video.fec_enabled = False
    config.video.rtx_enabled = True
    config.video.custom_width = 1920
    config.video.custom_height = 1080
    config.allow_dynamic_max_bitrate = False
    for payload in config.video.payloads:
        for rule in payload.rules:
            rule.formats = ScreenFormat.HD1080_60
            rule.preferred_format = ScreenFormat.HD1080_60

    video, audio = offers.create_offers(config=config, audio_enabled=False)
    v, a = _blob(video), _blob(audio)
    assert _desktop_caps(v) == _desktop_caps(a) == [30_000_000]
    assert v[1] == [0]
    screen = _fields(v[5][0])
    assert screen[2] == [1]  # RTCP remains independently enabled.
    assert screen[4:5] if False else screen[4] == [1920]
    assert screen[5] == [1080]
    assert screen[6] == [1] and screen[7] == [0]
    assert screen[8] == [1] and screen[9] == [1]
    assert screen[10] == [0] and screen[11] == [1]
    assert 13 not in screen  # None omits a capability; False explicitly writes 0.
    assert len(screen[3]) == 1
    payload = _fields(screen[3][0])
    assert payload[1] == [100]
    assert b"LTR;" not in payload[3][0]
    for rule in payload[2]:
        assert _fields(rule)[3] == _fields(rule)[4] == [int(ScreenFormat.HD1080_60)]
    assert _fields(a[3][0])[4] == [1000]


def test_fresh_stream_identity_is_not_stored_in_editable_config(monkeypatch):
    ids = iter((1, 2, 3, 4))
    monkeypatch.setattr(offers.secrets, "randbits", lambda _: next(ids))
    config = MediaOfferConfig()
    first_video, first_audio = offers.create_offers(config=config)
    second_video, second_audio = offers.create_offers(config=config)
    assert [offers.extract_offer_ssrc(o, is_video=v) for o, v in (
        (first_video, True), (first_audio, False),
        (second_video, True), (second_audio, False),
    )] == [1, 2, 3, 4]
    assert plistlib.loads(first_video)["avcMediaStreamOptionCallID"] != plistlib.loads(second_video)["avcMediaStreamOptionCallID"]


def test_bitrate_setter_splits_shared_selectors_and_removes_conflicting_caps():
    config = MediaOfferConfig(bandwidth=[
        BandwidthSettings(1, 60_000_000, 0x41000),
        BandwidthSettings(0, 40_000_000, 0x40000),
    ])
    config.set_video_bitrate(30_000_000)
    assert config.bandwidth == [
        BandwidthSettings(1, 60_000_000, 0x1000),
        BandwidthSettings(0, 30_000_000, 0x40000),
    ]
    config.set_video_bitrate(10_000_000)  # Encoding-valid; not a claimed server range.
    assert config.bandwidth[1].max_bandwidth_bps == 10_000_000


def test_known_bandwidth_modes_are_split_into_base_and_extension_fields():
    entry = BandwidthSettings.for_mode(
        BandwidthMode.SCREEN_WIFI | BandwidthMode.LOW_LATENCY_SCREEN_WIRED, 20_000_000)
    assert _fields(entry.to_protobuf()) == {1: [0x10], 2: [20_000_000], 3: [0x40000]}
    with pytest.raises(ValueError, match="unknown selector"):
        BandwidthSettings.for_mode(BandwidthMode(0x80000), 20_000_000)


def test_config_instances_do_not_share_nested_defaults():
    first, second = MediaOfferConfig(), MediaOfferConfig()
    first.video.payloads[0].rules[0].formats = ScreenFormat.HD1080_30
    first.bandwidth[0].max_bandwidth_bps = 1
    assert second.video.payloads[0].rules[0].formats == ScreenFormat.LEGACY
    assert second.bandwidth[0].max_bandwidth_bps == 40_000_000


@pytest.mark.parametrize("value", [-1, 0, 1 << 32, True, 2.5])
def test_bitrate_setter_rejects_invalid_values(value):
    with pytest.raises(ValueError, match="video bitrate"):
        MediaOfferConfig().set_video_bitrate(value)


@pytest.mark.parametrize("name,value", [
    ("tiles_per_frame", 0), ("tiles_per_frame", -1),
    ("pixel_formats", 0x40), ("hdr_modes", 0x10),
    ("fec_enabled", 1), ("codec", "vp9"), ("custom_width", 1 << 32),
])
def test_invalid_video_settings_fail_before_serialization(name, value):
    config = MediaOfferConfig()
    setattr(config.video, name, value)
    with pytest.raises(ValueError):
        config.validate()


def test_avc_never_advertises_unsupported_ltrp():
    config = MediaOfferConfig()
    config.video.codec = "avc"
    screen = _fields(config.video.to_protobuf(1))
    assert screen[6] == [1] and screen[7] == [0]
    assert [_fields(p)[1][0] for p in screen[3]] == [123]
    assert all(b"LTR;" not in _fields(p)[3][0] for p in screen[3])
    config.video.ltrp_enabled = True
    with pytest.raises(ValueError, match="LTRP"):
        config.validate()


def test_explicit_audio_capability_override_preserves_legacy_precedence():
    config = MediaOfferConfig()
    config.audio.payload_flags = 12345
    _, audio = offers.create_offers(config=config, audio_enabled=False)
    assert _fields(_blob(audio)[3][0])[4] == [12345]


def test_session_sends_snapshot_and_aligns_decoder_and_ltrp(monkeypatch):
    monkeypatch.setenv("ISS_VIDEO_CODEC", "hevc")
    monkeypatch.setenv("ISS_TILES_PER_FRAME", "4")
    config = MediaOfferConfig().set_video_bitrate(30_000_000)
    config.video.codec = "avc"
    session = Session(SessionConfig(
        host="example.invalid", username="user", password="unused", media_offer=config,
    ))
    config.set_video_bitrate(40_000_000)
    config.video.codec = "hevc"
    assert session._video_codec == "avc" and session._ltr_enabled is False
    assert session.num_tiles == 1

    class CapturedHandshake(Exception):
        pass

    def capture(*args, **kwargs):
        v = _blob(kwargs["video_offer"])
        assert _desktop_caps(v) == [30_000_000]
        assert _fields(v[5][0])[6] == [1]
        raise CapturedHandshake

    monkeypatch.setattr(session_module, "connect_and_negotiate", capture)
    with pytest.raises(CapturedHandshake):
        session._negotiate_and_burst(session._config)


def test_resize_fallback_keeps_configured_offer(monkeypatch):
    config = MediaOfferConfig().set_video_bitrate(30_000_000)
    session = Session(SessionConfig(
        host="example.invalid", username="user", password="unused", media_offer=config,
    ))
    sent = []
    session._negotiation = SimpleNamespace(
        keys=object(), sock=object(),
        cipher=SimpleNamespace(encrypt_and_send=lambda sock, data: sent.append(data)),
    )

    def capture(audio, video, keys, **kwargs):
        assert _desktop_caps(_blob(video)) == [30_000_000]
        return b"captured"

    monkeypatch.setattr("isharescreen.proxy.protocol.negotiation.build_0x1c", capture)
    session._schedule_post_layout_arm()
    assert sent == [b"captured"]


def test_srtp_grouping_accepts_offered_tile_count_without_global_environment(monkeypatch):
    monkeypatch.setenv("ISS_TILES_PER_FRAME", "4")
    decryptor = SRTPDecryptor(bytes(32), bytes(14))
    decryptor._counts.update({100: 10, 101: 20, 102: 30, 103: 40})
    assert decryptor.get_primary_ssrc_group(group_size=1) == {103}
    assert decryptor.get_primary_ssrc_group(group_size=4) == {100, 101, 102, 103}
