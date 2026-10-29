"""Editable schema for Screen Sharing's startup media offer.

Field names and selectors follow the 24G624 static analysis in
docs/analysis/media-negotiation-24G624.md. Validation checks encoding and known
capabilities, not whether a remote encoder will honor a requested value.
The legacy bandwidth list is deliberately retained byte-for-byte by default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from enum import IntEnum, IntFlag

from .protobuf import field_bytes, field_varint


def _uint(name: str, value: int, bits: int = 32, *, minimum: int = 0) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value < 1 << bits:
        raise ValueError(f"{name} must be an integer in {minimum}..{(1 << bits) - 1}")


def _bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def _optional_uint(number: int, name: str, value: int | None) -> bytes:
    if value is None:
        return b""
    _uint(name, value)
    return field_varint(number, value)


def _optional_bool(number: int, name: str, value: bool | None) -> bytes:
    if value is None:
        return b""
    _bool(name, value)
    return field_varint(number, int(value))


class BandwidthMode(IntFlag):
    DEFAULT_WIFI = 0x0001
    DEFAULT_LTE = 0x0002
    FACETIME_WIFI = 0x0004
    FACETIME_LTE = 0x0008
    SCREEN_WIFI = 0x0010
    SCREEN_LTE = 0x0020
    FACETIME_3G = 0x0040
    SCREEN_3G = 0x0080
    DEFAULT_3G = 0x0100
    FACETIME_2G = 0x0200
    SCREEN_2G = 0x0400
    DEFAULT_2G = 0x0800
    IPAD_COMPANION_AWDL = 0x1000
    IPAD_COMPANION_USB = 0x2000
    FACETIME_5G = 0x4000
    REMOTE_CAMERA_AWDL = 0x8000
    REMOTE_CAMERA_USB = 0x10000
    MULTIWAY_SCREEN_WIFI = 0x20000
    LOW_LATENCY_SCREEN_WIRED = 0x40000


class PixelFormat(IntFlag):
    YUV420_8_FULL = 0x01
    YUV420_8_VIDEO = 0x02
    YUV420_10_VIDEO = 0x04
    YUV444_8_FULL = 0x08
    YUV444_8_VIDEO = 0x10
    YUV444_10_FULL = 0x20
    ALL = 0x3F


class HDRMode(IntFlag):
    """Bitmap bits for native modes 0..3, not PQ/HLG identifiers."""

    SDR = 0x01
    MODE_1 = 0x02
    MODE_2 = 0x04
    MODE_3 = 0x08
    ALL = 0x0F


class RuleTransport(IntEnum):
    WIFI = 1
    CELLULAR = 2


class RuleOperation(IntEnum):
    ENCODE = 1
    DECODE = 2


class ScreenFormat(IntFlag):
    """Desktop format/FPS capability bits, not arbitrary display dimensions."""

    XGA_30 = 0x000001       # 1024x768
    XGA_60 = 0x000002
    W1136_H640_30 = 0x000004
    W1136_H640_60 = 0x000008
    W1334_H750_30 = 0x000010
    W1334_H750_60 = 0x000020
    W1664_H1248_30 = 0x000040
    W1664_H1248_60 = 0x000080
    HD1080_30 = 0x000100
    HD1080_60 = 0x000200
    W2208_H1242_30 = 0x000400
    W2208_H1242_60 = 0x000800
    W2048_H1536_30 = 0x001000
    W2048_H1536_60 = 0x002000
    HD720_30 = 0x004000
    HD720_60 = 0x008000
    W1920_H896_30 = 0x010000
    W1920_H896_60 = 0x020000
    W2732_H2048_60 = 0x040000
    W2388_H1668_60 = 0x080000
    W2224_H1668_60 = 0x100000
    W3072_H1728_60 = 0x200000
    UHD4K_60 = 0x400000
    LEGACY = 0x00C3C3
    ALL = 0x7FFFFF


@dataclass
class BandwidthSettings:
    """MediaBlob.F9 entry; raw selectors allow preserving legacy unknown bits."""

    configuration: int
    max_bandwidth_bps: int
    configuration_extension: int | None = None

    @classmethod
    def for_mode(cls, mode: BandwidthMode, max_bandwidth_bps: int) -> BandwidthSettings:
        """Split known mode bits into the correct base/extension wire fields."""
        _uint("bandwidth mode", mode, minimum=1)
        if int(mode) & ~0x7FFFF:
            raise ValueError("bandwidth mode contains unknown selector bits")
        _uint("max_bandwidth_bps", max_bandwidth_bps)
        return cls(int(mode) & 0xFFF, max_bandwidth_bps, int(mode) & 0x7F000 or None)

    def to_protobuf(self) -> bytes:
        # Enum fields are int32; supported bitmaps are nonnegative.
        _uint("configuration", self.configuration, bits=31)
        _uint("max_bandwidth_bps", self.max_bandwidth_bps)
        body = field_varint(1, self.configuration) + field_varint(2, self.max_bandwidth_bps)
        if self.configuration_extension is not None:
            _uint("configuration_extension", self.configuration_extension, bits=31)
            body += field_varint(3, self.configuration_extension)
        return body


def _legacy_bandwidth_settings() -> list[BandwidthSettings]:
    # Preserve existing wire values, including unknown 0x80000/0x100000 bits.
    # F1/F3 are selectors, not audio codec markers or buffer sizes.
    return [BandwidthSettings(*values) for values in (
        (0, 40_000_000, 0x3000), (0, 6_000_000, 0x20000),
        (4074, 0, 0x4000), (16, 4100, None),
        (0, 75_000_000, 0x80000), (0, 20_000_000, 0x18000),
        (4, 6500, None), (0, 60_000_000, 0x40000),
        (1, 299, None), (0, 100_000_000, 0x100000),
    )]


@dataclass
class VideoRuleCollection:
    transport: RuleTransport = RuleTransport.WIFI
    operation: RuleOperation = RuleOperation.DECODE
    formats: ScreenFormat = ScreenFormat.LEGACY
    preferred_format: ScreenFormat = ScreenFormat(0)
    formats_ext1: int | None = None
    preferred_format_ext1: int | None = None

    def to_protobuf(self) -> bytes:
        _uint("rule transport", self.transport)
        _uint("rule operation", self.operation)
        if self.transport not in (RuleTransport.WIFI, RuleTransport.CELLULAR):
            raise ValueError("rule transport must be WIFI (1) or CELLULAR (2)")
        if self.operation not in (RuleOperation.ENCODE, RuleOperation.DECODE):
            raise ValueError("rule operation must be ENCODE (1) or DECODE (2)")
        _uint("formats", self.formats)
        _uint("preferred_format", self.preferred_format)
        if int(self.formats) & ~int(ScreenFormat.ALL):
            raise ValueError("formats contains unknown desktop format bits")
        if int(self.preferred_format) & ~int(self.formats):
            raise ValueError("preferred_format must be a subset of formats")
        return (
            field_varint(1, int(self.transport))
            + field_varint(2, int(self.operation))
            + field_varint(3, int(self.formats))
            + field_varint(4, int(self.preferred_format))
            + _optional_uint(5, "formats_ext1", self.formats_ext1)
            + _optional_uint(6, "preferred_format_ext1", self.preferred_format_ext1)
        )


@dataclass
class VideoPayloadSettings:
    payload: int
    rules: list[VideoRuleCollection]
    feature_string: str
    parameter_set: int

    def to_protobuf(self, *, ltrp_enabled: bool) -> bytes:
        _uint("payload", self.payload)
        _uint("parameter_set", self.parameter_set)
        if not self.rules:
            raise ValueError("a video payload needs at least one rule collection")
        features = self.feature_string
        if not ltrp_enabled:
            features = ";".join(token for token in features.split(";") if token != "LTR")
        return (
            field_varint(1, self.payload)
            + b"".join(field_bytes(2, rule.to_protobuf()) for rule in self.rules)
            + field_bytes(3, features.encode("ascii"))
            + field_varint(4, self.parameter_set)
        )


def _rules() -> list[VideoRuleCollection]:
    return [VideoRuleCollection(operation=op) for op in RuleOperation]


def _legacy_video_payloads() -> list[VideoPayloadSettings]:
    # Select by observed OUTPUT, not the codec suggested by the feature string:
    # payload 123 (HEVC-labelled params) yields AVC; payload 100 yields HEVC.
    return [
        VideoPayloadSettings(123, _rules() + _rules(),
            "FLS;MS:-1;LF:-1;LTR;CABAC;POS:0;EOD:1;HTS:2;RR:3;AR:16/9,5/8;XR:16/9,5/8;", 1),
        VideoPayloadSettings(100, _rules(),
            "FLS;LF:-1;POS:5;EOD:1;HTS:2;RR:3;POSE:4;AR:16/9,5/8;XR:16/9,5/8;", 14),
    ]


@dataclass
class VideoSettings:
    """MediaBlob.F5 screenSettings. None keeps optional fields absent/automatic.

    SSRC is supplied fresh by the offer builder rather than stored here.
    codec is a local bank/decoder selector, not a protobuf field.
    """

    codec: str = "both"
    allow_rtcp_feedback: bool | None = None
    payloads: list[VideoPayloadSettings] = field(default_factory=_legacy_video_payloads)
    custom_width: int | None = None
    custom_height: int | None = None
    tiles_per_frame: int | None = None
    ltrp_enabled: bool | None = None
    pixel_formats: PixelFormat = PixelFormat.ALL
    hdr_modes: HDRMode = HDRMode.SDR
    fec_enabled: bool | None = None
    rtx_enabled: bool | None = None
    black_frame_on_clear_screen_enabled: bool | None = True
    foveation_supported: bool | None = None

    @property
    def effective_tiles_per_frame(self) -> int:
        return self.tiles_per_frame if self.tiles_per_frame is not None else (1 if self.codec == "avc" else 4)

    @property
    def effective_ltrp_enabled(self) -> bool:
        return self.ltrp_enabled if self.ltrp_enabled is not None else self.codec != "avc"

    def to_protobuf(self, session_id: int) -> bytes:
        if self.codec not in ("both", "hevc", "avc"):
            raise ValueError("video.codec must be 'both', 'hevc', or 'avc'")
        _uint("SSRC", session_id)
        _uint("tiles_per_frame", self.effective_tiles_per_frame, minimum=1)
        ltrp = self.effective_ltrp_enabled
        _bool("ltrp_enabled", ltrp)
        if self.codec == "avc" and ltrp:
            raise ValueError("AVC does not support the client's LTRP acknowledgment path")
        rtcp = ltrp if self.allow_rtcp_feedback is None else self.allow_rtcp_feedback
        _bool("allow_rtcp_feedback", rtcp)
        _uint("pixel_formats", self.pixel_formats)
        _uint("hdr_modes", self.hdr_modes)
        if int(self.pixel_formats) & ~int(PixelFormat.ALL):
            raise ValueError("pixel_formats contains unknown capability bits")
        if not self.pixel_formats:
            raise ValueError("pixel_formats must advertise at least one format")
        if self.codec == "avc" and not (self.pixel_formats & (
            PixelFormat.YUV420_8_FULL | PixelFormat.YUV420_8_VIDEO
        )):
            raise ValueError("AVC requires an 8-bit YUV420 pixel format")
        if int(self.hdr_modes) & ~int(HDRMode.ALL):
            raise ValueError("hdr_modes contains unknown capability bits")
        payloads = [p for p in self.payloads if self.codec == "both" or p.payload == (123 if self.codec == "avc" else 100)]
        if not payloads:
            raise ValueError("no payload bank matches video.codec")
        return (
            field_varint(1, session_id) + field_varint(2, int(rtcp))
            + b"".join(field_bytes(3, p.to_protobuf(ltrp_enabled=ltrp)) for p in payloads)
            + _optional_uint(4, "custom_width", self.custom_width)
            + _optional_uint(5, "custom_height", self.custom_height)
            + field_varint(6, self.effective_tiles_per_frame)
            + field_varint(7, int(ltrp))
            + field_varint(8, int(self.pixel_formats)) + field_varint(9, int(self.hdr_modes))
            + _optional_bool(10, "fec_enabled", self.fec_enabled)
            + _optional_bool(11, "rtx_enabled", self.rtx_enabled)
            + _optional_bool(12, "black_frame_on_clear_screen_enabled", self.black_frame_on_clear_screen_enabled)
            + _optional_bool(13, "foveation_supported", self.foveation_supported)
        )


@dataclass
class AudioSettings:
    """MediaBlob.F3; payload_flags is a capability mask, NOT a bitrate.

    None chooses the existing audio-on/off masks (24191/1000). An explicit
    payload_flags override wins, matching legacy ISS_AUDIO_F4 behavior.
    """

    audio_unit_model: int = 0
    support_flags: int = 0
    payload_flags: int | None = None
    secondary_flags: int = 0
    use_sbr: bool = False

    def to_protobuf(self, session_id: int, *, enabled: bool = True) -> bytes:
        _uint("SSRC", session_id)
        _bool("audio enabled", enabled)
        payload_flags = self.payload_flags if self.payload_flags is not None else (24191 if enabled else 1000)
        values = (session_id, self.audio_unit_model, self.support_flags, payload_flags, self.secondary_flags)
        for number, value in enumerate(values, 1):
            _uint(f"audio F{number}", value)
        _bool("use_sbr", self.use_sbr)
        return b"".join(field_varint(n, v) for n, v in enumerate(values, 1)) + field_varint(6, int(self.use_sbr))


@dataclass
class MediaOfferConfig:
    """Session-independent editable values; per-offer SSRC/time/UUID stay fresh.

    Direct construction is independent of environment variables. Use
    from_environment() explicitly to start with the existing CLI/env choices.
    Set values before creating a Session; its configuration is snapshotted.
    """

    video: VideoSettings = field(default_factory=VideoSettings)
    audio: AudioSettings = field(default_factory=AudioSettings)
    bandwidth: list[BandwidthSettings] = field(default_factory=_legacy_bandwidth_settings)
    allow_dynamic_max_bitrate: bool = True
    allows_contents_change_with_aspect_preservation: bool = True
    baseband_codec_sample_rate: int = 0
    blob_version: int = 2
    media_control_info_version: int = 0
    access_network_type: int = 1

    @classmethod
    def from_environment(cls) -> MediaOfferConfig:
        codec = os.environ.get("ISS_VIDEO_CODEC", "both").lower()
        if codec not in ("both", "hevc", "avc"):
            codec = "both"
        config = cls(video=VideoSettings(codec=codec))
        config.video.ltrp_enabled = codec != "avc" and os.environ.get("ISS_LTRP", "1") != "0"
        if "ISS_TILES_PER_FRAME" in os.environ:
            try:
                config.video.tiles_per_frame = max(1, int(os.environ["ISS_TILES_PER_FRAME"]))
            except ValueError:
                pass
        if "ISS_AUDIO_F4" in os.environ:
            config.audio.payload_flags = int(os.environ["ISS_AUDIO_F4"])
        return config

    def set_video_bitrate(self, bps: int) -> MediaOfferConfig:
        """Declare a low-latency desktop bandwidth cap, not constant bitrate.

        Split grouped selectors so other modes retain their original caps;
        replace all matching entries to avoid conflicting duplicate requests.
        """
        _uint("video bitrate", bps, minimum=1)
        selector = int(BandwidthMode.LOW_LATENCY_SCREEN_WIRED)
        entries = []
        inserted = False
        for entry in self.bandwidth:
            extension = entry.configuration_extension or 0
            if extension & selector:
                remaining = extension & ~selector
                if remaining or entry.configuration:
                    entries.append(replace(entry, configuration_extension=remaining or None))
                if not inserted:
                    entries.append(BandwidthSettings.for_mode(BandwidthMode.LOW_LATENCY_SCREEN_WIRED, bps))
                    inserted = True
            else:
                entries.append(entry)
        if not inserted:
            entries.append(BandwidthSettings.for_mode(BandwidthMode.LOW_LATENCY_SCREEN_WIRED, bps))
        self.bandwidth = entries
        return self

    def validate(self) -> None:
        """Check all editable fields before opening a connection."""
        self.video.to_protobuf(1)
        self.audio.to_protobuf(1)
        for entry in self.bandwidth:
            entry.to_protobuf()
        for name in ("allow_dynamic_max_bitrate", "allows_contents_change_with_aspect_preservation"):
            _bool(name, getattr(self, name))
        for name in ("baseband_codec_sample_rate", "blob_version", "media_control_info_version", "access_network_type"):
            _uint(name, getattr(self, name))
