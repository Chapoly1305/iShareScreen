from __future__ import annotations

import plistlib
import struct
import zlib


ADAPTIVE_RECT_ENCODING_NAMES: dict[int, str] = {
    0x000003EA: "RFBMediaStreamMessage3",
    0x000003F2: "RFBMediaStreamMessage1",
    0x000003F3: "RFBMediaStreamMessage2",
}

MEDIA_STREAM_OPTIONS_HEADER_LEN = 36
MEDIA_STREAM_OPTIONS_KEY_LEN = 46
MEDIA_STREAM_OPTIONS_KEYS_PER_STREAM_LEN = MEDIA_STREAM_OPTIONS_KEY_LEN * 2


def be_u16(data: bytes, off: int) -> int | None:
    if off + 2 > len(data):
        return None
    return struct.unpack_from(">H", data, off)[0]


def be_u32(data: bytes, off: int) -> int | None:
    if off + 4 > len(data):
        return None
    return struct.unpack_from(">I", data, off)[0]


def is_adaptive_encoding(enc: int | None) -> bool:
    return enc in ADAPTIVE_RECT_ENCODING_NAMES


def parse_adaptive_rect(body: bytes, rect: dict[str, object]) -> None:
    payload_len = be_u16(body, 16)
    rect["payload_len"] = payload_len
    payload = body[16 : 16 + 2 + payload_len] if payload_len is not None else b""
    rect["payload_prefix_hex"] = payload[:64].hex()

    enc = rect.get("encoding")
    if enc == 0x3F2:
        media_version = be_u16(body, 18)
        media_type = be_u16(body, 20)
        field6 = be_u16(body, 22)
        field8 = be_u16(body, 24)
        field10 = be_u16(body, 26)
        field12 = be_u32(body, 28)
        rect["media_init_version"] = media_version
        rect["media_init_type"] = media_type
        rect["media_init_u16_6"] = field6
        rect["media_init_u16_8"] = field8
        rect["media_init_u16_10"] = field10
        rect["media_init_u32_12"] = field12
        rect["media_init_field6"] = field6
        rect["media_init_field8"] = field8
        rect["media_init_next_field"] = field10
        rect["media_init_field12_u32"] = field12
        if media_version == 1 and media_type == 1:
            rect["media_init_branch"] = "stage1"
            rect["media_init_next_udp_port"] = field10
        elif media_version == 2 and media_type == 2:
            rect["media_init_branch"] = "stage2"
        else:
            rect["media_init_branch"] = "unknown"

    zlib_offset = payload.find(b"\x78\xda")
    if zlib_offset >= 0:
        rect["zlib_offset"] = zlib_offset
        try:
            dec = zlib.decompress(payload[zlib_offset:])
            rect["zlib_decoded_len"] = len(dec)
            rect["zlib_decoded_text_prefix"] = dec[:160].decode("utf-8", errors="replace")
        except Exception as exc:
            rect["zlib_error"] = str(exc)

    plist_offset = payload.find(b"bplist00")
    if plist_offset >= 0:
        rect["bplist_offset"] = plist_offset
        rect["bplist_prefix"] = payload[plist_offset : plist_offset + 64].hex()


def _summarize_plist_value(value: object) -> object:
    if isinstance(value, bytes):
        summary: dict[str, object] = {
            "len": len(value),
            "hex_prefix": value[:24].hex(),
        }
        if value.startswith(b"x\xda"):
            summary["zlib"] = True
            try:
                dec = zlib.decompress(value)
                summary["zlib_decoded_len"] = len(dec)
                try:
                    summary["zlib_decoded_text_prefix"] = dec[:160].decode("utf-8", errors="replace")
                except Exception:
                    pass
            except Exception as exc:
                summary["zlib_error"] = str(exc)
        return summary
    return value


def parse_media_stream_options(body: bytes) -> dict[str, object]:
    flags = be_u32(body, 6)
    offer_flag_names: list[str] = []
    if flags is not None:
        if flags & 0x1:
            offer_flag_names.append("stream1_supports_60fps")
        if flags & 0x2:
            offer_flag_names.append("stream2_supports_60fps")
        if flags & 0x4:
            offer_flag_names.append("do_not_send_cursor")
        if flags & 0x8:
            offer_flag_names.append("apple_remote_desktop_viewer")
        unknown = flags & ~0xF
        if unknown:
            offer_flag_names.append(f"unknown_bits_0x{unknown:08x}")
    declared_size = be_u16(body, 2)
    out: dict[str, object] = {
        "mode_family": "adaptive_media",
        "raw_len": len(body),
        "message_size_be16": declared_size,
        "message_version_be16": be_u16(body, 4),
        "message_flags_be32": flags,
        "message_flag_names": offer_flag_names,
        "audio_offer_len_be16": be_u16(body, 10),
        "video1_offer_len_be16": be_u16(body, 12),
        "video2_offer_len_be16": be_u16(body, 14),
        "session_id_hex": body[20:36].hex() if len(body) >= 36 else None,
    }
    if declared_size is not None:
        out["declared_size_matches_body_minus_4"] = declared_size == max(0, len(body) - 4)

    audio_offer_len = out["audio_offer_len_be16"]
    video1_offer_len = out["video1_offer_len_be16"]
    video2_offer_len = out["video2_offer_len_be16"]
    if isinstance(audio_offer_len, int) and isinstance(video1_offer_len, int) and isinstance(video2_offer_len, int):
        stream_key_blocks: list[dict[str, object]] = []
        audio_offer_offset = MEDIA_STREAM_OPTIONS_HEADER_LEN + MEDIA_STREAM_OPTIONS_KEYS_PER_STREAM_LEN
        video1_offer_offset = audio_offer_offset + audio_offer_len + MEDIA_STREAM_OPTIONS_KEYS_PER_STREAM_LEN
        video2_offer_offset = video1_offer_offset + video1_offer_len + MEDIA_STREAM_OPTIONS_KEYS_PER_STREAM_LEN
        out["audio_offer_offset"] = audio_offer_offset
        out["video1_offer_offset"] = video1_offer_offset
        if video2_offer_len:
            out["video2_offer_offset"] = video2_offer_offset

        segments: list[tuple[str, int, int]] = [
            ("audio_offer", audio_offer_offset, audio_offer_len),
            ("video1_offer", video1_offer_offset, video1_offer_len),
        ]
        if video2_offer_len:
            segments.append(("video2_offer", video2_offer_offset, video2_offer_len))
        out["offer_segment_lengths"] = {name: seg_len for name, _, seg_len in segments}

        stream_layout: list[tuple[str, int, int]] = [
            ("audio", MEDIA_STREAM_OPTIONS_HEADER_LEN, audio_offer_len),
            ("video1", audio_offer_offset + audio_offer_len, video1_offer_len),
        ]
        if video2_offer_len:
            stream_layout.append(("video2", video1_offer_offset + video1_offer_len, video2_offer_len))
        for stream_name, keys_off, offer_len in stream_layout:
            key1 = body[keys_off : keys_off + MEDIA_STREAM_OPTIONS_KEY_LEN]
            key2 = body[
                keys_off + MEDIA_STREAM_OPTIONS_KEY_LEN : keys_off + MEDIA_STREAM_OPTIONS_KEYS_PER_STREAM_LEN
            ]
            if len(key1) != MEDIA_STREAM_OPTIONS_KEY_LEN or len(key2) != MEDIA_STREAM_OPTIONS_KEY_LEN:
                continue
            stream_key_blocks.append(
                {
                    "stream": stream_name,
                    "keys_offset": keys_off,
                    "offer_len": offer_len,
                    "key1_hex": key1.hex(),
                    "key2_hex": key2.hex(),
                    "key1_first30_hex": key1[:30].hex(),
                    "key2_first30_hex": key2[:30].hex(),
                    "key1_last30_hex": key1[-30:].hex(),
                    "key2_last30_hex": key2[-30:].hex(),
                }
            )
        if stream_key_blocks:
            out["stream_key_blocks"] = stream_key_blocks

        parsed_plists: list[dict[str, object]] = []
        valid_plists = 0
        for name, seg_off, seg_len in segments:
            if seg_len <= 0:
                continue
            seg = body[seg_off : seg_off + seg_len]
            if len(seg) != seg_len:
                continue
            entry: dict[str, object] = {
                "segment": name,
                "offset": seg_off,
                "len": seg_len,
            }
            if not seg.startswith(b"bplist00"):
                entry["error"] = "segment_not_bplist"
                parsed_plists.append(entry)
                continue
            try:
                plist_obj = plistlib.loads(seg)
            except Exception as exc:
                entry["error"] = str(exc)
                parsed_plists.append(entry)
                continue
            if not isinstance(plist_obj, dict):
                entry["error"] = "plist_not_dict"
                parsed_plists.append(entry)
                continue
            entry["keys"] = sorted(plist_obj.keys())
            for key, value in plist_obj.items():
                entry[key] = _summarize_plist_value(value)
            parsed_plists.append(entry)
            valid_plists += 1
        if valid_plists > 0:
            out["parsed_plists"] = parsed_plists
            first = next((p for p in parsed_plists if "keys" in p), parsed_plists[0])
            if "avcMediaStreamOptionCallID" in first:
                out["call_id"] = first["avcMediaStreamOptionCallID"]
            if "avcMediaStreamNegotiatorMode" in first:
                out["negotiator_mode"] = first["avcMediaStreamNegotiatorMode"]
            out["note"] = (
                "Adaptive AV Conference media negotiation message; decoded with "
                "fixed native 0x1c layout: 36-byte header + per-stream key pairs "
                "(46+46) + plist offers"
            )
            return out

    positions: list[int] = []
    idx = 0
    while True:
        pos = body.find(b"bplist00", idx)
        if pos < 0:
            break
        positions.append(pos)
        idx = pos + 1
    out["bplist_offsets"] = positions

    parsed_plists = []
    for pos in positions:
        try:
            plist_obj = plistlib.loads(body[pos:])
        except Exception:
            continue
        if isinstance(plist_obj, dict):
            parsed: dict[str, object] = {
                "keys": sorted(plist_obj.keys()),
            }
            for key, value in plist_obj.items():
                parsed[key] = _summarize_plist_value(value)
            parsed_plists.append(parsed)
    out["parsed_plists"] = parsed_plists
    if parsed_plists:
        first = parsed_plists[0]
        if "avcMediaStreamOptionCallID" in first:
            out["call_id"] = first["avcMediaStreamOptionCallID"]
        if "avcMediaStreamNegotiatorMode" in first:
            out["negotiator_mode"] = first["avcMediaStreamNegotiatorMode"]
    out["note"] = (
        "Adaptive AV Conference media negotiation message; fixed header includes "
        "version, flags, audio/video offer lengths, and a 16-byte media session "
        "id before the AVFoundation-style plist payloads"
    )
    return out
