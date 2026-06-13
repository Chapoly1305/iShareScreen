from __future__ import annotations

import struct

from Crypto.Cipher import AES  # type: ignore[import-untyped]
from .mode_adaptive import ADAPTIVE_RECT_ENCODING_NAMES, is_adaptive_encoding, parse_adaptive_rect, parse_media_stream_options
from .mode_rfb import RFB_RECT_ENCODING_NAMES, is_rfb_encoding, parse_full_quality_rect


def be_u16(data: bytes, off: int) -> int | None:
    if off + 2 > len(data):
        return None
    return struct.unpack_from(">H", data, off)[0]


def be_u32(data: bytes, off: int) -> int | None:
    if off + 4 > len(data):
        return None
    return struct.unpack_from(">I", data, off)[0]


def be_u64(data: bytes, off: int) -> int | None:
    if off + 8 > len(data):
        return None
    return struct.unpack_from(">Q", data, off)[0]


def be_f32(data: bytes, off: int) -> float | None:
    if off + 4 > len(data):
        return None
    return struct.unpack_from(">f", data, off)[0]


def be_f64(data: bytes, off: int) -> float | None:
    if off + 8 > len(data):
        return None
    return struct.unpack_from(">d", data, off)[0]


def be_i32(data: bytes, off: int) -> int | None:
    if off + 4 > len(data):
        return None
    return struct.unpack_from(">i", data, off)[0]


def keysym_name(value: int | None) -> str | None:
    if value is None:
        return None
    named = {
        0xFF0D: "Return",
        0xFFE1: "Shift_L",
    }
    if value in named:
        return named[value]
    if 0x20 <= value <= 0x7E:
        return chr(value)
    return None


def parse_len_prefixed_string(data: bytes, off: int) -> tuple[str | None, int | None]:
    n = be_u16(data, off)
    if n is None or off + 2 + n > len(data):
        return None, None
    raw = data[off + 2 : off + 2 + n]
    return raw.decode("utf-8", errors="replace").rstrip("\x00"), off + 2 + n


def guess_client_message_type(msg_id: int | None) -> str:
    m = {
        0: "SetPixelFormat",
        2: "SetEncodings",
        3: "FramebufferUpdateRequest",
        4: "KeyEvent",
        5: "PointerEvent",
        6: "ClientCutText",
        8: "AppleScaleFactorMessage",
        9: "AutoFrameBufferUpdate",
        10: "SetModeMessage",
        13: "SetDisplayMessage",
        16: "EncryptedInputEvent",
        18: "SetEncryptionMessage",
        21: "AutoPasteboardCommand",
        26: "KeyboardSourceShare",
        28: "MediaStreamOptions",
        29: "SetDisplayConfiguration",
        33: "ViewerInfo",
    }
    if msg_id is None:
        return "client_empty"
    return m.get(msg_id, f"client_extension_0x{msg_id:02x}")


def guess_server_message_type(msg_id: int | None) -> str:
    m = {
        0: "FramebufferUpdate",
        1: "SetColorMapEntries",
        2: "Bell",
        3: "ServerCutText",
        28: "MediaStreamOptions",
        150: "server_extension_0x96",
    }
    if msg_id is None:
        return "server_empty"
    return m.get(msg_id, f"server_extension_0x{msg_id:02x}")


def rectangle_encoding_name(enc: int | None) -> str:
    if enc is None:
        return "unknown"
    if enc in ADAPTIVE_RECT_ENCODING_NAMES:
        return ADAPTIVE_RECT_ENCODING_NAMES[enc]
    if enc in RFB_RECT_ENCODING_NAMES:
        return RFB_RECT_ENCODING_NAMES[enc]
    return f"rect_0x{enc:08x}"


def encoding_mode_name(enc: int | None) -> str:
    if is_adaptive_encoding(enc):
        return "adaptive_media"
    if is_rfb_encoding(enc):
        return "full_quality_rfb"
    return "unknown"


def _is_valid_framebuffer_update_header(body: bytes) -> bool:
    """RFC 6143 §7.6.1: FramebufferUpdate header must have zero padding and sane rect count."""
    if len(body) < 4:
        return False
    padding = body[1]
    rect_count = be_u16(body, 2)
    # padding must be zero; rect_count must be non-zero and not absurd
    if padding != 0:
        return False
    if rect_count is None or rect_count == 0 or rect_count > 5000:
        return False
    # if we have rect data, check that dimensions are within u16 range
    if len(body) >= 16:
        x = be_u16(body, 4)
        y = be_u16(body, 6)
        w = be_u16(body, 8)
        h = be_u16(body, 10)
        # each dimension must fit in u16; 0×0 is valid for pseudo-rects
        if x is None or y is None or w is None or h is None:
            return False
    return True


def parse_framebuffer_update(body: bytes) -> dict[str, object]:
    out: dict[str, object] = {
        "rfb_msg": body[0] if body else None,
        "rect_count": be_u16(body, 2),
    }
    if not _is_valid_framebuffer_update_header(body):
        out["_invalid_header"] = True
        out["_padding_byte"] = body[1] if len(body) > 1 else None
        return out
    if len(body) < 16:
        return out
    enc = be_u32(body, 12)
    rect: dict[str, object] = {
        "x": be_u16(body, 4),
        "y": be_u16(body, 6),
        "width": be_u16(body, 8),
        "height": be_u16(body, 10),
        "encoding": enc,
        "encoding_name": rectangle_encoding_name(enc),
        "mode_family": encoding_mode_name(enc),
    }
    if is_adaptive_encoding(enc):
        parse_adaptive_rect(body, rect)
    else:
        parse_full_quality_rect(body, rect)
    out["first_rect"] = rect
    return out


def parse_set_pixel_format(body: bytes) -> dict[str, object]:
    return {
        "message": "SetPixelFormat",
        "length": len(body),
        "bits_per_pixel": body[4] if len(body) > 4 else None,
        "depth": body[5] if len(body) > 5 else None,
        "big_endian": body[6] if len(body) > 6 else None,
        "true_color": body[7] if len(body) > 7 else None,
        "red_max": be_u16(body, 8),
        "green_max": be_u16(body, 10),
        "blue_max": be_u16(body, 12),
        "red_shift": body[14] if len(body) > 14 else None,
        "green_shift": body[15] if len(body) > 15 else None,
        "blue_shift": body[16] if len(body) > 16 else None,
    }


def parse_set_encodings(body: bytes) -> dict[str, object]:
    count = be_u16(body, 2)
    encodings: list[dict[str, object]] = []
    if count is not None:
        off = 4
        for _ in range(count):
            enc_u32 = be_u32(body, off)
            enc_i32 = be_i32(body, off)
            if enc_u32 is None or enc_i32 is None:
                break
            encodings.append(
                {
                    "signed": enc_i32,
                    "hex": f"0x{enc_u32:08x}",
                    "name": rectangle_encoding_name(enc_u32),
                    "mode_family": encoding_mode_name(enc_u32),
                }
            )
            off += 4
    return {"encoding_count": count, "encodings": encodings}


def parse_framebuffer_update_request(body: bytes) -> dict[str, object]:
    return {
        "incremental": body[1] if len(body) > 1 else None,
        "x": be_u16(body, 2),
        "y": be_u16(body, 4),
        "width": be_u16(body, 6),
        "height": be_u16(body, 8),
    }


def parse_auto_framebuffer_update(body: bytes) -> dict[str, object]:
    interval_ms = be_u32(body, 4)
    return {
        "enable": be_u16(body, 2),
        "interval_ms": interval_ms,
        "interval_kind": "sentinel" if interval_ms == 0xFFFFFFFF else "explicit",
        "x": be_u16(body, 8),
        "y": be_u16(body, 10),
        "width": be_u16(body, 12),
        "height": be_u16(body, 14),
    }


def parse_auto_pasteboard_command(body: bytes) -> dict[str, object]:
    return {"selector": be_u16(body, 2)}


def parse_set_display_message(body: bytes) -> dict[str, object]:
    return {"combine_all": body[1] if len(body) > 1 else None, "target": be_u32(body, 4)}


def parse_set_mode_message(body: bytes) -> dict[str, object]:
    return {"mode": be_u16(body, 2), "raw_hex": body.hex()}


def parse_client_extension_0x10(body: bytes, ecb_key_hex: str | None = None) -> dict[str, object]:
    out: dict[str, object] = {
        "subtype": body[1] if len(body) > 1 else None,
        "encrypted_payload_hex": body[2:].hex(),
        "writer": "_RFBPostX11KeyAndMouseCore",
        "payload_ciphertext_len": max(0, len(body) - 2),
        "note": (
            "bytes 2..17 are encrypted in place before send; "
            "subtype 3 is used when keyboard-source-share state flag f05 is set, "
            "otherwise subtype 1"
        ),
    }
    if len(body) != 18 or ecb_key_hex is None:
        return out

    payload_ciphertext = body[2:18]
    payload_plain = AES.new(bytes.fromhex(ecb_key_hex), AES.MODE_ECB).decrypt(payload_ciphertext)
    subtype = body[1]
    out["decryption_key_source"] = "current_client_transport_key"
    out["decrypted_payload_hex"] = payload_plain.hex()
    if subtype == 1:
        keysym = be_u32(payload_plain, 2)
        out["subtype_name"] = "LegacyX11KeyEvent"
        out["layout"] = {
            "byte_0": "event_marker",
            "byte_1": "key_state",
            "bytes_2_5_be32": "keysym_be32",
            "bytes_6_9_be32": "event_delta_be32",
            "bytes_10_11_be16": "unknown_zero_be16",
            "bytes_12_13_be16": "keyboard_type_be16",
            "bytes_14_15_be16": "local_keycode_be16",
        }
        out["decrypted"] = {
            "event_marker": payload_plain[0],
            "key_state": payload_plain[1],
            "key_state_name": "down" if payload_plain[1] == 1 else "up" if payload_plain[1] == 0 else None,
            "keysym_be32": keysym,
            "keysym_name": keysym_name(keysym),
            "event_delta_be32": be_u32(payload_plain, 6),
            "unknown_zero_be16": be_u16(payload_plain, 10),
            "keyboard_type_be16": be_u16(payload_plain, 12),
            "local_keycode_be16": be_u16(payload_plain, 14),
        }
        out["note"] = (
            "subtype 1 is the legacy encrypted X11 key-event path; BN proves the "
            "writer is _RFBPostX11KeyAndMouseCore, keysym comes from the X11 key "
            "mapping path, bytes 6..9 carry the big-endian event delta from "
            "_UDPPacketTime(), bytes 12..13 carry the keyboard type, and bytes "
            "14..15 carry the original local keycode"
        )
        return out

    if subtype == 3:
        out["subtype_name"] = "LegacyEncryptedMouseEvent"
        out["layout"] = {
            "bytes_0_1_be16": "unknown_zero_be16",
            "bytes_2_5_be32": "legacy_arg4_be32",
            "bytes_6_9_be32": "event_delta_be32",
            "byte_10": "event_marker_ff",
            "byte_11": "button_mask",
            "bytes_12_13_be16": "framebuffer_x_be16",
            "bytes_14_15_be16": "framebuffer_y_be16",
        }
        out["decrypted"] = {
            "unknown_zero_be16": be_u16(payload_plain, 0),
            "legacy_arg4_be32": be_u32(payload_plain, 2),
            "event_delta_be32": be_u32(payload_plain, 6),
            "event_marker_ff": payload_plain[10],
            "button_mask": payload_plain[11],
            "framebuffer_x_be16": be_u16(payload_plain, 12),
            "framebuffer_y_be16": be_u16(payload_plain, 14),
        }
        out["note"] = (
            "subtype 3 is the legacy encrypted mouse-event path from "
            "_RFBPostMouseEventWithClickCountCore into _RFBPostX11KeyAndMouseCore; "
            "bytes 6..9 carry the big-endian event delta from _UDPPacketTime(), "
            "byte 11 carries the remapped button mask, and bytes 12..15 carry "
            "the big-endian framebuffer x/y coordinates after the native scale "
            "and clamp logic; in the observed native trace bytes 0..5 are zero "
            "for ordinary move events and legacy_arg4_be32 remains zero"
        )
        return out

    out["layout"] = {
        "byte_0": "event_prefix_or_flags",
        "byte_1": "event_code_or_flags",
        "bytes_2_5_be32": "event_value_be32",
        "bytes_6_9_be32": "event_delta_be32",
        "byte_10": "tail_prefix_or_flags",
        "byte_11": "tail_state_or_button",
        "bytes_12_13_be16": "coord_x_be16",
        "bytes_14_15_be16": "coord_y_be16",
    }
    out["decrypted"] = {
        "event_prefix_or_flags": payload_plain[0],
        "event_code_or_flags": payload_plain[1],
        "event_value_be32": be_u32(payload_plain, 2),
        "event_delta_be32": be_u32(payload_plain, 6),
        "tail_prefix_or_flags": payload_plain[10],
        "tail_state_or_button": payload_plain[11],
        "coord_x_be16": be_u16(payload_plain, 12),
        "coord_y_be16": be_u16(payload_plain, 14),
    }
    out["subtype_name"] = "EncryptedInputEventSubtype"
    out["note"] = (
        "bytes 2..17 are AES-ECB-decrypted with the current client transport key; "
        "BN proves the writer is _RFBPostX11KeyAndMouseCore and the 16-byte block "
        "is encrypted in place via the AESKeyECBSend context at connection+0x978"
    )
    return out


def parse_set_encryption_message(body: bytes) -> dict[str, object]:
    if len(body) >= 12:
        return {
            "form": "command_1_long",
            "command": be_u16(body, 4),
            "method_count": be_u16(body, 6),
            "method": be_u32(body, 8),
            "raw_hex": body.hex(),
        }
    if len(body) >= 8:
        return {
            "form": "short_followup",
            "command": be_u16(body, 2),
            "value": be_u16(body, 4),
            "raw_hex": body.hex(),
        }
    return {"form": "truncated", "raw_hex": body.hex()}


def parse_scale_factor_message(body: bytes) -> dict[str, object]:
    scale = be_f64(body, 2)
    return {
        "scale": scale,
        "flags_be16_at_1": be_u16(body, 1),
        "raw_hex": body.hex(),
        "note": "strong-inference: native 0x08 carries a BE-double-like scale value",
    }


def _bitmap_set_bits(bitmap: bytes) -> list[int]:
    bits: list[int] = []
    for byte_index, value in enumerate(bitmap):
        for bit in range(8):
            if value & (1 << (7 - bit)):
                bits.append(byte_index * 8 + bit)
    return bits


def parse_viewer_info(body: bytes) -> dict[str, object]:
    payload = body[2:]
    if len(payload) < 64:
        return {"raw_hex": body.hex()}
    words: list[int] = []
    off = 0
    while off + 4 <= len(payload) - 32:
        value = be_u32(payload, off)
        if value is None:
            break
        words.append(value)
        off += 4
    bitmap = payload[-32:]
    return {
        "reserved_after_msg_id": body[1] if len(body) > 1 else None,
        "viewer_app": words[0] if len(words) > 0 else None,
        "viewer_version_major": words[1] if len(words) > 1 else None,
        "viewer_version_minor": words[2] if len(words) > 2 else None,
        "viewer_version_patch": words[3] if len(words) > 3 else None,
        "system_version_major": words[4] if len(words) > 4 else None,
        "system_version_minor": words[5] if len(words) > 5 else None,
        "system_version_patch": words[6] if len(words) > 6 else None,
        "viewer_info_version": words[7] if len(words) > 7 else None,
        "header_words": words,
        "command_bitmap_hex": bitmap.hex(),
        "command_ids_set": _bitmap_set_bits(bitmap),
        "raw_hex": body.hex(),
    }


def parse_set_display_configuration(body: bytes) -> dict[str, object]:
    out: dict[str, object] = {
        "message_size": be_u16(body, 2),
        "version": be_u16(body, 4),
        "display_count": be_u16(body, 6),
        "flags": be_u32(body, 8),
        "first_display_info_size": be_u16(body, 12),
    }
    if len(body) <= 12:
        out["raw_tail_hex"] = ""
        return out

    descriptor_offset = 12
    descriptor_size = be_u16(body, descriptor_offset)
    descriptor: dict[str, object] = {
        "descriptor_offset": descriptor_offset,
        "display_info_size": descriptor_size,
    }

    if descriptor_size is None or descriptor_offset + descriptor_size > len(body):
        descriptor["raw_hex"] = body[descriptor_offset:].hex()
        out["first_display"] = descriptor
        return out

    descriptor_end = descriptor_offset + descriptor_size
    descriptor_body = body[descriptor_offset:descriptor_end]
    mode_count = be_u16(body, descriptor_offset + 0x9A)
    mode_table_offset = descriptor_offset + 0x9C

    if descriptor_size >= 0x9C:
        descriptor.update(
            {
                "display_info_region_hex": descriptor_body[2:0x7A].hex(),
                "display_flags": be_u32(body, descriptor_offset + 0x7A),
                "display_type": be_u32(body, descriptor_offset + 0x7E),
                "physical_width_mm_f32": be_f32(body, descriptor_offset + 0x82),
                "physical_height_mm_f32": be_f32(body, descriptor_offset + 0x86),
                "max_width_u32": be_u32(body, descriptor_offset + 0x8A),
                "max_height_u32": be_u32(body, descriptor_offset + 0x8E),
                "current_mode_index": be_u16(body, descriptor_offset + 0x92),
                "preferred_mode_index": be_u16(body, descriptor_offset + 0x94),
                "unknown_u32": be_u32(body, descriptor_offset + 0x96),
                "mode_count": mode_count,
            }
        )

        modes: list[dict[str, object]] = []
        if mode_count is not None:
            for index in range(mode_count):
                mode_offset = mode_table_offset + index * 0x1C
                if mode_offset + 0x1C > descriptor_end:
                    break
                modes.append(
                    {
                        "index": index,
                        "offset": mode_offset - descriptor_offset,
                        "width": be_u32(body, mode_offset + 0x00),
                        "height": be_u32(body, mode_offset + 0x04),
                        "scaled_width": be_u32(body, mode_offset + 0x08),
                        "scaled_height": be_u32(body, mode_offset + 0x0C),
                        "refresh_rate_f64": be_f64(body, mode_offset + 0x10),
                        "flags": be_u32(body, mode_offset + 0x18),
                    }
                )
        descriptor["modes"] = modes

    if descriptor_end < len(body):
        out["raw_tail_hex"] = body[descriptor_end:].hex()
    else:
        out["raw_tail_hex"] = ""
    out["first_display"] = descriptor
    return out


def parse_client_message(body: bytes, ecb_key_hex: str | None = None) -> tuple[str, dict[str, object]]:
    msg_id = body[0] if body else None
    name = guess_client_message_type(msg_id)
    if msg_id == 0:
        return name, parse_set_pixel_format(body)
    if msg_id == 2:
        return name, parse_set_encodings(body)
    if msg_id == 3:
        return name, parse_framebuffer_update_request(body)
    if msg_id == 8:
        return name, parse_scale_factor_message(body)
    if msg_id == 9:
        return name, parse_auto_framebuffer_update(body)
    if msg_id == 10:
        return name, parse_set_mode_message(body)
    if msg_id == 13:
        return name, parse_set_display_message(body)
    if msg_id == 16:
        return name, parse_client_extension_0x10(body, ecb_key_hex=ecb_key_hex)
    if msg_id == 18:
        return name, parse_set_encryption_message(body)
    if msg_id == 21:
        return name, parse_auto_pasteboard_command(body)
    if msg_id == 28:
        return name, parse_media_stream_options(body)
    if msg_id == 29:
        return name, parse_set_display_configuration(body)
    if msg_id == 33:
        return name, parse_viewer_info(body)
    return name, {}


def parse_server_message(body: bytes) -> tuple[str, dict[str, object]]:
    msg_id = body[0] if body else None
    name = guess_server_message_type(msg_id)
    if msg_id == 0:
        parsed = parse_framebuffer_update(body)
        if parsed.get("_invalid_header"):
            # First byte 0x00 is a coincidental match — this is likely a
            # media data block whose first byte happens to be zero (P ≈ 1/256).
            # Reclassify as an opaque blob so it doesn't produce a bogus rect.
            blob_name = "server_media_blob_0x00"
            blob_parsed: dict[str, object] = {
                "raw_len": len(body),
                "direction": "server",
                "note": "Reclassified from FramebufferUpdate: padding=0x%02x (must be 0) rect_count=%d (absurd)"
                        % (parsed.get("_padding_byte", -1), parsed.get("rect_count", -1)),
                "has_zlib_header": b"\x78\xda" in body,
                "has_bplist00": b"bplist00" in body,
            }
            return blob_name, blob_parsed
        return name, parsed
    if msg_id == 28:
        parsed = parse_media_stream_options(body)
        declared_size = parsed.get("message_size_be16")
        parsed["direction"] = "server"
        parsed["declared_size_matches_len_minus_4"] = (
            isinstance(declared_size, int) and declared_size == max(len(body) - 4, 0)
        )
        return name, parsed

    out: dict[str, object] = {
        "raw_len": len(body),
        "direction": "server",
    }
    if len(body) >= 3:
        out["u16_be_1"] = be_u16(body, 1)
    if len(body) >= 5:
        out["u16_be_3"] = be_u16(body, 3)
    if len(body) >= 6:
        out["u32_be_2"] = be_u32(body, 2)
    if len(body) >= 9:
        out["u32_be_5"] = be_u32(body, 5)
    out["has_bplist00"] = b"bplist00" in body
    out["has_zlib_header"] = b"\x78\xda" in body
    return name, out
