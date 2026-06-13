from __future__ import annotations

import struct


RFB_RECT_ENCODING_NAMES: dict[int, str] = {
    0x00000000: "Raw",
    0x00000006: "Zlib",
    0x00000010: "ZRLE",
    0x0000044C: "ARDPointerRebase",
    0x0000044D: "ARDDisplayLayoutSelector",
    0x0000044F: "EncodeEncryptionInfo",
    0x00000450: "CursorImage",
    0x00000451: "AppleDisplayLayout",
    0x00000453: "VendorKeysymEncoding",
    0x00000455: "KeyboardInputSource",
    0x00000456: "DeviceInfo",
    0xFFFFFF11: "RichCursor",
    0xFFFFFF21: "DesktopSize",
}


def be_u16(data: bytes, off: int) -> int | None:
    if off + 2 > len(data):
        return None
    return struct.unpack_from(">H", data, off)[0]


def be_u32(data: bytes, off: int) -> int | None:
    if off + 4 > len(data):
        return None
    return struct.unpack_from(">I", data, off)[0]


def parse_len_prefixed_string(data: bytes, off: int) -> tuple[str | None, int | None]:
    n = be_u16(data, off)
    if n is None or off + 2 + n > len(data):
        return None, None
    raw = data[off + 2 : off + 2 + n]
    return raw.decode("utf-8", errors="replace").rstrip("\x00"), off + 2 + n


def is_rfb_encoding(enc: int | None) -> bool:
    return enc in RFB_RECT_ENCODING_NAMES


def parse_full_quality_rect(body: bytes, rect: dict[str, object]) -> None:
    enc = rect.get("encoding")
    if enc == 0x00000006:
        rect["compressed_len"] = be_u32(body, 16)
        return
    if enc == 0x450:
        rect["cache_id"] = be_u32(body, 16)
        rect["compressed_len"] = be_u32(body, 20)
        return
    if enc == 0x451:
        rect["version"] = be_u16(body, 18)
        rect["scaled_width"] = be_u16(body, 20)
        rect["scaled_height"] = be_u16(body, 22)
        rect["ui_width"] = be_u16(body, 24)
        rect["ui_height"] = be_u16(body, 26)
        rect["selected_screen"] = be_u32(body, 28)
        rect["flags"] = be_u32(body, 32)
        return
    if enc == 0x453:
        count = be_u16(body, 20)
        rect["value_count"] = count
        values: list[int] = []
        if count is not None:
            off = 22
            for _ in range(count):
                value = be_u32(body, off)
                if value is None:
                    break
                values.append(value)
                off += 4
        rect["values"] = values
        return
    if enc == 0x455:
        rect["version"] = be_u16(body, 18)
        rect["flags"] = be_u32(body, 20)
        source_id, _ = parse_len_prefixed_string(body, 24)
        rect["source_id"] = source_id
        return
    if enc == 0x456:
        rect["version"] = be_u16(body, 18)
        rect["block_count"] = be_u32(body, 20)
        rect["flags"] = be_u32(body, 24)
