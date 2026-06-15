from __future__ import annotations

import zlib
from dataclasses import dataclass

from .message_parser import be_u16, be_u32


@dataclass
class Rect:
    x: int
    y: int
    width: int
    height: int
    encoding: int
    payload: bytes


class ZlibStreamDecoder:
    def __init__(self) -> None:
        self.stream = zlib.decompressobj()

    def decode_rect(self, rect: Rect) -> bytes:
        if len(rect.payload) < 4:
            raise ValueError("zlib rectangle payload too short")
        compressed_len = be_u32(rect.payload, 0)
        if compressed_len is None:
            raise ValueError("missing zlib compressed length")
        compressed = rect.payload[4 : 4 + compressed_len]
        expected = rect.width * rect.height * 4
        out = bytearray()
        pending = compressed
        while len(out) < expected:
            chunk = self.stream.decompress(pending, expected - len(out))
            out.extend(chunk)
            pending = self.stream.unconsumed_tail
            if not pending and not chunk:
                break
        if len(out) != expected:
            raise ValueError(f"zlib rectangle decoded to {len(out)} bytes, expected {expected}")
        if pending:
            raise ValueError(f"zlib rectangle left {len(pending)} compressed bytes unconsumed")
        if self.stream.unused_data:
            raise ValueError(f"zlib rectangle produced unexpected unused data ({len(self.stream.unused_data)} bytes)")
        return bytes(out)


def framebuffer_update_expected_len(body: bytes) -> int | None:
    if len(body) < 4 or body[0] != 0:
        return None
    rect_count = be_u16(body, 2)
    if rect_count is None:
        return None
    off = 4
    for _ in range(rect_count):
        if off + 12 > len(body):
            return None
        width = be_u16(body, off + 4) or 0
        height = be_u16(body, off + 6) or 0
        encoding = be_u32(body, off + 8)
        if encoding is None:
            return None
        off += 12

        payload_len: int | None
        if encoding == 0x00000000:
            payload_len = width * height * 4
        elif encoding == 0x00000006:
            compressed_len = be_u32(body, off)
            payload_len = None if compressed_len is None else 4 + compressed_len
        elif encoding in {0x0000044C, 0x0000044D, 0xFFFFFF21}:
            payload_len = 0
        elif encoding in {0x00000450, 0x00000451, 0x00000453, 0x00000455, 0x00000456, 0x000003F2}:
            payload_len = _metadata_payload_len(body, off)
        else:
            return None

        if payload_len is None:
            return None
        if off + payload_len > len(body):
            return off + payload_len
        off += payload_len
    return off


def _metadata_payload_len(body: bytes, off: int) -> int | None:
    declared = be_u16(body, off)
    if declared is None:
        return None
    return 2 + declared


def parse_framebuffer_rects(body: bytes) -> list[Rect]:
    if len(body) < 4 or body[0] != 0:
        return []
    rect_count = be_u16(body, 2) or 0
    rects: list[Rect] = []
    off = 4
    for _ in range(rect_count):
        if off + 12 > len(body):
            break
        x = be_u16(body, off) or 0
        y = be_u16(body, off + 2) or 0
        width = be_u16(body, off + 4) or 0
        height = be_u16(body, off + 6) or 0
        encoding = be_u32(body, off + 8) or 0
        off += 12

        payload_len: int | None
        if encoding == 0x00000000:
            payload_len = width * height * 4
        elif encoding == 0x00000006:
            compressed_len = be_u32(body, off)
            payload_len = None if compressed_len is None else 4 + compressed_len
        elif encoding in {0x0000044C, 0x0000044D, 0xFFFFFF21}:
            payload_len = 0
        elif encoding in {0x00000450, 0x00000451, 0x00000453, 0x00000455, 0x00000456, 0x000003F2}:
            payload_len = _metadata_payload_len(body, off)
        else:
            payload_len = None

        if payload_len is None or off + payload_len > len(body):
            payload = body[off:]
            rects.append(Rect(x=x, y=y, width=width, height=height, encoding=encoding, payload=payload))
            break
        payload = body[off : off + payload_len]
        rects.append(Rect(x=x, y=y, width=width, height=height, encoding=encoding, payload=payload))
        off += payload_len
    return rects


def decode_raw_pixels(rect: Rect) -> bytes:
    return rect.payload


def decode_zlib_pixels(rect: Rect) -> bytes:
    if len(rect.payload) < 4:
        raise ValueError("zlib rectangle payload too short")
    compressed_len = be_u32(rect.payload, 0)
    if compressed_len is None:
        raise ValueError("missing zlib compressed length")
    compressed = rect.payload[4 : 4 + compressed_len]
    pixels = zlib.decompress(compressed)
    expected = rect.width * rect.height * 4
    if len(pixels) != expected:
        raise ValueError(f"zlib rectangle decoded to {len(pixels)} bytes, expected {expected}")
    return pixels


def display_layout_backing_size(rect: Rect) -> tuple[int, int] | None:
    if rect.encoding != 0x00000451:
        return None
    payload = rect.payload
    if len(payload) < 12:
        return None
    logical_width = be_u16(payload, 4)
    logical_height = be_u16(payload, 6)
    backing_width = be_u16(payload, 8)
    backing_height = be_u16(payload, 10)
    if backing_width and backing_height and backing_width < 0xFFFF and backing_height < 0xFFFF:
        return backing_width, backing_height
    if logical_width and logical_height and logical_width < 0xFFFF and logical_height < 0xFFFF:
        return logical_width, logical_height
    return None
