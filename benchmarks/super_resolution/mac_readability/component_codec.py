"""Exact source-pattern dictionary feasibility, without text recognition.

Connected foreground components are deduplicated only when dimensions and ALL
luma bytes match. No approximate shape matching, OCR substitution or changed
characters. Chroma remains lossless420. Whole-frame experimental encoding;
progressive delivery and integration are not implemented.
"""

import hashlib
import struct
import zlib

import numpy as np

try:
    from .detail_transport import planes_nv12, encode_plane, decode_plane
except ImportError:
    from detail_transport import planes_nv12, encode_plane, decode_plane

HEADER = struct.Struct(">4sHHBII32s")
SIZE = struct.Struct(">I")
SHAPE = struct.Struct(">HH")
POSITION = struct.Struct(">IHH")


def blob(data):
    return SIZE.pack(len(data)) + data


def read_blob(packet, offset):
    if offset + 4 > len(packet):
        raise ValueError("Truncated block")
    size = SIZE.unpack_from(packet, offset)[0]
    offset += 4
    if offset + size > len(packet):
        raise ValueError("Truncated payload")
    return packet[offset : offset + size], offset + size


def decompress_bounded(data, limit):
    decoder = zlib.decompressobj()
    raw = decoder.decompress(data, limit + 1)
    if (
        len(raw) > limit
        or not decoder.eof
        or decoder.unconsumed_tail
        or decoder.unused_data
    ):
        raise ValueError("Oversized or invalid compressed block")
    return raw


def encode(raw, width, height):
    import cv2

    y, u, v = planes_nv12(raw, width, height)
    bg = int(np.bincount(y.ravel(), minlength=256).argmax())
    mask = (y != bg).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    dictionary = {}
    entries = []
    instances = []
    for label in range(1, count):
        x, top, w, h, _ = map(int, stats[label])
        delta = (y[top : top + h, x : x + w] - bg).copy()
        delta[labels[top : top + h, x : x + w] != label] = 0
        key = SHAPE.pack(w, h) + delta.tobytes()
        if key not in dictionary:
            dictionary[key] = len(entries)
            entries.append(key)
        instances.append(POSITION.pack(dictionary[key], x, top))
    if sum(map(len, entries)) > width * height * 8:
        raise ValueError("Dictionary exceeds prototype resource limit")
    body = blob(zlib.compress(b"".join(entries), 6)) + blob(
        zlib.compress(b"".join(instances), 6)
    )
    body += encode_plane(u) + encode_plane(v)
    packet = (
        HEADER.pack(
            b"ISC1",
            width,
            height,
            bg,
            len(entries),
            len(instances),
            hashlib.sha256(raw).digest(),
        )
        + body
    )
    return packet, dict(
        components=len(instances),
        unique_patterns=len(entries),
        exact_reused_instances=len(instances) - len(entries),
        dictionary_raw_bytes=sum(map(len, entries)),
        dictionary_zlib_bytes=len(zlib.compress(b"".join(entries), 6)),
        positions_zlib_bytes=len(zlib.compress(b"".join(instances), 6)),
    )


def decode(packet):
    if len(packet) < HEADER.size:
        raise ValueError("Truncated frame")
    magic, w, h, bg, unique, count, identity = HEADER.unpack_from(packet)
    if (
        magic != b"ISC1"
        or w % 2
        or h % 2
        or not 0 < w <= 4096
        or not 0 < h <= 2160
        or unique > count
        or count > w * h
    ):
        raise ValueError("Invalid frame header")
    encoded, offset = read_blob(packet, HEADER.size)
    dictionary = decompress_bounded(encoded, w * h * 8)
    encoded, offset = read_blob(packet, offset)
    positions = decompress_bounded(encoded, count * POSITION.size)
    if len(positions) != count * POSITION.size:
        raise ValueError("Invalid placement count")
    entries = []
    i = 0
    for _ in range(unique):
        if i + 4 > len(dictionary):
            raise ValueError("Truncated dictionary")
        pw, ph = SHAPE.unpack_from(dictionary, i)
        i += 4
        if not 0 < pw <= w or not 0 < ph <= h or i + pw * ph > len(dictionary):
            raise ValueError("Invalid component dimensions")
        entries.append(
            np.frombuffer(dictionary[i : i + pw * ph], np.uint8).reshape(ph, pw)
        )
        i += pw * ph
    if i != len(dictionary):
        raise ValueError("Trailing dictionary data")
    y = np.full((h, w), bg, np.uint8)
    for i in range(count):
        key, x, top = POSITION.unpack_from(positions, i * POSITION.size)
        if key >= len(entries):
            raise ValueError("Unknown component")
        delta = entries[key]
        ph, pw = delta.shape
        if x + pw > w or top + ph > h:
            raise ValueError("Component outside frame")
        region = y[top : top + ph, x : x + pw]
        mask = delta != 0
        region[mask] = delta[mask] + bg
    u, offset = decode_plane(packet, offset, w // 2, h // 2)
    v, offset = decode_plane(packet, offset, w // 2, h // 2)
    if offset != len(packet):
        raise ValueError("Trailing frame data")
    uv = np.stack((u, v), axis=-1).tobytes()
    raw = y.tobytes() + uv
    if hashlib.sha256(raw).digest() != identity:
        raise ValueError("Pixel identity check failed")
    return raw
