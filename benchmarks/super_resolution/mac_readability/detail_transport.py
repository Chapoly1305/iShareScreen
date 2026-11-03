"""Experimental revision-safe lossless 420 tiles, independent of OCR and SR.

Each tile carries its source generation and content identity. Plane filters are
reversible byte deltas; compression does not synthesize characters. The sender
must announce a new generation before its new base image is displayed.
This is a benchmark protocol, not an integrated Screen Sharing transport.
"""

import hashlib
import math
import struct
import zlib

import numpy as np

HEADER = struct.Struct(">4sQ16sHHHH")
PLANE = struct.Struct(">BI")
MAX_TILE = 256


def planes_nv12(raw, width, height):
    if width % 2 or height % 2 or len(raw) != width * height * 3 // 2:
        raise ValueError("Invalid 420 dimensions or byte length")
    flat = np.frombuffer(raw, np.uint8)
    n = width * height
    return (
        flat[:n].reshape(height, width),
        flat[n::2].reshape(height // 2, width // 2),
        flat[n + 1 :: 2].reshape(height // 2, width // 2),
    )


def encode_plane(plane):
    candidates = [plane.tobytes()]
    for axis in (1, 0):
        delta = plane.copy()
        if axis == 1:
            delta[:, 1:] = plane[:, 1:] - plane[:, :-1]
        else:
            delta[1:, :] = plane[1:, :] - plane[:-1, :]
        candidates.append(delta.tobytes())
    choices = [(zlib.compress(value, 6), mode) for mode, value in enumerate(candidates)]
    compressed, mode = min(choices, key=lambda v: (len(v[0]), v[1]))
    return PLANE.pack(mode, len(compressed)) + compressed


def decode_plane(data, offset, width, height):
    if offset + PLANE.size > len(data):
        raise ValueError("Truncated plane header")
    mode, size = PLANE.unpack_from(data, offset)
    offset += PLANE.size
    if mode > 2 or size > 2 * width * height + 1024 or offset + size > len(data):
        raise ValueError("Invalid compressed plane")
    decoder = zlib.decompressobj()
    raw = decoder.decompress(data[offset : offset + size], width * height + 1)
    if (
        len(raw) != width * height
        or not decoder.eof
        or decoder.unused_data
        or decoder.unconsumed_tail
    ):
        raise ValueError("Invalid uncompressed plane size")
    plane = np.frombuffer(raw, np.uint8).reshape(height, width)
    if mode:
        plane = np.cumsum(plane, axis=1 if mode == 1 else 0, dtype=np.uint64).astype(
            np.uint8
        )
    return plane, offset + size


def frame_tiles(raw, width, height, generation, previous_raw=None):
    planes = planes_nv12(raw, width, height)
    previous = (
        planes_nv12(previous_raw, width, height) if previous_raw is not None else None
    )
    identity = hashlib.sha256(raw).digest()[:16]
    tiles = []
    for y in range(0, height, MAX_TILE):
        for x in range(0, width, MAX_TILE):
            w, h = min(MAX_TILE, width - x), min(MAX_TILE, height - y)
            blocks = [
                p[y // s : (y + h) // s, x // s : (x + w) // s]
                for p, s in zip(planes, (1, 2, 2))
            ]
            if previous is not None and all(
                np.array_equal(b, p[y // s : (y + h) // s, x // s : (x + w) // s])
                for b, p, s in zip(blocks, previous, (1, 2, 2))
            ):
                continue
            packet = HEADER.pack(b"ISR1", generation, identity, x, y, w, h)
            packet += b"".join(encode_plane(b) for b in blocks)
            luma = blocks[0].astype(np.int16)
            edge = float(
                np.abs(np.diff(luma, axis=0)).sum()
                + np.abs(np.diff(luma, axis=1)).sum()
            )
            # Source pixels only. No text labels, OCR results or category routing.
            tiles.append(
                dict(
                    x=x,
                    y=y,
                    width=w,
                    height=h,
                    edge_energy=edge,
                    priority=edge / wire_bytes(len(packet)),
                    packet=packet,
                )
            )
    return sorted(tiles, key=lambda v: (-v["priority"], v["y"], v["x"]))


def wire_bytes(payload):
    """Model 1200-byte application fragments plus 16 framing + 28 IPv4/UDP bytes.

    Excludes ACK, retransmission, VPN, Ethernet and encryption. This is an
    optimistic accounting model, never a measured network traffic claim.
    """
    return payload + math.ceil(payload / 1200) * 44


class Receiver:
    def __init__(self, width, height):
        if width % 2 or height % 2:
            raise ValueError("420 requires even dimensions")
        self.width, self.height = width, height
        self.generation = -1
        self.identity = None
        self.planes = [np.zeros((height // s, width // s), np.uint8) for s in (1, 2, 2)]
        self.mask = np.zeros((height, width), bool)

    def begin(self, generation, identity):
        if generation <= self.generation or len(identity) != 16:
            raise ValueError("Source revisions must increase")
        self.generation, self.identity = generation, identity
        self.mask.fill(False)  # Never expose old-generation text as refinement.

    def accept(self, packet):
        if len(packet) < HEADER.size:
            raise ValueError("Truncated tile")
        magic, generation, identity, x, y, w, h = HEADER.unpack_from(packet)
        if magic != b"ISR1":
            raise ValueError("Unknown tile version")
        if generation != self.generation or identity != self.identity:
            return False
        if (
            not 0 < w <= MAX_TILE
            or not 0 < h <= MAX_TILE
            or any(v % 2 for v in (x, y, w, h))
            or x + w > self.width
            or y + h > self.height
        ):
            raise ValueError("Invalid tile geometry")
        decoded, offset = [], HEADER.size
        for s in (1, 2, 2):
            plane, offset = decode_plane(packet, offset, w // s, h // s)
            decoded.append(plane)
        if offset != len(packet):
            raise ValueError("Trailing tile bytes")
        # Commit only once the entire packet has validated.
        for target, block, s in zip(self.planes, decoded, (1, 2, 2)):
            target[y // s : (y + h) // s, x // s : (x + w) // s] = block
        self.mask[y : y + h, x : x + w] = True
        return True

    def planar(self):
        return np.concatenate([p.ravel() for p in self.planes]).reshape(
            self.height * 3 // 2, self.width
        )
