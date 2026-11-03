"""Lossless full-width 420 strips; boundaries chosen from source pixels only."""

import hashlib

import numpy as np

try:
    from .detail_transport import (
        HEADER,
        Receiver,
        decode_plane,
        encode_plane,
        planes_nv12,
        wire_bytes,
    )
except ImportError:
    from detail_transport import (
        HEADER,
        Receiver,
        decode_plane,
        encode_plane,
        planes_nv12,
        wire_bytes,
    )


def cuts_from_luma(luma, aligned=True):
    height = luma.shape[0]
    if height % 2:
        raise ValueError("420 needs even height")
    energy = np.abs(np.diff(luma.astype(np.int16), axis=1)).sum(axis=1)
    cuts = [0]
    while height - cuts[-1] > 96:
        origin = cuts[-1]
        if aligned:
            # Prefer a low horizontal-contrast row gap near a64px strip. The
            # two rows on each side protect antialiased glyph descenders.
            options = range(origin + 32, min(origin + 96, height - 32) + 1, 2)
            boundary = min(
                options,
                key=lambda y: (
                    int(energy[y - 2 : y + 2].sum()),
                    abs(y - origin - 64),
                    y,
                ),
            )
        else:
            boundary = origin + 64
        cuts.append(boundary)
    cuts.append(height)
    return cuts


def frame_strips(raw, width, height, generation, aligned=True):
    if width > 4096:
        raise ValueError("Prototype maximum width4096")
    planes = planes_nv12(raw, width, height)
    cuts = cuts_from_luma(planes[0], aligned)
    identity = hashlib.sha256(raw).digest()[:16]
    tiles = []
    for y, end in zip(cuts, cuts[1:]):
        h = end - y
        blocks = [p[y // s : end // s, :] for p, s in zip(planes, (1, 2, 2))]
        packet = HEADER.pack(b"ISS2", generation, identity, 0, y, width, h)
        packet += b"".join(encode_plane(b) for b in blocks)
        luma = blocks[0].astype(np.int16)
        edge = float(
            np.abs(np.diff(luma, axis=0)).sum() + np.abs(np.diff(luma, axis=1)).sum()
        )
        tiles.append(
            dict(
                x=0,
                y=y,
                width=width,
                height=h,
                edge_energy=edge,
                priority=edge / wire_bytes(len(packet)),
                packet=packet,
            )
        )
    return sorted(tiles, key=lambda t: (-t["priority"], t["y"]))


class StripReceiver(Receiver):
    def accept(self, packet):
        if len(packet) < HEADER.size:
            raise ValueError("Truncated strip")
        magic, generation, identity, x, y, w, h = HEADER.unpack_from(packet)
        if magic != b"ISS2":
            raise ValueError("Unknown strip format")
        if generation != self.generation or identity != self.identity:
            return False
        if (
            x != 0
            or w != self.width
            or w > 4096
            or not 0 < h <= 96
            or y % 2
            or h % 2
            or y + h > self.height
        ):
            raise ValueError("Invalid strip geometry")
        blocks = []
        offset = HEADER.size
        for s in (1, 2, 2):
            block, offset = decode_plane(packet, offset, w // s, h // s)
            blocks.append(block)
        if offset != len(packet):
            raise ValueError("Trailing bytes")
        for target, block, s in zip(self.planes, blocks, (1, 2, 2)):
            target[y // s : (y + h) // s, :] = block
        self.mask[y : y + h, :] = True
        return True
