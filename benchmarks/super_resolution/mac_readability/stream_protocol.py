"""Bounded multiplexed benchmark framing over authenticated SSH stdio.

Video, source announcements, supplements and controls share one paced forward
stream. This framing does not turn an application cap into an IP/VPN wire cap.
"""

import struct

HEADER = struct.Struct(">4sBIIIIH")
MAGIC = b"ISS3"
FRAGMENT = 1200
MAX_MESSAGE = 16 * 1024 * 1024
KINDS = {
    "config": 1,
    "source": 2,
    "video": 3,
    "detail": 4,
    "ping": 5,
    "finish": 6,
    "bye": 7,
}


def fragments(kind, generation, message_id, payload):
    if kind not in KINDS.values() or not 0 < len(payload) <= MAX_MESSAGE:
        raise ValueError("Invalid message")
    return [
        HEADER.pack(
            MAGIC,
            kind,
            generation,
            message_id,
            offset,
            len(payload),
            len(payload[offset : offset + FRAGMENT]),
        )
        + payload[offset : offset + FRAGMENT]
        for offset in range(0, len(payload), FRAGMENT)
    ]


def read_exact(stream, size):
    chunks = []
    remaining = size
    while remaining:
        part = stream.read(remaining)
        if not part:
            raise EOFError("Stream ended mid-message" if chunks else "Stream ended")
        chunks.append(part)
        remaining -= len(part)
    return b"".join(chunks)


class Reader:
    def __init__(self):
        self.pending = {}
        self.bytes = 0
        self.messages = 0

    def read(self, stream):
        header = read_exact(stream, HEADER.size)
        magic, kind, generation, identity, offset, total, size = HEADER.unpack(header)
        if (
            magic != MAGIC
            or kind not in KINDS.values()
            or not 0 < total <= MAX_MESSAGE
            or not 0 < size <= FRAGMENT
            or offset + size > total
        ):
            raise ValueError("Invalid fragment header")
        data = read_exact(stream, size)
        self.bytes += HEADER.size + size
        if offset == 0:
            if identity in self.pending:
                raise ValueError("Repeated message start")
            if len(self.pending) >= 8:
                raise ValueError("Too many unfinished messages")
            self.pending[identity] = (kind, generation, total, bytearray())
        if identity not in self.pending:
            raise ValueError("Missing message start")
        saved_kind, saved_gen, saved_total, buffer = self.pending[identity]
        if (saved_kind, saved_gen, saved_total) != (
            kind,
            generation,
            total,
        ) or offset != len(buffer):
            raise ValueError("Fragment continuity failed")
        buffer.extend(data)
        if len(buffer) == total:
            del self.pending[identity]
            self.messages += 1
            return kind, generation, identity, bytes(buffer)
        return None

    def discard_old_detail(self, generation):
        # The sender cancels obsolete unsent supplement fragments on a source
        # change; release their partial receiver buffers at that announcement.
        for identity, value in list(self.pending.items()):
            if value[0] == KINDS["detail"] and value[1] < generation:
                del self.pending[identity]
