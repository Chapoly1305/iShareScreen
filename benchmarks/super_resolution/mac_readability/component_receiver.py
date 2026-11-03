"""Generation gate for exact full-frame supplements; decode outside the lock."""

import threading

from .component_codec import HEADER, decode


class ComponentReceiver:
    def __init__(self):
        self._lock = threading.Lock()
        self.generation = -1
        self.identity = None
        self.latest = None

    def begin(self, generation, identity):
        if len(identity) != 32:
            raise ValueError("Expected full source SHA256")
        with self._lock:
            if generation <= self.generation:
                raise ValueError("Source revision must increase")
            self.generation, self.identity, self.latest = generation, identity, None

    def accept(self, generation, packet):
        if len(packet) < HEADER.size:
            raise ValueError("Truncated supplement")
        identity = HEADER.unpack_from(packet)[-1]
        with self._lock:
            if generation != self.generation or identity != self.identity:
                return None
        raw = decode(packet)
        with self._lock:
            # A later source announcement can arrive while decompression runs.
            if generation != self.generation or identity != self.identity:
                return None
            self.latest = raw
            return raw
