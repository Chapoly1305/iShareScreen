"""HTTP/3 Capsule Protocol (RFC 9297) — the wire format WebTransport uses on
the CONNECT (session) stream once the session is established.

aioquic 1.3 implements the old WebTransport-over-HTTP/3 draft-02, where the
CONNECT stream carries nothing after the response. The standardized draft that
Safari/WebKit speaks (draft-ietf-webtrans-http3) instead runs the Capsule
Protocol on that stream: a sequence of ``[type varint][length varint][payload]``
records. The only one we must understand for session lifecycle is
CLOSE_WEBTRANSPORT_SESSION; the rest we skip.

This is a small, self-contained streaming decoder so a capsule split across
several QUIC STREAM frames is reassembled correctly.
"""

from enum import IntEnum
from typing import Iterator, Optional

from aioquic.buffer import UINT_VAR_MAX_SIZE, Buffer, BufferReadError

# Upper bound on a single capsule's declared payload length. The capsules we
# care about on the CONNECT stream (CLOSE_WEBTRANSPORT_SESSION and the flow-
# control ones we skip) are at most a few bytes; a QUIC varint can encode up to
# 2**62, so without a cap a peer could advertise a gigantic length and make the
# decoder buffer bytes toward it forever (memory-exhaustion DoS). Anything over
# this is treated as a protocol error.
_MAX_CAPSULE_LEN = 1 << 20  # 1 MiB — orders of magnitude over any real capsule


class CapsuleType(IntEnum):
    # RFC 9297 §5.4 — an HTTP Datagram carried on the stream itself.
    DATAGRAM_RFC = 0x00
    # draft-ietf-masque-h3-datagram-04 §8.2 — pre-RFC datagram capsules.
    DATAGRAM_DRAFT04 = 0xFF37A0
    REGISTER_DATAGRAM_CONTEXT_DRAFT04 = 0xFF37A1
    REGISTER_DATAGRAM_NO_CONTEXT_DRAFT04 = 0xFF37A2
    CLOSE_DATAGRAM_CONTEXT_DRAFT04 = 0xFF37A3
    # draft-ietf-webtrans-http3 — orderly session teardown from either peer.
    CLOSE_WEBTRANSPORT_SESSION = 0x2843


class H3Capsule:
    """One capsule: a type and its opaque payload."""

    def __init__(self, type: int, data: bytes) -> None:
        # `type` is a plain int (not CapsuleType) so we round-trip capsules of
        # types we don't recognize without losing them.
        self.type = type
        self.data = data

    def encode(self) -> bytes:
        buf = Buffer(capacity=len(self.data) + 2 * UINT_VAR_MAX_SIZE)
        buf.push_uint_var(self.type)
        buf.push_uint_var(len(self.data))
        buf.push_bytes(self.data)
        return buf.data


class H3CapsuleDecoder:
    """Streaming decoder: feed it stream bytes as they arrive with append(),
    mark end-of-stream with final(), and iterate to pull out whole capsules.
    A partial capsule at the tail is buffered until the rest arrives."""

    def __init__(self) -> None:
        self._buffer: Optional[Buffer] = None
        self._type: Optional[int] = None
        self._length: Optional[int] = None
        self._final: bool = False

    def append(self, data: bytes) -> None:
        # Guard, not assert: `python -O` strips asserts, and this is driven by
        # peer stream events whose ordering we don't fully control. Late data
        # after final() is simply ignored — the session is already over.
        if self._final:
            return
        if len(data) == 0:
            return
        if self._buffer:
            remaining = self._buffer.pull_bytes(
                self._buffer.capacity - self._buffer.tell())
            self._buffer = Buffer(data=(remaining + data))
        else:
            self._buffer = Buffer(data=data)

    def final(self) -> None:
        self._final = True

    def __iter__(self) -> Iterator[H3Capsule]:
        try:
            while self._buffer is not None:
                if self._type is None:
                    self._type = self._buffer.pull_uint_var()
                if self._length is None:
                    self._length = self._buffer.pull_uint_var()
                    if self._length > _MAX_CAPSULE_LEN:
                        raise ValueError(
                            f"capsule length {self._length} exceeds cap "
                            f"{_MAX_CAPSULE_LEN}")
                if self._buffer.capacity - self._buffer.tell() < self._length:
                    if self._final:
                        raise ValueError("insufficient capsule buffer")
                    return
                capsule = H3Capsule(
                    self._type, self._buffer.pull_bytes(self._length))
                self._type = None
                self._length = None
                if self._buffer.tell() == self._buffer.capacity:
                    self._buffer = None
                yield capsule
        except BufferReadError:
            # Ran off the end of the buffer mid-varint. If the stream is over
            # that's a real error; otherwise just wait for more bytes.
            if self._final:
                raise
            if not self._buffer:
                return
            size = self._buffer.capacity - self._buffer.tell()
            if size >= UINT_VAR_MAX_SIZE:
                raise
            return
