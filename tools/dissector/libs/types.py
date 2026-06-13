from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PacketSegment:
    frame_number: int
    timestamp_epoch: float
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    seq: int
    payload: bytes


@dataclass(frozen=True)
class ConversationKey:
    a_ip: str
    a_port: int
    b_ip: str
    b_port: int


@dataclass
class ConversationSummary:
    key: ConversationKey
    packet_count: int = 0
    payload_bytes: int = 0
    first_ts: float | None = None
    last_ts: float | None = None
    server_endpoint: tuple[str, int] | None = None


@dataclass
class ChunkSpan:
    start: int
    end: int
    timestamp_epoch: float
    frame_number: int
    seq: int
    payload_len: int


@dataclass
class ReassembledStream:
    direction: str
    data: bytes
    spans: list[ChunkSpan]
    base_seq: int | None
    gap_count: int = 0
    overlap_count: int = 0

    def timestamp_for_offset(self, offset: int) -> float | None:
        for span in self.spans:
            if span.start <= offset < span.end:
                return span.timestamp_epoch
        if self.spans:
            return self.spans[-1].timestamp_epoch
        return None

    def frame_for_offset(self, offset: int) -> int | None:
        for span in self.spans:
            if span.start <= offset < span.end:
                return span.frame_number
        if self.spans:
            return self.spans[-1].frame_number
        return None


@dataclass
class RekeyInfo:
    server_offset: int
    frame_number: int
    timestamp_epoch: float
    counter: int
    wrap_key_hex: str
    next_key_hex: str
    next_iv_hex: str


@dataclass
class DecodedRecord:
    stream_id: str
    record_index: int
    direction: str
    timestamp_epoch: float | None
    frame_number: int | None
    tcp_offset: int
    cipher_len: int
    msg_id: int | None
    msg_name: str
    record_kind: str
    plain_len: int
    plain_hex: str
    body_bytes: bytes = b""
    filler_hex: str = ""
    trailing_sha1_hex: str = ""
    parsed: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("body_bytes", None)
        return out
