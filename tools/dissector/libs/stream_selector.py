from __future__ import annotations

from collections import defaultdict

from .types import ConversationKey, ConversationSummary, PacketSegment


def canonical_key(segment: PacketSegment) -> ConversationKey:
    a = (segment.src_ip, segment.src_port)
    b = (segment.dst_ip, segment.dst_port)
    if a <= b:
        return ConversationKey(a[0], a[1], b[0], b[1])
    return ConversationKey(b[0], b[1], a[0], a[1])


def summarize_conversations(segments: list[PacketSegment]) -> list[ConversationSummary]:
    by_key: dict[ConversationKey, ConversationSummary] = {}
    for segment in segments:
        key = canonical_key(segment)
        summary = by_key.get(key)
        if summary is None:
            summary = ConversationSummary(key=key)
            if segment.src_port == 5900:
                summary.server_endpoint = (segment.src_ip, segment.src_port)
            elif segment.dst_port == 5900:
                summary.server_endpoint = (segment.dst_ip, segment.dst_port)
            by_key[key] = summary
        summary.packet_count += 1
        summary.payload_bytes += len(segment.payload)
        summary.first_ts = segment.timestamp_epoch if summary.first_ts is None else min(summary.first_ts, segment.timestamp_epoch)
        summary.last_ts = segment.timestamp_epoch if summary.last_ts is None else max(summary.last_ts, segment.timestamp_epoch)
        if summary.server_endpoint is None:
            if segment.src_port == 5900:
                summary.server_endpoint = (segment.src_ip, segment.src_port)
            elif segment.dst_port == 5900:
                summary.server_endpoint = (segment.dst_ip, segment.dst_port)
    return sorted(by_key.values(), key=lambda s: (s.first_ts or 0.0, s.key.a_ip, s.key.a_port, s.key.b_ip, s.key.b_port))


def select_conversation(
    segments: list[PacketSegment],
    client_ip: str,
    server_ip: str,
    server_port: int,
) -> list[PacketSegment]:
    out: list[PacketSegment] = []
    for segment in segments:
        if segment.src_ip == client_ip and segment.dst_ip == server_ip and segment.dst_port == server_port:
            out.append(segment)
        elif segment.src_ip == server_ip and segment.src_port == server_port and segment.dst_ip == client_ip:
            out.append(segment)
    return sorted(out, key=lambda s: (s.timestamp_epoch, s.frame_number))
