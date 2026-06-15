from __future__ import annotations

from .types import ChunkSpan, PacketSegment, ReassembledStream


def reassemble_direction(segments: list[PacketSegment], direction: str) -> ReassembledStream:
    if not segments:
        return ReassembledStream(direction=direction, data=b"", spans=[], base_seq=None)

    ordered = sorted(segments, key=lambda s: (s.timestamp_epoch, s.frame_number, s.seq))
    base_seq = ordered[0].seq
    stream = bytearray()
    spans: list[ChunkSpan] = []
    gap_count = 0
    overlap_count = 0
    wrap_base = 0
    prev_seq = ordered[0].seq
    prev_ext_seq = ordered[0].seq

    for segment in ordered:
        if segment.seq < prev_seq and prev_seq - segment.seq > 0x80000000:
            wrap_base += 0x100000000
        ext_seq = segment.seq + wrap_base
        while ext_seq + 0x80000000 < prev_ext_seq:
            ext_seq += 0x100000000
        logical_start = ext_seq - base_seq
        if logical_start > len(stream):
            gap_count += 1
            prev_seq = segment.seq
            prev_ext_seq = ext_seq
            continue
        overlap = len(stream) - logical_start
        payload = segment.payload
        if overlap >= len(payload):
            overlap_count += 1
            prev_seq = segment.seq
            prev_ext_seq = ext_seq
            continue
        if overlap > 0:
            overlap_count += 1
            payload = payload[overlap:]
            logical_start += overlap
        start = len(stream)
        stream.extend(payload)
        spans.append(
            ChunkSpan(
                start=start,
                end=len(stream),
                timestamp_epoch=segment.timestamp_epoch,
                frame_number=segment.frame_number,
                seq=segment.seq,
                payload_len=len(segment.payload),
            )
        )
        prev_seq = segment.seq
        prev_ext_seq = ext_seq

    return ReassembledStream(
        direction=direction,
        data=bytes(stream),
        spans=spans,
        base_seq=base_seq,
        gap_count=gap_count,
        overlap_count=overlap_count,
    )
