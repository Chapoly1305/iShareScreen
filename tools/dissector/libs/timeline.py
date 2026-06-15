from __future__ import annotations

from dataclasses import dataclass

from .types import DecodedRecord


@dataclass
class PresentationFrame:
    frame_index: int
    pts_epoch: float
    width: int
    height: int
    source_record_indices: list[int]


def group_server_records(records: list[DecodedRecord]) -> list[list[DecodedRecord]]:
    server_records = [r for r in records if r.direction == "server" and r.msg_name == "FramebufferUpdate" and r.timestamp_epoch is not None]
    if not server_records:
        return []
    groups: list[list[DecodedRecord]] = []
    current: list[DecodedRecord] = []
    current_ts: float | None = None
    for record in server_records:
        if current_ts is None or abs(record.timestamp_epoch - current_ts) < 1e-9:
            current.append(record)
            current_ts = record.timestamp_epoch
            continue
        groups.append(current)
        current = [record]
        current_ts = record.timestamp_epoch
    if current:
        groups.append(current)
    return groups
