import io

import pytest

from benchmarks.super_resolution.mac_readability.stream_protocol import (
    Reader,
    fragments,
    KINDS,
)


def test_interleaved_video_and_supplement_and_cancelled_source():
    detail = fragments(KINDS["detail"], 1, 1, b"a" * 3000)
    video = fragments(KINDS["video"], 1, 2, b"b" * 1500)
    wire = detail[:1] + video + detail[1:]
    reader = Reader()
    results = []
    stream = io.BytesIO(b"".join(wire))
    for _ in wire:
        result = reader.read(stream)
        if result:
            results.append(result)
    assert [x[3] for x in results] == [b"b" * 1500, b"a" * 3000]
    assert reader.bytes == sum(map(len, wire)) and not reader.pending
    reader.read(io.BytesIO(detail[0]))
    reader.discard_old_detail(2)
    assert not reader.pending
    with pytest.raises(ValueError):
        reader.read(io.BytesIO(detail[1]))


def test_short_reads_and_truncated_or_out_of_order_messages():
    class Short(io.BytesIO):
        def read(self, size=-1):
            return super().read(min(size, 7))

    wire = fragments(KINDS["source"], 4, 1, b"x" * 1501)
    reader = Reader()
    assert reader.read(Short(wire[0])) is None
    assert reader.read(Short(wire[1]))[3] == b"x" * 1501
    with pytest.raises(EOFError):
        Reader().read(io.BytesIO(wire[0][:-1]))
    with pytest.raises(ValueError):
        Reader().read(io.BytesIO(wire[1]))
