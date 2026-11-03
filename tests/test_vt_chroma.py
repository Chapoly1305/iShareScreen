"""Preserve HEVC source chroma and refuse unsupported native plane layouts."""

from types import SimpleNamespace
import re
import threading

import numpy as np
import pytest

from isharescreen.proxy.media import vtdecode
from isharescreen.proxy.media.hwcaps import _HEVC444_SAMPLE
from isharescreen.enhancement.live import tile_rgb

# Public format parameter set from the actual Mac 420 calibration stream.
SPS_420 = bytes.fromhex(
    "420101016000000300b00000030000030099a001a420084588043b914822b9f84f42fa85d58dab87d5d046aea09d5d5056aeaa0bd5d55066aeaaa0dd5d555076aeaaaa0fd5d5555021abaaaaa94dc303404020"
)
SPS_444 = next(
    n
    for n in re.split(b"\x00\x00\x00?\x01", _HEVC444_SAMPLE)
    if n and (n[0] >> 1) & 63 == 33
)


@pytest.mark.parametrize("sps,expected", [(SPS_420, 0x34323066), (SPS_444, 0x34343466)])
def test_session_selects_source_sampling(monkeypatch, sps, expected):
    decoder = vtdecode.VTHevcDecoder.__new__(vtdecode.VTHevcDecoder)
    decoder._lock = threading.Lock()
    monkeypatch.setattr(decoder, "_build_session_locked", lambda: None)
    decoder.set_params(b"vps", sps, {0: b"pps"})
    assert decoder._output_format == expected


class Plane:
    def __init__(self, data):
        self.data = data

    def as_buffer(self, size):
        assert len(self.data) == size
        return self.data


def quartz(fmt, cw, ch):
    state = SimpleNamespace(unlocks=0)

    def unlock(*args):
        state.unlocks += 1

    q = SimpleNamespace(
        CVPixelBufferLockBaseAddress=lambda *args: 0,
        CVPixelBufferUnlockBaseAddress=unlock,
        CVPixelBufferGetWidth=lambda b: 4,
        CVPixelBufferGetHeight=lambda b: 4,
        CVPixelBufferGetPlaneCount=lambda b: 2,
        CVPixelBufferGetPixelFormatType=lambda b: fmt,
        CVPixelBufferGetBytesPerRowOfPlane=lambda b, i: 8,
        CVPixelBufferGetHeightOfPlane=lambda b, i: 4 if i == 0 else ch,
        CVPixelBufferGetWidthOfPlane=lambda b, i: 4 if i == 0 else cw,
        CVPixelBufferGetBaseAddressOfPlane=lambda b, i: Plane(
            bytes([100 if i == 0 else 128]) * (32 if i == 0 else 8 * ch)
        ),
    )
    return q, state


@pytest.mark.parametrize("fmt,cw,ch", [(0x34323066, 2, 2), (0x34343466, 4, 4)])
def test_native_planes_reach_rgb_consumer_with_correct_shape(monkeypatch, fmt, cw, ch):
    q, state = quartz(fmt, cw, ch)
    monkeypatch.setattr(vtdecode, "_Q", q, raising=False)
    decoder = vtdecode.VTHevcDecoder.__new__(vtdecode.VTHevcDecoder)
    decoder._seen_fmt = decoder._fmt_warned = False
    frame = decoder._cvbuf_to_tile(object())
    assert (frame.width, frame.height, frame.chroma_width, frame.chroma_height) == (
        4,
        4,
        cw,
        ch,
    )
    assert len(frame.y) == 32 and len(frame.u) == 8 * ch and frame.v is None
    pixels = tile_rgb(frame)
    assert pixels.shape == (4, 4, 3) and np.max(pixels) - np.min(pixels) <= 1
    assert state.unlocks == 1


@pytest.mark.parametrize("fmt,cw,ch", [(0x34323076, 2, 2), (0x34323066, 4, 4)])
def test_reject_limited_range_or_mislabeled_chroma(monkeypatch, fmt, cw, ch):
    q, state = quartz(fmt, cw, ch)
    monkeypatch.setattr(vtdecode, "_Q", q, raising=False)
    decoder = vtdecode.VTHevcDecoder.__new__(vtdecode.VTHevcDecoder)
    decoder._seen_fmt = decoder._fmt_warned = False
    assert decoder._cvbuf_to_tile(object()) is None
    assert state.unlocks == 1
