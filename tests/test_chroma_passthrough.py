"""Unit tests for the nv24 chroma-passthrough fast path.

Apple's HW VideoToolbox delivers HEVC RExt 4:4:4 as nv24 (Y plane + a
full-resolution *interleaved* UV plane). The old path deinterleaved UV on the
CPU (a stride-2 gather per tile per frame — ~half a core at 4-tile/60fps under
the GIL, measured). The fast path hands the interleaved plane to the GPU
verbatim as an rg8unorm texture and lets the fragment shader read .r=Cb / .g=Cr.

These tests don't touch a GPU. They prove two things locally:

  1. The nv24 branch emits a passthrough `TileFrame` (`v is None`, raw
     interleaved UV in `u`, correct strides) — no CPU deinterleave.
  2. Sampling that rg8 texture (texel x → bytes [2x], [2x+1]) reproduces the
     EXACT U/V bytes the old CPU deinterleave produced. This is the
     correctness guarantee that the pixels on screen are unchanged.

`ISS_LEGACY_CHROMA` still selects the CPU deinterleave path.
"""
from __future__ import annotations

import types

import numpy as np

from isharescreen.proxy.media import hevc
from isharescreen.proxy.media.hevc import _av_frame_to_tile


class _FakePlane:
    """Stands in for an `av.VideoPlane`: `bytes(plane)` copies the buffer
    (incl. stride padding) and `.line_size` is the row pitch."""
    def __init__(self, buf: bytes, line_size: int):
        self._buf = bytes(buf)
        self.line_size = line_size

    def __bytes__(self) -> bytes:
        return self._buf


class _FakeFrame:
    def __init__(self, fmt, w, h, planes):
        self.format = types.SimpleNamespace(name=fmt)
        self.width = w
        self.height = h
        self.planes = planes
        self.decode_error_flags = 0
        self.flags = 0


def _nv24_frame(w, h, *, uv_pad=0):
    """Synthetic nv24 frame. UV is interleaved Cb,Cr at full res; `uv_pad`
    adds stride padding past the 2*w live bytes to exercise the padding skip."""
    y_line = w
    uv_line = 2 * w + uv_pad
    ybuf = (np.arange(h * y_line) % 256).astype(np.uint8).tobytes()
    uv = np.zeros((h, uv_line), dtype=np.uint8)
    # Distinct, position-dependent Cb/Cr so a channel swap would be caught.
    live = (np.arange(h * 2 * w).reshape(h, 2 * w) % 251).astype(np.uint8)
    uv[:, : 2 * w] = live
    yp = _FakePlane(ybuf, y_line)
    uvp = _FakePlane(uv.tobytes(), uv_line)
    return _FakeFrame("nv24", w, h, [yp, uvp]), ybuf, uv, uv_line


def test_nv24_emits_passthrough_tileframe():
    w, h = 64, 8
    frame, ybuf, uv, uv_line = _nv24_frame(w, h, uv_pad=16)
    tile, had_err = _av_frame_to_tile(frame, [None], set())
    assert had_err is False
    assert tile.is_nv12_passthrough          # v is None
    assert tile.v is None
    assert tile.y == ybuf                     # luma owned verbatim
    assert tile.u == uv.tobytes()             # interleaved UV, NOT deinterleaved
    assert tile.uv_stride == uv_line          # interleaved row pitch (padded)
    assert tile.chroma_width == w             # rg8 texels per row
    assert tile.chroma_height == h
    assert tile.width == w and tile.height == h


def _nv12_frame(w, h, *, uv_pad=0):
    """Synthetic nv12 frame: Y at full res + HALF-res interleaved Cb,Cr.
    `uv_pad` adds stride padding past the live `w` bytes/row (w//2 pairs)."""
    y_line = w
    uv_line = w + uv_pad                       # w//2 (Cb,Cr) pairs = w live bytes
    ybuf = (np.arange(h * y_line) % 256).astype(np.uint8).tobytes()
    uv = np.zeros((h // 2, uv_line), dtype=np.uint8)
    uv[:, :w] = (np.arange((h // 2) * w).reshape(h // 2, w) % 251).astype(np.uint8)
    return _FakeFrame("nv12", w, h, [_FakePlane(ybuf, y_line),
                                     _FakePlane(uv.tobytes(), uv_line)]), ybuf, uv, uv_line


def test_nv12_emits_halfres_biplanar_passthrough():
    """AVC/H.264 4:2:0: nv12 must passthrough to a HALF-res biplanar
    TileFrame (no CPU deinterleave, no swscale 4:2:0->4:4:4 upsample). The
    GPU renderer bilinearly upsamples chroma via the chroma_scale uniform."""
    w, h = 64, 8
    frame, ybuf, uv, uv_line = _nv12_frame(w, h, uv_pad=8)
    tile, had_err = _av_frame_to_tile(frame, [None], set())
    assert had_err is False
    assert tile.is_nv12_passthrough and tile.v is None
    assert tile.y == ybuf                      # luma verbatim
    assert tile.u == uv.tobytes()              # interleaved half-res UV verbatim
    assert tile.uv_stride == uv_line
    assert tile.chroma_width == w // 2         # 4:2:0 → half horizontal
    assert tile.chroma_height == h // 2        # 4:2:0 → half vertical
    assert tile.width == w and tile.height == h


def test_yuv420p_emits_halfres_planar():
    """Software-decoded H.264 (yuv420p) → half-res PLANAR TileFrame (not
    upsampled to 4:4:4); the renderer chroma_scale handles the upsampling."""
    w, h = 64, 8
    yp = _FakePlane((np.arange(h * w) % 256).astype(np.uint8).tobytes(), w)
    up = _FakePlane((np.arange((h // 2) * (w // 2)) % 251).astype(np.uint8).tobytes(), w // 2)
    vp = _FakePlane((np.arange((h // 2) * (w // 2)) % 247).astype(np.uint8).tobytes(), w // 2)
    frame = _FakeFrame("yuv420p", w, h, [yp, up, vp])
    tile, _ = _av_frame_to_tile(frame, [None], set())
    assert tile.v is not None                  # planar, NOT passthrough
    assert tile.chroma_width == w // 2 and tile.chroma_height == h // 2


def test_rg8_sampling_reproduces_cpu_deinterleave():
    """The crux: reading the rg8 texture (.r=byte[2x], .g=byte[2x+1]) must yield
    the same U and V planes the old stride-2 CPU deinterleave produced."""
    w, h = 64, 8
    frame, _ybuf, uv, uv_line = _nv24_frame(w, h, uv_pad=16)
    tile, _ = _av_frame_to_tile(frame, [None], set())

    # OLD behaviour (hevc.py fallback / what the screen used to show):
    view = uv[:, : 2 * w]
    u_old = view[:, 0::2]
    v_old = view[:, 1::2]

    # NEW: the GPU receives `tile.u` with bytes_per_row=tile.uv_stride into an
    # rg8 texture. Texel (x,y) samples .r=byte[y,2x], .g=byte[y,2x+1]. Model
    # that read directly from the bytes the renderer uploads.
    rg = (np.frombuffer(tile.u, np.uint8)
          .reshape(tile.chroma_height, tile.uv_stride)[:, : 2 * w]
          .reshape(h, w, 2))
    u_new = rg[:, :, 0]
    v_new = rg[:, :, 1]

    assert np.array_equal(u_new, u_old)       # .r == Cb (U)
    assert np.array_equal(v_new, v_old)       # .g == Cr (V)


def test_legacy_chroma_env_still_deinterleaves(monkeypatch):
    """ISS_LEGACY_CHROMA falls back to the CPU deinterleave (planar TileFrame),
    and that planar U/V equals the rg8 reconstruction — proving both paths
    are bit-identical, just one pays the CPU cost."""
    monkeypatch.setattr(hevc, "_LEGACY_CHROMA", True)
    w, h = 64, 8
    frame, _ybuf, uv, _line = _nv24_frame(w, h, uv_pad=0)
    tile, _ = _av_frame_to_tile(frame, [None], set())

    assert tile.v is not None                 # planar, NOT passthrough
    assert not tile.is_nv12_passthrough
    view = uv[:, : 2 * w]
    assert tile.u == view[:, 0::2].tobytes()
    assert tile.v == view[:, 1::2].tobytes()
