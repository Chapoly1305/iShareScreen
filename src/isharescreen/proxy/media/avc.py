"""H.264 (AVC) decoder for Apple's screen stream.

Apple sends H.264 4:2:0 (yuvj420p) when the client advertises the field1=123
codec bank. Reverse-engineered from live captures:

  * The 4 tiles are decoded as ONE timestamp-ordered H.264 stream through a
    single shared `av.CodecContext`. This was settled empirically: feeding all
    tiles' NALs to one context in arrival (timestamp) order decodes clean at
    60fps, whereas per-tile contexts or out-of-order feeding conceal/gray.
    Output frames are routed back to the tile that fed them via a FIFO, which
    is correct because there is no B-frame reordering (one slice = one AU,
    emitted in order).
  * Apple does NOT emit type-5 IDR NALs. Keyframes are intra (I) slices carried
    in ordinary type-1 NALs; it re-keys by spinning up a fresh SSRC generation.
    So "have we got a keyframe" can't be detected from the NAL type — instead
    the first frame the decoder actually EMITS is the keyframe signal.

H.264 4:2:0 is hardware-decodable everywhere (the point of this path vs HEVC
4:4:4), though this decoder is software-only for now. Frame extraction + the
quality gate are reused verbatim from the HEVC path.
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from typing import Callable, Optional

import av

from .avc_nalu import h264_nal_type
from .decode_common import (
    _CODEC_FLAG_LOW_DELAY, _CODEC_FLAG2_FAST,
    _NAL_START_CODE, _TileSlot, _av_frame_to_tile,
)
from .tiles import TileFrame
from .quality_gate import FrameQualityGate

log = logging.getLogger(__name__)

# Hardware decoders to try per platform for H.264. Unlike Apple's HEVC RExt
# 4:4:4 (which DXVA2/D3D11VA/VAAPI cannot decode, so the "HW" context silently
# falls back to software), H.264 4:2:0 is the universally hardware-decodable
# profile — these accelerators DO bind, which is the whole point of the AVC
# path: real GPU decode on the Windows/Linux boxes where 4:4:4 can't.
_H264_HWACCELS: dict[str, tuple[str, ...]] = {
    "darwin": ("videotoolbox",),
    "win32": ("d3d11va", "dxva2"),
    "linux": ("vaapi",),
    "*": (),
}


def _h264_hwaccels() -> tuple[str, ...]:
    return _H264_HWACCELS.get(sys.platform, _H264_HWACCELS["*"])


# pts→tile routing map bounds. Each fed slice gets a monotonic pts and the map
# is drained as frames emit (one in → one out under low-delay), so it normally
# holds ≤ num_tiles entries. The cap only guards a pathological non-emitting
# decoder from unbounded growth; prune keeps the most-recent window.
_PTS_MAP_SOFT_MAX = 256
_PTS_MAP_PRUNE_KEEP = 64


def _nal_is_keyframe(nalu: bytes) -> bool:
    """True if a raw H.264 NAL re-roots the DPB (a keyframe). Type-5 is a
    real IDR; Apple's HP encoder also keys via intra (I) slices in ordinary
    type-1 NALs, detected from the slice header's slice_type ∈ {2,7} (I /
    I-only-picture). The slice header after the 1-byte NAL header is
    first_mb_in_slice ue(v), slice_type ue(v) on the EPB-stripped RBSP."""
    if len(nalu) < 2:
        return False
    t = nalu[0] & 0x1F
    if t == 5:               # NAL_SLICE_IDR
        return True
    if t == 1:               # NAL_SLICE_NONIDR — intra if slice_type is I
        from .bitstream import BitReader, remove_emulation_prevention
        try:
            br = BitReader(remove_emulation_prevention(nalu[1:]))
            br.read_ue()                  # first_mb_in_slice
            return br.read_ue() in (2, 7)  # slice_type
        except Exception:
            return False
    return False


class AvcDecoder:
    def __init__(
        self,
        num_tiles: int,
        *,
        prefer_hwaccel: bool = True,
        enable_quality_gate: bool = True,
        on_frame_published: Optional[Callable[[int], None]] = None,
    ) -> None:
        if num_tiles <= 0:
            raise ValueError("num_tiles must be positive")
        self.num_tiles = num_tiles
        # ISS_FORCE_SW_DECODE=1 forces software decode — removes the
        # platform-specific HW decoder (e.g. VideoToolbox) as a variable when
        # validating the protocol/stream itself.
        if os.environ.get("ISS_FORCE_SW_DECODE", "0") == "1":
            prefer_hwaccel = False
        self._prefer_hwaccel = prefer_hwaccel
        self._on_frame_published = on_frame_published
        self._tiles = [_TileSlot() for _ in range(num_tiles)]
        # ONE shared context: the 4 tiles are decoded as a single timestamp-
        # ordered stream (verified — feeding all tiles' NALs to one context in
        # ts order is what decodes clean; separate contexts / out-of-order
        # feeding conceal). Output frames are routed back to the tile that fed
        # them via a pts→tile map (each slice gets a monotonic pts), since
        # H.264 here has no B-frame reordering.
        self._codec: Optional[av.codec.context.CodecContext] = None
        # Guards every _codec access. feed_nalu runs on the video-process
        # thread while restart()/close() can fire from the stall-watchdog
        # thread — without this lock a teardown could free the context mid-
        # decode (use-after-free). Mirrors HevcDecoder._codec_lock.
        self._codec_lock = threading.Lock()
        # pts→tile routing (mirrors HevcDecoder): each fed slice gets a
        # monotonic pts; emitted frames are mapped back to their source tile
        # via frame.pts. Robust against any decoder drop/reorder, unlike a
        # strict in/out FIFO.
        self._next_pts = 0
        self._pts_to_tile: dict[int, int] = {}
        self._pts_submit_t: dict[int, float] = {}
        self._reformatter: list = [None]
        self._seen_fmts: set = set()
        self._gate = FrameQualityGate(num_tiles, enabled=enable_quality_gate)
        self._sps = b""
        self._pps = b""
        # Opens on the first emitted frame (Apple has no type-5 IDR — keyframes
        # are intra slices in type-1 NALs); until then output is cold-DPB fill.
        self._dpb_ready = False
        # After a restart the DPB is empty; drop slices until the first keyframe
        # re-roots it (see feed_nalu). Armed at construction so the initial
        # burst's IDR opens the gate.
        self._await_key = True
        self._hw_name: Optional[str] = None  # software-only for now
        self.nalu_counts_per_tile: list[dict[int, int]] = [
            {} for _ in range(num_tiles)
        ]
        # LTRP is HEVC-only; keep the attribute so the session's LTR-ack path
        # is a no-op instead of crashing.
        self.last_clean_donl: list[Optional[int]] = [None] * num_tiles
        # Decode latency monitoring: EMA of submit→frame round-trip (ms).
        self._decode_latency_ms: float = 0.0
        self._queue_full_drops: int = 0

    # -- setup ---------------------------------------------------------

    def set_params(self, vps: bytes, sps: bytes, all_pps: dict) -> None:
        """Install SPS/PPS. `vps` is ignored (H.264 has none); all_pps is the
        {pps_id: pps_nal} map harvested from an avcC config. Tiles share the
        same geometry so one SPS/PPS seeds every tile context."""
        self._sps = sps or b""
        self._pps = next(iter(all_pps.values())) if all_pps else b""

    def _build_extradata(self) -> bytes:
        return _NAL_START_CODE + self._sps + _NAL_START_CODE + self._pps

    def _ensure_codec_locked(self) -> Optional[av.codec.context.CodecContext]:
        """Build the shared context if needed (HW accel first, SW fallback).
        Caller MUST hold _codec_lock."""
        if self._codec is not None or not (self._sps and self._pps):
            return self._codec
        extradata = self._build_extradata()
        if self._prefer_hwaccel:
            for hw_type in _h264_hwaccels():
                ctx = self._try_hwaccel_locked(hw_type, extradata)
                if ctx is not None:
                    self._codec = ctx
                    self._hw_name = hw_type
                    log.info("AVC decode: hardware (%s)", hw_type)
                    return self._codec
        self._codec = self._make_sw_context(extradata)
        self._hw_name = None
        log.info("AVC decode: software")
        return self._codec

    def _try_hwaccel_locked(
        self, hw_type: str, extradata: bytes,
    ) -> Optional[av.codec.context.CodecContext]:
        try:
            from av.codec.hwaccel import HWAccel
            hw = HWAccel(device_type=hw_type)
            c = av.CodecContext.create("h264", "r", hwaccel=hw)
            c.extradata = extradata
            c.flags = _CODEC_FLAG_LOW_DELAY
            c.flags2 = _CODEC_FLAG2_FAST
            c.open()
            return c
        except Exception as e:
            log.info("AVC hwaccel %s unavailable: %s", hw_type, e)
            return None

    @staticmethod
    def _make_sw_context(extradata: bytes) -> av.codec.context.CodecContext:
        # SLICE threading (parallelise within a frame, no reordering/latency)
        # so software H.264 keeps pace with the 60fps stream on slower CPUs.
        c = av.CodecContext.create("h264", "r")
        c.extradata = extradata
        c.thread_type = "SLICE"
        c.thread_count = 0
        c.flags = _CODEC_FLAG_LOW_DELAY
        c.flags2 = _CODEC_FLAG2_FAST
        c.open()
        return c

    def start(self) -> None:
        if not (self._sps and self._pps):
            raise RuntimeError("set_params() must be called before start()")
        with self._codec_lock:
            self._ensure_codec_locked()

    # -- feed ----------------------------------------------------------

    def feed_burst(self, tile_nalu_cache: dict) -> None:
        for ti, nalus in tile_nalu_cache.items():
            for nalu in nalus:
                self.feed_nalu(nalu, ti)

    def feed_nalu(self, nalu: bytes, tile_idx: int, donl: Optional[int] = None) -> None:
        if not nalu or not (0 <= tile_idx < self.num_tiles):
            return
        t = h264_nal_type(nalu[0])
        bucket = self.nalu_counts_per_tile[tile_idx]
        bucket[t] = bucket.get(t, 0) + 1
        if t in (7, 8):  # SPS/PPS already in extradata
            # Apple's AVC stream carries params out-of-band in the 0x92 avcC
            # config, NOT in-band — verified: no type-7 arrives on resize. So
            # there's nothing to capture here; the session re-harvests the
            # avcC config and calls set_params()+restart() when the geometry
            # changes (see Session._maybe_reharvest_avc_config).
            return
        nb = nalu if isinstance(nalu, bytes) else bytes(nalu)
        is_key = _nal_is_keyframe(nb)
        # Post-restart keyframe gate. After a restart (incl. the session's
        # SSRC-adoption / VT-wedge restart) the DPB is empty, so feeding the
        # inter (P) slices that arrive before the next IDR makes the decoder
        # reference frames it never decoded → "reference frames N+1 exceeds
        # max" / -12909, and the session's restart-on-wedge then re-wedges on
        # the next P-slice flood — a permanent freeze on resize. Drop slices
        # until a keyframe re-roots the DPB. Direct-decode (no parser) has no
        # tolerance here, so this gate is what makes it safe; the FIR the
        # session sends on restart fetches the keyframe that clears it.
        if self._await_key:
            if is_key:
                self._await_key = False
            else:
                return
        # Each Apple tile-frame is one slice = one complete access unit, so we
        # build the av.Packet directly and decode() it — NO ctx.parse(). The
        # libav H.264 parser can't tell an AU is complete until the *next* slice
        # delimits it (Apple sends no AUD), so it holds every frame back ~one
        # frame-interval (~45ms @ 22fps) — invisible to the decode queue but
        # felt as input lag. Direct-decode emits immediately under LOW_DELAY.
        # Emitted frames route back to their source tile via frame.pts (a map,
        # not an in/out FIFO), surviving any decoder drop/reorder. Mirrors
        # HevcDecoder._decode_one. Runs under _codec_lock so a concurrent
        # restart()/close() can't free the context mid-decode.
        import time as _time
        published: list[int] = []
        with self._codec_lock:
            ctx = self._ensure_codec_locked()
            if ctx is None:
                return  # no params yet — don't register a pts that never drains
            # Every keyframe re-roots the shared DPB for all tiles, so re-mark
            # IDR observation on each one (mirrors HevcDecoder). Without this
            # the gate only ever marks the first frame: a tile that drops into
            # `keyframe_required` mid-stream (a post-SSRC-adoption P-frame
            # error) can never satisfy mark_clean's IDR-observed condition, so
            # it FIR-storms / grays out forever even as fresh IDRs arrive.
            if is_key:
                for _t in range(self.num_tiles):
                    self._gate.mark_idr_observed(_t)
            pkt = av.Packet(_NAL_START_CODE + nb)
            pts = self._next_pts
            pkt.pts = pts
            pkt.dts = pts
            self._next_pts += 1
            self._pts_to_tile[pts] = tile_idx
            self._pts_submit_t[pts] = _time.monotonic()
            if len(self._pts_to_tile) > _PTS_MAP_SOFT_MAX:
                cutoff = pts - _PTS_MAP_PRUNE_KEEP
                self._pts_to_tile = {
                    k: v for k, v in self._pts_to_tile.items() if k > cutoff
                }
                self._pts_submit_t = {
                    k: v for k, v in self._pts_submit_t.items() if k > cutoff
                }
            try:
                frames = ctx.decode(pkt)
            except Exception:
                # Decode raised → no frame will carry this pts; drop it so the
                # map doesn't leak the in-flight entry.
                self._pts_to_tile.pop(pts, None)
                self._pts_submit_t.pop(pts, None)
                self._gate.mark_decode_error(tile_idx)
                return
            for frame in frames:
                ti = self._pts_to_tile.pop(frame.pts, tile_idx)
                submit_t = self._pts_submit_t.pop(frame.pts, None)
                if submit_t is not None:
                    latency_ms = (_time.monotonic() - submit_t) * 1000
                    self._decode_latency_ms = (
                        0.1 * latency_ms + 0.9 * self._decode_latency_ms
                    )
                if not self._dpb_ready:
                    self._dpb_ready = True
                    for _t in range(self.num_tiles):
                        self._gate.mark_idr_observed(_t)
                slot = self._tiles[ti]
                with slot.lock:
                    slot.raw_frame = frame
                    slot.good_count += 1
                published.append(ti)
        # Notify outside the codec lock to avoid holding it across the callback.
        if self._on_frame_published is not None:
            for ti in published:
                self._on_frame_published(ti)

    # -- consume -------------------------------------------------------

    def get_frame(self, tile_idx: int) -> Optional[TileFrame]:
        if not self._dpb_ready:
            return None
        slot = self._tiles[tile_idx]
        with slot.lock:
            frame = slot.raw_frame
            count = slot.good_count
            already = count <= slot.last_evaluated_count
        if frame is None or already:
            return None
        tile_frame, had_error = _av_frame_to_tile(
            frame, self._reformatter, self._seen_fmts,
        )
        with slot.lock:
            slot.last_evaluated_count = count
        if tile_frame is None:
            return None
        if had_error:
            self._gate.mark_decode_error(tile_idx)
        else:
            self._gate.mark_clean(tile_idx)
            with slot.lock:
                slot.clean_count += 1
        if not self._gate.should_publish(tile_idx, tile_frame):
            return None
        return tile_frame

    def consume_fir_request(self) -> set:
        return self._gate.consume_fir_request()

    def tile_state(self, tile_idx: int):
        return self._gate.tile_state(tile_idx)

    @property
    def hw_accel(self) -> Optional[str]:
        return self._hw_name

    @property
    def bad_tiles(self) -> set:
        return self._gate.bad_tiles

    @property
    def decode_latency_ms(self) -> float:
        return self._decode_latency_ms

    @property
    def decode_queue_depth(self) -> int:
        return len(self._pts_to_tile)

    @property
    def decode_queue_cap(self) -> int:
        return 512

    @property
    def decode_queue_drops(self) -> int:
        return self._queue_full_drops

    @property
    def good_counts(self) -> list:
        return [t.good_count for t in self._tiles]

    @property
    def clean_counts(self) -> list:
        return [t.clean_count for t in self._tiles]

    def restart(self) -> None:
        """Tear down + rebuild the shared codec context. May fire from the
        stall-watchdog thread, so it takes _codec_lock to avoid freeing the
        context while feed_nalu is mid-decode. Resets the gate too (HEVC does
        this in _teardown) so post-restart publish/FIR decisions start clean."""
        with self._codec_lock:
            if self._codec is not None:
                try:
                    self._codec.close()
                except Exception:
                    pass
                self._codec = None
            self._pts_to_tile.clear()
            self._pts_submit_t.clear()
            self._dpb_ready = False
            self._await_key = True
            self._decode_latency_ms = 0.0
            self._gate.reset()
            if self._sps and self._pps:
                self._ensure_codec_locked()

    def close(self) -> None:
        with self._codec_lock:
            if self._codec is not None:
                try:
                    self._codec.close()
                except Exception:
                    pass
                self._codec = None
