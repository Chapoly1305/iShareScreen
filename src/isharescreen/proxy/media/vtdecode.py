"""VideoToolbox-native HEVC decoder (macOS) — a drop-in for `HevcDecoder`.

Why this exists: libav's HEVC decoder via its VideoToolbox *hwaccel* glue
mismanages the DPB on Apple's Screen-Sharing stream — it periodically loses a
recent reference and then BLOCKS (`avcodec_send_packet` → EAGAIN) on a missing
ref, which forces our drain/drop recovery, which gaps the stream, which
cascades into a multi-second freeze (see the decode-stall notes). The native
viewer never hits this because it decodes through a `VTDecompressionSession`
directly: VideoToolbox does its own bitstream parsing + DPB and simply conceals
the odd missing-ref frame without ever blocking. This module does the same.

Validated offline against captured feeds: decodes Apple HEVC RExt 4:4:4 to
`kCVPixelFormatType_444YpCbCr8BiPlanarFullRange` (`444f` = nv24) and never
blocks. The two output planes (Y r8 + interleaved UV) map straight onto the
existing nv24 *passthrough* `TileFrame` (v is None) and the biplanar GPU path.

One shared session decodes all tiles' NALUs (same shared-context model as the
libav path — Apple uses cross-tile references). Output is dispatched to the
right tile by a per-decode closure. macOS-only; import guarded.
"""
from __future__ import annotations

import logging
import ctypes
import queue
import os
import struct
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .tiles import TileFrame
from .nalu import IDR_RANGE
from .quality_gate import FrameQualityGate

log = logging.getLogger(__name__)

# Imported lazily / guarded so the module is importable (and skippable) off
# macOS. `available()` reports whether the native path can be used.
try:
    import VideoToolbox as _VT
    import CoreMedia as _CM
    import Quartz as _Q
    _VT_OK = True
except Exception:  # pragma: no cover - non-macOS / missing pyobjc
    _VT_OK = False

_LOCK_READONLY = 0x00000001                       # kCVPixelBufferLock_ReadOnly
_NV24_FMT = 0x34343466                             # '444f'
_VT_DEBUG = os.environ.get("ISS_VT_DEBUG") == "1"
# Diagnostic: dump every VCL NALU we feed VT (Annex-B framed, in feed order)
# so the feed ORDER can be analysed offline against POC. Same env var as the
# libav path; only the active decoder writes.
_NALU_DUMP_PATH = os.environ.get("ISS_NALU_DUMP")
_nalu_dump_f = open(_NALU_DUMP_PATH, "wb") if _NALU_DUMP_PATH else None


def available() -> bool:
    """True if VideoToolbox + pyobjc are importable (macOS with the deps)."""
    return _VT_OK


@dataclass(slots=True)
class _VTSlot:
    """Latest decoded `TileFrame` for one tile + a monotonic sequence so an
    out-of-order async callback can't replace a newer frame with an older one."""
    frame: Optional[TileFrame] = None
    seq: int = 0
    good_count: int = 0
    last_evaluated_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class _PixelBufferLease:
    """Retain and lock a CVPixelBuffer until its last exported plane dies."""
    def __init__(self, buf):
        self.buf = buf

    def plane(self, base, length):
        array = (ctypes.c_ubyte * length).from_buffer(base.as_buffer(length))
        array._pixel_buffer_lease = self
        return memoryview(array).cast("B").toreadonly()

    def __del__(self):
        try:
            _Q.CVPixelBufferUnlockBaseAddress(self.buf, _LOCK_READONLY)
        except Exception:
            pass


class VTHevcDecoder:
    """Native VideoToolbox HEVC decoder. Mirrors `HevcDecoder`'s public API:
    `set_params`, `feed_burst`, `feed_nalu`, `get_frame`, `restart`, `close`,
    plus the `_gate` / `good_counts` / `hw_accel` attributes session.py reads."""

    def __init__(
        self,
        num_tiles: int,
        *,
        prefer_hwaccel: bool = True,           # accepted for signature parity
        enable_quality_gate: bool = True,
        on_frame_published: Optional[Callable[[int], None]] = None,
    ) -> None:
        if not _VT_OK:
            raise RuntimeError("VideoToolbox/pyobjc not available")
        if num_tiles <= 0:
            raise ValueError("num_tiles must be positive")
        self.num_tiles = num_tiles
        self._on_frame_published = on_frame_published
        self._tiles = [_VTSlot() for _ in range(num_tiles)]
        # Successful output carries its submitted DONL into Session's LTR ACK.
        # Failed submissions and decode errors never advance this anchor.
        self.last_clean_donl: list = [None] * num_tiles
        # Per-tile NAL-unit-type histogram (diagnostic; read by the snapshot).
        self.nalu_counts_per_tile: list = [{} for _ in range(num_tiles)]
        # session.py reads `_decoder._gate._states[ti].bad_streak` (stall
        # watchdog) and `_decoder._rps_tracker` (ISS_DPB_TRACE only — libav
        # log path, never fires for VT). Provide a real gate, a None tracker.
        self._gate = FrameQualityGate(num_tiles)
        self._rps_tracker = None
        # Display name surfaced as the "decoder" field in the control-socket
        # snapshot + the TUI header line. Distinct from the libav hwaccel name
        # ("videotoolbox") so an operator can see which path is live: this
        # direct-VTDecompressionSession path vs libav's VT *hwaccel* glue.
        self._hw_name = "videotoolbox-native"

        self._fmt = None                       # CMVideoFormatDescription
        self._session = None                   # VTDecompressionSession
        self._params: Optional[tuple] = None   # (vps, sps, (pps...))
        self._seq = 0                          # monotonic decode counter
        self._feed_total = 0
        self._feeds_at_last_pub = 0
        self._last_pub_t = 0.0
        self._last_err_fir_t = 0.0
        self._restart_suppressed = 0
        self._decode_errors = 0                # benign VT conceals (diagnostic)
        self._lock = threading.Lock()          # guards session create/teardown
        self._seen_fmt = False
        self._fmt_warned = False                # H1: warn once on non-nv24 output
        # Decode latency monitoring: EMA of submit→frame round-trip (ms).
        # Native submission stays synchronous on the dedicated ordered worker.
        # Queue wait is reported separately from submit-to-output latency.
        self._decode_latency_ms: float = 0.0
        self._submit_t: float = 0.0
        self._queue = queue.Queue(maxsize=max(2, num_tiles * 2))
        self._stop = threading.Event()
        self._worker = None
        self._feedback_wakeup = None
        self._queue_wait_ms = 0.0
        self._inflight = 0

    # -- params / session ---------------------------------------------------

    def set_params(self, vps: bytes, sps: bytes, all_pps: dict) -> None:
        """Build the CMVideoFormatDescription + decompression session from the
        VPS/SPS/PPS harvested at session start. Re-callable (rebuilds)."""
        pps = tuple(bytes(p) for p in all_pps.values()) if all_pps else ()
        self._params = (bytes(vps), bytes(sps), pps)
        if self._worker is not None and self._worker.is_alive():
            self._queue.join()
        with self._lock:
            for slot in self._tiles:
                with slot.lock:
                    slot.frame = None
                    slot.last_evaluated_count = slot.good_count
            self._build_session_locked()

    def _build_session_locked(self) -> None:
        if self._params is None:
            return
        vps, sps, pps = self._params
        psets = (vps, sps) + pps
        sizes = tuple(len(p) for p in psets)
        st, fmt = _CM.CMVideoFormatDescriptionCreateFromHEVCParameterSets(
            None, len(psets), psets, sizes, 4, None, None)
        if st != 0 or fmt is None:
            raise RuntimeError(f"CMVideoFormatDescription create failed: {st}")
        # Ask for the native 4:4:4 biplanar output (nv24) so no conversion.
        attrs = {_Q.kCVPixelBufferPixelFormatTypeKey: _NV24_FMT}
        st, session = _VT.VTDecompressionSessionCreate(
            None, fmt, None, attrs, None, None)
        if st != 0 or session is None:
            # fall back to letting VT pick the format
            st, session = _VT.VTDecompressionSessionCreate(
                None, fmt, None, None, None, None)
            if st != 0 or session is None:
                raise RuntimeError(f"VTDecompressionSession create failed: {st}")
        self._teardown_session_locked()
        self._fmt = fmt
        self._session = session
        try:
            status, using_hw = _VT.VTSessionCopyProperty(
                session, _VT.kVTDecompressionPropertyKey_UsingHardwareAcceleratedVideoDecoder,
                None, None)
            log.info("VideoToolbox hardware decoder: %s", using_hw if status == 0 else "unknown")
        except Exception:
            log.debug("VT hardware selection property unavailable", exc_info=True)
        log.info("VTHevcDecoder: native VideoToolbox session ready (%d tiles)",
                 self.num_tiles)

    def _teardown_session_locked(self) -> None:
        if self._session is not None:
            try:
                _VT.VTDecompressionSessionWaitForAsynchronousFrames(self._session)
                _VT.VTDecompressionSessionInvalidate(self._session)
            except Exception as e:
                log.debug("VT session teardown: %s", e)
            self._session = None

    # -- feed ---------------------------------------------------------------

    def feed_burst(self, tile_nalu_cache: dict) -> None:
        """Feed the session-start IDR burst in ROUND-ROBIN (tile-0 NAL i,
        tile-1 NAL i, … then NAL i+1) — the order Apple's encoder emits frames
        in real time, so VT's DPB sees POCs in natural decode order. (Feeding
        each tile's NALUs grouped instead corrupts the cross-tile references.)"""
        if not tile_nalu_cache:
            return
        max_burst = max((len(v) for v in tile_nalu_cache.values()), default=0)
        for idx in range(max_burst):
            for ti in sorted(tile_nalu_cache):
                nalus = tile_nalu_cache[ti]
                if idx < len(nalus):
                    self.feed_nalu(nalus[idx], ti)

    def feed_nalu(self, nalu, tile_idx, donl=None):
        if self._worker is None:
            return self._decode_nalu(nalu, tile_idx, donl)
        while not self._stop.is_set():
            try:
                self._queue.put((bytes(nalu), tile_idx, donl, time.perf_counter()), timeout=0.01)
                return
            except queue.Full:
                continue  # Bound backlog without dropping reference pictures.

    def set_feedback_wakeup(self, callback):
        self._feedback_wakeup = callback

    def _decode_loop(self):
        try:
            while not self._stop.is_set():
                try:
                    nalu, ti, donl, queued = self._queue.get(timeout=0.02)
                except queue.Empty:
                    continue
                try:
                    self._queue_wait_ms = (time.perf_counter() - queued) * 1000
                    self._inflight = 1
                    with self._lock:
                        self._decode_nalu(nalu, ti, donl)
                except Exception:
                    log.exception("native decode worker failed for tile %d", ti)
                    self._gate.mark_decode_error(ti)
                    if self._feedback_wakeup is not None:
                        self._feedback_wakeup()
                finally:
                    self._inflight = 0
                    self._queue.task_done()
        finally:
            while True:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except queue.Empty:
                    break
            if self._stop.is_set():
                with self._lock:
                    self._teardown_session_locked()

    def _decode_nalu(self, nalu: bytes, tile_idx: int,
                  donl: Optional[int] = None) -> None:
        """Wrap one NAL unit in a length-prefixed CMSampleBuffer and submit it
        for async decode. The output is dispatched to `tile_idx` by the closure
        below. No RPS pre-check / no drop: VT manages its own DPB and conceals
        missing refs instead of blocking.

        `donl` is carried through successful output for the host's LTR ACK;
        VideoToolbox still owns its own DPB and reference selection."""
        session = self._session
        if session is None or not nalu:
            return
        t = (nalu[0] >> 1) & 0x3F
        if 0 <= tile_idx < self.num_tiles:
            b = self.nalu_counts_per_tile[tile_idx]
            b[t] = b.get(t, 0) + 1
        if t in IDR_RANGE:
            # A tile-0 IDR re-roots the shared context for ALL tiles (Apple
            # IDRs on tile 0 only; cross-tile refs re-root the rest), so fan
            # out — matches the libav path. This is half of the gate's
            # two-condition `keyframe_required` clear; `mark_clean` on the
            # next good frame is the other half. Without it the gate — armed
            # by an SSRC adoption or a stuck event from session.py, not by VT
            # (VT decodes fine) — stays armed forever and FIR-storms ("4 tiles
            # still need recovery; Apple not responding to FIR").
            for ti in range(self.num_tiles):
                self._gate.mark_idr_observed(ti)
        # Only VCL slices (0-31) are decode frames. VPS/SPS/PPS (32-34), SEI
        # (39/40) etc. reached the decoder inline on the libav path, but VT
        # takes its parameter sets from the CMVideoFormatDescription — feeding
        # a non-VCL NALU as a sample buffer just produces a decode error.
        if t >= 32:
            return
        if _nalu_dump_f is not None:
            _nalu_dump_f.write(b"\x00\x00\x00\x01" + (
                nalu if isinstance(nalu, bytes) else bytes(nalu)))
        self._feed_total += 1
        if _VT_DEBUG and self._feed_total % 200 == 0:
            log.info("VTdbg feed/tile=%s good=%s errs=%d",
                     [sum(d.values()) for d in self.nalu_counts_per_tile],
                     self.good_counts, self._decode_errors)
        sb = self._make_sample(nalu)
        if sb is None:
            return
        self._seq += 1
        seq = self._seq
        self._submit_t = time.monotonic()
        def _handler(status, info_flags, image_buffer, pts, dur,
                     _ti=tile_idx, _seq=seq, _donl=donl):
            self._on_decoded(_ti, _seq, status, image_buffer, _donl)
        try:
            # Preserve deterministic reference order on the decode worker.
            # Receive and UI threads never execute this native call.
            result = _VT.VTDecompressionSessionDecodeFrameWithOutputHandler(
                session, sb, 0, None, _handler)
            status = result[0] if isinstance(result, tuple) else result
            if status:
                # VT does not call the output handler after a submit error.
                self._on_decoded(tile_idx, seq, status, None, donl)
        except Exception as e:
            log.debug("VT decode submit failed (tile %d): %s", tile_idx, e)
            self._on_decoded(tile_idx, seq, -1, None, donl)

    def _make_sample(self, nalu: bytes):
        payload = struct.pack(">I", len(nalu)) + (
            nalu if isinstance(nalu, bytes) else bytes(nalu))
        st, bb = _CM.CMBlockBufferCreateWithMemoryBlock(
            None, None, len(payload), None, None, 0, len(payload), 0, None)
        if st != 0 or bb is None:
            return None
        _CM.CMBlockBufferReplaceDataBytes(payload, bb, 0, len(payload))
        st, sb = _CM.CMSampleBufferCreateReady(
            None, bb, self._fmt, 1, 0, (), 1, (len(payload),), None)
        if st != 0:
            return None
        return sb

    # -- output -------------------------------------------------------------

    def _on_decoded(self, tile_idx, seq, status, image_buffer, donl=None) -> None:
        """Publish retained Y + interleaved-UV planes to the latest tile slot.

        On a decode error we do NOT touch the gate's `bad_streak`: VideoToolbox
        conceals the odd missing-ref frame and recovers on Apple's next IDR
        (~1.7 s) on its own — it never BLOCKS the way libav's hwaccel does. The
        gate's decode-error path is tuned for that libav wedge; driving it from
        VT's benign conceals just makes the session.py Path-B watchdog restart
        the session in a loop. A genuine VT freeze (no frames at all) is still
        caught by the gap-based Path-A watchdog. We only count errors for the
        diagnostic snapshot."""
        if status != 0 or image_buffer is None:
            self._decode_errors += 1
            # Request a recovery keyframe EARLY (throttled) rather than waiting
            # for session.py's 3 s "stuck" watchdog. A single missing/undecodable
            # frame breaks the inter-frame chain → every later frame conceals
            # until an IDR; Apple answers a FIR in ~0.2 s, so FIRing on the first
            # error of a burst turns a ~3 s freeze into a sub-second blip. The
            # gate's FIR send is itself rate-limited, and the 0.5 s throttle here
            # keeps bad_streak low so the Path-B watchdog never trips (and VT's
            # restart() is a no-op regardless) — no storm, no cascade.
            now = time.monotonic()
            if (0 <= tile_idx < self.num_tiles
                    and now - self._last_err_fir_t >= 0.5):
                self._last_err_fir_t = now
                self._gate.mark_decode_error(tile_idx)
                if self._feedback_wakeup is not None:
                    self._feedback_wakeup()
            return
        tile = self._cvbuf_to_tile(image_buffer)
        if tile is None:
            return
        slot = self._tiles[tile_idx]
        with slot.lock:
            if seq < slot.seq:                 # a newer frame already landed
                return
            slot.frame = tile
            slot.seq = seq
            slot.good_count += 1
        # Diagnostic: measure the session-wide publish gap. _on_decoded is
        # single-threaded (ordered decode worker), so the gap
        # between any two calls is the real publish gap. A long gap with MANY
        # feeds-in-between = VT stalled while being fed; with ~0 feeds = the
        # RX/host starved us. This is what the Path-A "decoder stuck" watchdog
        # trips on.
        now = time.monotonic()
        if self._last_pub_t > 0.0:
            gap = now - self._last_pub_t
            if gap > 1.5:
                fed = self._feed_total - self._feeds_at_last_pub
                if fed > 30:
                    # Fed but couldn't decode for >1.5 s = a real decode stall.
                    log.warning("VT decode stall %.2fs — fed %d NALUs, all "
                                "concealed (broken ref chain)", gap, fed)
                elif _VT_DEBUG:
                    # ~0 feeds = host/RX sent nothing (static screen / idle).
                    log.info("VT publish gap %.2fs — host idle (fed %d)", gap, fed)
        self._last_pub_t = now
        self._feeds_at_last_pub = self._feed_total
        if self._submit_t > 0.0:
            latency_ms = (now - self._submit_t) * 1000
            self._decode_latency_ms = 0.1 * latency_ms + 0.9 * self._decode_latency_ms
        # The clean-frame half of the gate's two-condition recovery clear
        # (pairs with mark_idr_observed). Lets `keyframe_required` discharge
        # so the FIR storm stops once a real frame flows again.
        self._gate.mark_clean(tile_idx)
        # Ack decoded output, never packet receipt or failed/concealed output.
        if donl is not None:
            self.last_clean_donl[tile_idx] = donl
        if self._on_frame_published is not None:
            try:
                self._on_frame_published(tile_idx)
            except Exception as e:
                log.debug("on_frame_published raised: %s", e)

    def _cvbuf_to_tile(self, buf) -> Optional[TileFrame]:
        if _Q.CVPixelBufferLockBaseAddress(buf, _LOCK_READONLY) != 0:
            return None
        lease = _PixelBufferLease(buf)
        try:
            w = _Q.CVPixelBufferGetWidth(buf)
            h = _Q.CVPixelBufferGetHeight(buf)
            if _Q.CVPixelBufferGetPlaneCount(buf) < 2:
                return None
            # H1: only nv24 (444f) is the biplanar 4:4:4 layout the rest of the
            # pipeline assumes. The fallback decode session (set_params) can
            # hand us a VT-chosen format (e.g. 4:2:0 biplanar) that would pass
            # the plane-count guard above and be mislabeled 4:4:4 → garbage
            # chroma. Reject it rather than render garbage.
            fmt = _Q.CVPixelBufferGetPixelFormatType(buf)
            if fmt != _NV24_FMT:
                if not self._fmt_warned:
                    self._fmt_warned = True
                    log.warning("VTHevcDecoder: non-nv24 output format 0x%x — "
                                "dropping frames (expected 444f)", fmt)
                return None
            y_rb = _Q.CVPixelBufferGetBytesPerRowOfPlane(buf, 0)
            uv_rb = _Q.CVPixelBufferGetBytesPerRowOfPlane(buf, 1)
            y_h = _Q.CVPixelBufferGetHeightOfPlane(buf, 0)
            uv_h = _Q.CVPixelBufferGetHeightOfPlane(buf, 1)
            y_base = _Q.CVPixelBufferGetBaseAddressOfPlane(buf, 0)
            uv_base = _Q.CVPixelBufferGetBaseAddressOfPlane(buf, 1)
            if y_base is None or uv_base is None:
                return None
            y_bytes = lease.plane(y_base, y_rb * y_h)
            uv_bytes = lease.plane(uv_base, uv_rb * uv_h)
        except Exception as e:
            log.debug("CVPixelBuffer read failed: %s", e)
            return None
        if not self._seen_fmt:
            self._seen_fmt = True
            log.info("VTHevcDecoder output: nv24 %dx%d (y_stride=%d uv_stride=%d)",
                     w, h, y_rb, uv_rb)
        # nv24 passthrough: v is None, u carries the interleaved UV plane.
        return TileFrame(
            y=y_bytes, u=uv_bytes, v=None,
            width=w, height=h,
            y_stride=y_rb,
            uv_stride=uv_rb,
            chroma_width=w,
            chroma_height=h,
            owner=lease,
        )

    # -- consumer -----------------------------------------------------------

    def get_frame(self, tile_idx: int) -> Optional[TileFrame]:
        slot = self._tiles[tile_idx]
        with slot.lock:
            frame = slot.frame
            count = slot.good_count
            if frame is None or count <= slot.last_evaluated_count:
                return None
            slot.last_evaluated_count = count
            return frame

    def consume_fir_request(self) -> set:
        """Tile indices needing a fresh IDR (read + cleared) — gate-backed."""
        return self._gate.consume_fir_request()

    def tile_state(self, tile_idx: int):
        """Gate HUD-state snapshot for one tile."""
        return self._gate.tile_state(tile_idx)

    @property
    def hw_accel(self) -> Optional[str]:
        return self._hw_name

    @property
    def good_counts(self) -> list:
        return [t.good_count for t in self._tiles]

    @property
    def clean_counts(self) -> list:
        """Per-tile CLEAN (non-concealed) published-frame totals, the signal
        session.py uses to flag GRAY tiles (high frame rate, ~0 clean = gray).
        VideoToolbox conceals missing refs internally at Apple quality and does
        not publish the libav-style persistent-gray frames this metric exists
        to catch, so every published (good) frame counts as clean. Aliasing
        good_counts keeps API parity with HevcDecoder.clean_counts."""
        return [t.good_count for t in self._tiles]

    @property
    def bad_tiles(self) -> set:
        """Tiles the gate considers gray/concealing and awaiting recovery.
        Delegates to the gate (decoder-agnostic), matching HevcDecoder."""
        return self._gate.bad_tiles

    @property
    def decode_latency_ms(self) -> float:
        return self._decode_latency_ms

    @property
    def decode_queue_depth(self) -> int:
        return self._queue.qsize() + self._inflight

    @property
    def decode_queue_cap(self) -> int:
        return self._queue.maxsize + 1

    @property
    def decode_queue_drops(self) -> int:
        return 0

    # -- lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if self._worker is None:
            self._worker = threading.Thread(target=self._decode_loop, name="iss-vt-decode", daemon=True)
            self._worker.start()

    def restart(self) -> None:
        """DPB-PRESERVING no-op by design. session.py calls restart() on SSRC
        adoption and from the stall watchdog to "reset" the decoder — but for
        VideoToolbox a rebuild WIPES the DPB, producing a ~3 s
        all-frames-conceal burst until the next IDR, which re-trips the
        watchdog → restart → a self-sustaining freeze cascade (the same
        flush-is-the-engine pattern the libav path suffers). VT doesn't need
        it: it re-roots on the IDR the paired request_fir() pulls (or Apple's
        natural cadence) while keeping its DPB intact, so every "restart" path
        degrades to native-aligned FIR-only recovery. Only set_params() — a
        real format/resolution change with a new SPS — rebuilds the session."""
        self._restart_suppressed += 1
        if _VT_DEBUG:
            log.info("VT restart() suppressed (DPB-preserving, #%d) — "
                     "recovering via FIR/IDR, not a session wipe",
                     self._restart_suppressed)

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            if self._worker is not threading.current_thread():
                self._worker.join(timeout=2.0)
        else:
            with self._lock:
                self._teardown_session_locked()
        for slot in self._tiles:
            with slot.lock:
                slot.frame = None


__all__ = ["VTHevcDecoder", "available"]
