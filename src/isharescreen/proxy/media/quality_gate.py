"""Per-tile decoder-recovery state.

The recovery logic is *not* heuristic. We do not look at the pixel
contents of decoded frames to guess whether the decoder concealed
something — that approach can never distinguish real gray content
(curtain mode, a settings panel, a dark theme) from concealment
fill, and false-positives on real content trigger unnecessary FIR
storms / decoder restarts.

Instead, recovery is driven by the two signals we *can* trust:

  1. **RTP sequence-number gaps** — tracked at the SRTP RX layer,
     surfaced via NACK retransmits. Ground truth for "we lost a
     packet."
  2. **libavcodec error reports** — `AVFrame.decode_error_flags` on
     each decoded frame, plus libav log callbacks for messages like
     "Could not find ref with POC N" that escape the API but are
     emitted to the log system by the decoder when it had to
     conceal. Ground truth for "the decoder concealed something."

This file just manages the per-tile FIR-pending set + an opaque
`bad_streak` counter for diagnostics. `mark_decode_error(i)` is the
single public hook other code calls when one of the trusted signals
fires.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from .tiles import TileFrame  # noqa: F401  (kept for type-annotation parity)


log = logging.getLogger(__name__)


# ── public state (kept for HUD-overlay round-trip compat) ────────────

STATE_INIT = "init"
STATE_OK = "ok"
STATE_HOLD = "hold"


@dataclass(slots=True, frozen=True)
class TileVisState:
    state: str
    mean: float = 0.0
    std: float = 0.0


@dataclass(slots=True)
class _TileState:
    bad_streak: int = 0       # decoder-reported errors since last recovery
    needs_real_frame: bool = False
    vis: TileVisState = field(default_factory=lambda: TileVisState(STATE_INIT))


class FrameQualityGate:
    """Tracks per-tile keyframe-required state with libwebrtc-style
    sticky semantics. `mark_decode_error(i)` flags a tile as needing
    a fresh IDR; the flag *persists* until two conditions are both
    observed: (1) an IDR NAL was fed into the decoder for that tile,
    and (2) a frame decoded cleanly after that IDR. This guards
    against the silent-gray failure mode where the IDR response to
    a FIR is lost in the same packet-loss event that broke the DPB:
    `consume_fir_request()` keeps re-emitting FIR for any flagged
    tile every `_RE_ARM_INTERVAL_S` regardless of whether libav has
    logged any new errors, until the decoder confirms recovery.

    Methods are not thread-safe — call from the decoder's frame-publish
    thread (or hold the decoder's lock around them). The FIR set is
    safe to read from another thread *as long as* no `mark_decode_error`
    call is in progress.
    """

    # Time between successive FIR re-emits for a tile that's still in
    # `keyframe_required`. libwebrtc uses 200 ms for an in-process
    # codec, but Apple's HP encoder needs more headroom: too-fast
    # re-emits caused overlapping IDR responses where the decoder
    # accepted the first and rejected the rest as "Duplicate POC",
    # leaving the tile in a chronic break+recover loop. 1.0 s lets
    # Apple's typical 100-300 ms IDR generate+transmit window settle
    # before we conclude the response was lost, while still feeling
    # snappy when a retry is genuinely needed.
    _RE_ARM_INTERVAL_S: float = 1.00
    # After this many re-emits without seeing recovery, log a warning.
    # Doesn't stop the loop — a stuck tile can still recover late, and
    # the user has the F12 manual fallback. ~8 s of attempts.
    _RE_ARM_CAP: int = 8
    # After observing an IDR for a tile, ignore decode errors on that
    # tile for this long. Apple's P-frames already in flight when our
    # FIR was processed reference pre-IDR POCs and will error on
    # decode against the freshly-reset DPB; that's expected drain
    # noise, not a real recovery failure. Without this window each
    # IDR cycle ends with the post-IDR P-frames re-flagging the tile,
    # which sends another FIR, which arrives during the next drain,
    # forever. 500 ms covers Apple's typical pre-IDR P-frame queue
    # drain on a LAN.
    _POST_IDR_GRACE_S: float = 0.50

    def __init__(
        self,
        num_tiles: int,
        *,
        enabled: bool = True,           # kept for API back-compat
    ) -> None:
        if num_tiles <= 0:
            raise ValueError("num_tiles must be positive")
        self._num_tiles = num_tiles
        self._states: list[_TileState] = [_TileState() for _ in range(num_tiles)]
        # Sticky set: a tile stays here until both clear-conditions
        # are met (see `mark_clean`). Replaces the old one-shot
        # `_fir_pending` semantics.
        self._keyframe_required: set[int] = set()
        # Per-tile flag that flips True the moment an IDR NAL is fed
        # into the decoder for that tile (set via `mark_idr_observed`).
        # Cleared together with `_keyframe_required[i]` once a clean
        # frame decodes. Without this we'd false-clear on the first
        # post-error P-frame that libav happens not to flag (which is
        # exactly the silent-gray case — the decoder is producing
        # corrupt output but not erroring on each frame).
        self._idr_observed: set[int] = set()
        # Last time a FIR was actually emitted (drained) for each tile.
        # Initialised to 0 so the first error per tile fires immediately.
        import time as _time
        self._fir_last_t: list[float] = [0.0] * num_tiles
        # Per-tile re-emit count since last recovery. Reset on
        # `mark_idr_decoded`. Drives the cap warning.
        self._fir_attempts: list[int] = [0] * num_tiles
        self._cap_warned: list[bool] = [False] * num_tiles
        self._time = _time
        self.flicker_events = 0  # diagnostic counter
        # Rate-limit the per-tile error/recovered INFO logs. A tile
        # whose next-P-frame keeps erroring after each recovery (e.g.,
        # tile 0 with heavy motion at the top of the screen, or
        # encoder edge-case content) was flapping at ~500 ms and
        # producing 30+ INFO lines/min. Log only the first transition
        # per tile per `_LOG_THROTTLE_S`; subsequent flapping is at
        # DEBUG so it's still available with --verbose.
        self._LOG_THROTTLE_S: float = 5.0
        self._last_err_log_t: list[float] = [0.0] * num_tiles
        self._last_rec_log_t: list[float] = [0.0] * num_tiles
        self._cycle_count: list[int] = [0] * num_tiles
        # Per-tile timestamp of the most recent IDR observation.
        # Powers `_POST_IDR_GRACE_S`: errors within this window after
        # an IDR are noise from in-flight pre-IDR P-frames draining,
        # not real recovery failures.
        self._idr_observed_at: list[float] = [0.0] * num_tiles

    # -- main "publish" hook --------------------------------------------
    # Always publishes. Kept as a callable for API compatibility with
    # the old gate; no pixel inspection is done.
    def should_publish(self, tile_idx: int, tile: TileFrame) -> bool:
        state = self._states[tile_idx]
        state.vis = TileVisState(STATE_OK)
        return True

    # -- decoder-error path (the only escalation source) ----------------
    def mark_decode_error(self, tile_idx: int) -> None:
        """Trusted signal that libavcodec concealed / failed for this
        tile. Adds the tile to `keyframe_required` (sticky) — the
        session's tx-tick will FIR for it now and again every
        `_RE_ARM_INTERVAL_S` until recovery is observed.
        """
        if tile_idx < 0 or tile_idx >= self._num_tiles:
            return
        # Post-IDR grace: errors right after the tile's IDR are
        # almost always in-flight pre-IDR P-frames decoding against
        # the freshly-reset DPB. Suppress them so they don't trigger
        # an immediate re-FIR cycle.
        now = self._time.monotonic()
        if (self._idr_observed_at[tile_idx] > 0
                and now - self._idr_observed_at[tile_idx]
                    < self._POST_IDR_GRACE_S):
            log.debug(
                "tile %d post-IDR error suppressed (%.0f ms after IDR)",
                tile_idx,
                (now - self._idr_observed_at[tile_idx]) * 1000,
            )
            return
        state = self._states[tile_idx]
        state.bad_streak += 1
        state.needs_real_frame = True
        if tile_idx not in self._keyframe_required:
            self._keyframe_required.add(tile_idx)
            self._idr_observed.discard(tile_idx)
            self._fir_attempts[tile_idx] = 0
            self._cap_warned[tile_idx] = False
            self.flicker_events += 1
            self._cycle_count[tile_idx] += 1
            # Per-tile decode-error notice stays at DEBUG -- the
            # session's `DPB break: N events ...` WARNING is the
            # canonical user-facing signal (one per real loss event,
            # not one per tile). Without the demotion a single DPB
            # break paints four INFO lines in the panel even though
            # they're describing the same incident.
            if now - self._last_err_log_t[tile_idx] >= self._LOG_THROTTLE_S:
                log.debug("tile %d decode error → keyframe required", tile_idx)
                self._last_err_log_t[tile_idx] = now
            else:
                log.debug("tile %d decode error → keyframe required (cycle %d)",
                          tile_idx, self._cycle_count[tile_idx])

    # -- IDR observation hook -------------------------------------------
    def mark_idr_observed(self, tile_idx: int, *, suspicious: bool = False) -> None:
        """Decoder calls this when an IDR NAL (unit type 16-21) is
        fed for this tile. Half of the two-condition clear; the other
        half (a clean decoded frame after this point) lives in
        `mark_clean`.

        `suspicious=True` means the IDR's payload size was far smaller
        than recent IDRs from the same encoder — observed pattern:
        Apple's HP encoder occasionally responds to a FIR with a
        "minimal" IDR (~30 % of normal size, no fresh parameter sets)
        that parses as `nt=20` but doesn't actually clear the decoder's
        accumulated drift. iss treated those as recovery and stopped
        FIRing, leaving the stream visually stuck on a gray placeholder
        until F12. When the decoder flags an IDR as suspicious we
        deliberately do NOT add the tile to `_idr_observed`, so
        `mark_clean`'s two-condition discard won't trigger and the
        sticky FIR loop keeps firing until a real IDR arrives.
        """
        if tile_idx < 0 or tile_idx >= self._num_tiles:
            return
        # Stamp the time so the post-IDR grace window engages whether
        # or not the tile is currently in keyframe_required. This makes
        # the grace cover *every* IDR cycle, including ones triggered
        # by Apple's natural I-frame cadence (not just ones in response
        # to our FIRs).
        self._idr_observed_at[tile_idx] = self._time.monotonic()
        if suspicious:
            # Don't add to `_idr_observed` -- `mark_clean`'s two-condition
            # discard won't fire, `keyframe_required` stays sticky, and
            # `consume_fir_request` keeps issuing FIRs at the natural
            # `_RE_ARM_INTERVAL_S` cadence until either a real IDR
            # arrives (recovered) or `_RE_ARM_CAP` attempts are spent
            # (cap fires, single "gave up" warning, `keyframe_required`
            # cleared -- user F12 re-arms; next natural IDR also recovers).
            # We intentionally do NOT reset `_fir_attempts` here: doing so
            # produced an infinite-FIR storm when Apple kept emitting
            # small IDRs (observed in the field as ~60 FIRs/min). The
            # per-IDR `IDR arrival ... suspicious=True` DEBUG line in
            # `hevc.py` is the diagnostic trail; the once-per-incident
            # cap warning in `consume_fir_request` is the user-visible
            # signal.
            return
        if tile_idx in self._keyframe_required:
            self._idr_observed.add(tile_idx)

    # -- "I just published a clean frame" hook --------------------------
    # Decoders call this when a tile produces output without a
    # decode_error_flag set. Resets the streak; if an IDR was already
    # observed for this tile since the last error, also clears the
    # keyframe-required flag (libwebrtc two-condition).
    def mark_clean(self, tile_idx: int) -> None:
        if tile_idx < 0 or tile_idx >= self._num_tiles:
            return
        state = self._states[tile_idx]
        state.bad_streak = 0
        state.needs_real_frame = False
        if (tile_idx in self._keyframe_required
                and tile_idx in self._idr_observed):
            self._keyframe_required.discard(tile_idx)
            self._idr_observed.discard(tile_idx)
            self._fir_attempts[tile_idx] = 0
            self._cap_warned[tile_idx] = False
            now = self._time.monotonic()
            if now - self._last_rec_log_t[tile_idx] >= self._LOG_THROTTLE_S:
                log.debug("tile %d recovered (IDR + clean decode)", tile_idx)
                self._last_rec_log_t[tile_idx] = now
            else:
                log.debug("tile %d recovered (IDR + clean decode)", tile_idx)
            # If the tile keeps cycling (recovers + fails repeatedly),
            # warn once per throttle window with the cycle count so the
            # underlying-bug case stays visible without firehose logs.
            if self._cycle_count[tile_idx] >= 3:
                if now - self._last_err_log_t[tile_idx] >= self._LOG_THROTTLE_S:
                    log.warning(
                        "tile %d cycled %d times in %.0fs — keep recovering "
                        "but next P-frame keeps erroring (likely encoder/"
                        "content edge case)",
                        tile_idx, self._cycle_count[tile_idx],
                        self._LOG_THROTTLE_S,
                    )
                    self._cycle_count[tile_idx] = 0

    # -- FIR consumption -------------------------------------------------
    def consume_fir_request(self) -> set[int]:
        """Returns the set of tile indices to send FIR for on this tick.

        Apple's HP encoder only emits IDRs on the base SSRC (= tile 0);
        FIRs targeting other tiles' SSRCs don't produce any response.
        And one tile-0 IDR resets the shared codec context, which clears
        all tiles via `mark_idr_observed`'s fan-out. So the right
        recovery primitive is "if anything in keyframe_required, FIR
        tile 0 once, subject to re-arm cooldown" — never FIRing tiles
        1-3 saves wire bytes + avoids Apple processing them as no-ops.

        Returns at most {0}, gated by `_RE_ARM_INTERVAL_S` against the
        last tile-0 FIR time."""
        if not self._keyframe_required:
            return set()
        # Cap already hit: stop emitting FIRs but keep keyframe_required
        # so _check_stall()'s Path C can detect the persistent stuck state
        # and escalate to a decoder restart after a cooldown.
        # force_keyframe_all() (F12) clears this flag to re-arm the loop.
        if self._cap_warned[0]:
            return set()
        now = self._time.monotonic()
        if now - self._fir_last_t[0] < self._RE_ARM_INTERVAL_S:
            return set()
        self._fir_last_t[0] = now
        self._fir_attempts[0] += 1
        if (self._fir_attempts[0] >= self._RE_ARM_CAP
                and not self._cap_warned[0]):
            log.warning(
                "Apple not responding to FIR after %d attempts "
                "(%d tiles still need recovery); press F12 to retry",
                self._fir_attempts[0], len(self._keyframe_required),
            )
            self._cap_warned[0] = True
            # Do NOT clear keyframe_required or idr_observed here.
            # Keeping them lets _check_stall() detect the persistent
            # stall and escalate to an automatic decoder restart after
            # 30 s (Path C).  Previously cleared here, which hid the
            # stuck state from the session entirely.
            return set()
        return {0}

    # -- introspection ---------------------------------------------------
    def tile_state(self, tile_idx: int) -> TileVisState:
        return self._states[tile_idx].vis

    def needs_real_frame(self, tile_idx: int) -> bool:
        return self._states[tile_idx].needs_real_frame

    # -- lifecycle -------------------------------------------------------
    def reset(self, tile_idx: Optional[int] = None) -> None:
        if tile_idx is None:
            for i in range(self._num_tiles):
                self._states[i] = _TileState()
                self._fir_attempts[i] = 0
                self._cap_warned[i] = False
                self._cycle_count[i] = 0
                self._last_err_log_t[i] = 0.0
                self._last_rec_log_t[i] = 0.0
                self._idr_observed_at[i] = 0.0
            self._keyframe_required.clear()
            self._idr_observed.clear()
        else:
            self._states[tile_idx] = _TileState()
            self._keyframe_required.discard(tile_idx)
            self._idr_observed.discard(tile_idx)
            self._fir_attempts[tile_idx] = 0
            self._cap_warned[tile_idx] = False
            self._cycle_count[tile_idx] = 0
            self._last_err_log_t[tile_idx] = 0.0
            self._last_rec_log_t[tile_idx] = 0.0
            self._idr_observed_at[tile_idx] = 0.0

    # -- F12 manual fallback funnel -------------------------------------
    def force_keyframe_all(self) -> None:
        """Sets keyframe_required for every tile (single funnel for the
        F12 hotkey + any other 'just refresh everything' caller)."""
        # Explicitly reset tile-0's cap state first.  mark_decode_error()
        # only resets _fir_attempts / _cap_warned when the tile is NOT
        # already in keyframe_required; if the cap fired and keyframe_required
        # still holds the stuck tile, mark_decode_error is a no-op and FIR
        # would never resume without this explicit reset.
        self._fir_attempts[0] = 0
        self._cap_warned[0] = False
        for ti in range(self._num_tiles):
            self.mark_decode_error(ti)


__all__ = [
    "FrameQualityGate",
    "STATE_HOLD",
    "STATE_INIT",
    "STATE_OK",
    "TileVisState",
]
