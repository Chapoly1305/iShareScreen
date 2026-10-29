"""Packet-loss recovery tests.

These reproduce, in fast deterministic unit tests, what a lossy network does
to the desktop recovery pipeline. Packet loss shows up at the decode layer as
a sequence of decode-error marks (a lost slice → libav conceals → the gate is
told the tile is broken) with occasional clean frames when a good IDR gets
through. We drive the REAL `FrameQualityGate` and the REAL
`Session._drain_pending_fir` orchestration with those sequences — no video
data needed — and assert how recovery behaves.

Two layers:
  * gate-level  — does the gate keep asking for recovery under sustained loss,
                  and does it self-heal the moment the loss clears?
  * session-level — does `_drain_pending_fir` actually EMIT the recovery FIR in
                  the states loss produces? This is where the "artifact that
                  won't fix itself on a lossy, mostly-static screen" lives.
"""
from __future__ import annotations

import time
import types

from isharescreen.proxy.media.quality_gate import FrameQualityGate
from isharescreen.proxy.session import Session


# ── clock we advance by hand (drop-in for the gate's `time`) ──────────────
class _FakeClock:
    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _gate(num_tiles: int = 1) -> tuple[FrameQualityGate, _FakeClock]:
    g = FrameQualityGate(num_tiles)
    clk = _FakeClock()
    g._time = clk
    return g, clk


# ── gate-level: behaviour under sustained vs clearing loss ────────────────

def test_sustained_loss_keeps_asking_for_recovery():
    """Continuous loss (every frame concealed, no clean IDR gets through):
    the gate must keep issuing tile-0 FIRs — backing off to the slow cadence
    past the cap, but NEVER going permanently silent. Silence here would be
    the 'stuck forever' bug."""
    g, clk = _gate()
    fired = 0
    # 60 s of a fully shredded stream: a decode error every 20 ms, and we try
    # to drain a FIR each tick.
    for _ in range(3000):
        g.mark_decode_error(0)          # loss → tile stays broken
        if g.consume_fir_request():
            fired += 1
        clk.advance(0.020)
    assert 0 in g._keyframe_required     # never spuriously cleared
    # Past the cap it retries at the slow cadence, so over 60 s we still get a
    # steady trickle of FIRs — the stream can recover the instant loss clears.
    assert fired > 5, f"gate went silent under sustained loss (only {fired} FIRs)"
    assert g._fir_attempts[0] >= g._RE_ARM_CAP  # we did blow past the cap


def test_loss_clears_then_recovers_and_resets_cadence():
    """After a loss burst that blew past the cap, a single good IDR + clean
    frame must clear recovery and reset the FIR cadence back to fast."""
    g, clk = _gate()
    # Each real FIR is gated by _RE_ARM_INTERVAL_S (1s), so advance a full
    # interval per attempt to actually blow past the cap (8 attempts).
    for _ in range(g._RE_ARM_CAP + 3):
        g.mark_decode_error(0)
        g.consume_fir_request()
        clk.advance(g._RE_ARM_INTERVAL_S + 0.01)
    assert g._fir_attempts[0] >= g._RE_ARM_CAP
    # loss clears: IDR lands, then a clean decode
    clk.advance(g._POST_IDR_GRACE_S + 0.05)
    g.mark_idr_observed(0)
    clk.advance(0.01)
    g.mark_clean(0)
    assert 0 not in g._keyframe_required          # recovered
    assert g._fir_attempts[0] == 0                # cadence reset to fast


def test_recovery_idr_also_lost_stays_pending():
    """If the recovery IDR is ALSO dropped (loss continues), the tile must
    stay pending — not be declared recovered — until a clean frame finally
    arrives."""
    g, clk = _gate()
    g.mark_decode_error(0)
    for _ in range(20):                  # every attempted IDR is lost → error
        clk.advance(0.25)
        g.consume_fir_request()
        g.mark_decode_error(0)
    assert 0 in g._keyframe_required      # still broken, still asking
    clk.advance(g._POST_IDR_GRACE_S + 0.05)
    g.mark_idr_observed(0)
    clk.advance(0.01)
    g.mark_clean(0)
    assert 0 not in g._keyframe_required  # only NOW recovered


# ── session-level: does _drain_pending_fir actually SEND the FIR? ─────────

def _session_with_gate(g: FrameQualityGate, *, num_tiles: int = 1) -> Session:
    """A bare Session (no __init__) wired to a real gate, with the drive
    surface `_drain_pending_fir` reads. `sent_tiles` records emitted FIRs."""
    s = Session.__new__(Session)
    s._decoder = types.SimpleNamespace(
        _gate=g,
        consume_fir_request=g.consume_fir_request,
    )
    s._observed_tile_count = num_tiles          # `num_tiles` property reads this
    s._dpb_forceall_pending = False
    s._last_dpb_error_t = 0.0
    s._browser_healthy_t = 0.0                  # desktop: never marked healthy
    s._grayout_window_tiles = set()
    s._grayout_window_t = 0.0
    s.sent_tiles = []
    s._send_fir_for_tile = lambda ti, log_per_tile=True: (
        s.sent_tiles.append(ti) or True)
    return s


def test_active_screen_after_loss_emits_recovery_fir():
    """Baseline: a loss-broken tile WITH video still flowing (an active
    screen) drains a recovery FIR as expected."""
    g, _ = _gate()
    g.mark_decode_error(0)                       # loss corrupted the tile
    s = _session_with_gate(g)
    s._last_video_pkt_t = time.monotonic()       # packets arriving right now
    s._drain_pending_fir()
    assert s.sent_tiles == [0]


def test_armed_forceall_escalates_even_when_stream_goes_silent():
    """Regression for the Windows→Mac lossy-link stuck artifact (from a live
    log): an AVC reference break arms the force-all escalation, then the screen
    goes static so Apple stops sending. The 2.5s escalation — which is the auto
    equivalent of a manual Force-IDR — must STILL fire despite the Apple-idle
    guard, because a Force-IDR makes Apple emit a fresh IDR even while idle.

    Before the fix, the idle guard returned early and this never ran (the log's
    'no logs after' silence); after the fix, the armed one-shot escalates."""
    g, _ = _gate()
    g.mark_decode_error(0)                       # the reference break (smear)
    s = _session_with_gate(g)
    s._dpb_forceall_pending = True               # per-tile FIR armed this
    s._last_dpb_error_t = time.monotonic() - 3.0  # unrecovered for 3s (> 2.5)
    s._last_video_pkt_t = time.monotonic() - 5.0  # stream SILENT (idle guard on)
    forced = []
    s.request_fir = lambda tile=None: forced.append(tile)

    s._drain_pending_fir()

    assert forced == [None], "armed force-all escalation was trapped by idle guard"
    assert s._dpb_forceall_pending is False       # one-shot disarmed


def test_armed_forceall_disarms_on_recovery_even_when_silent():
    """The disarm-on-recovery path must also run while idle: if the picture
    recovered (keyframe_required cleared) the escalation must NOT fire."""
    g, _ = _gate()                               # no break → keyframe_required empty
    s = _session_with_gate(g)
    s._dpb_forceall_pending = True
    s._last_dpb_error_t = time.monotonic() - 3.0
    s._last_video_pkt_t = time.monotonic() - 5.0  # silent
    forced = []
    s.request_fir = lambda tile=None: forced.append(tile)

    s._drain_pending_fir()

    assert forced == []                           # recovered → no force-IDR
    assert s._dpb_forceall_pending is False        # disarmed


def test_idle_guard_still_suppresses_plain_pertile_fir():
    """The Apple-idle guard is intentionally KEPT for the ordinary per-tile
    consume path: on a genuinely static screen a plain FIR would be a wasted
    round-trip (Apple isn't encoding new frames). The stuck-artifact case is
    NOT handled by lifting this guard wholesale — it's handled by the armed
    force-all escalation, which now runs above the guard (see
    `test_armed_forceall_escalates_even_when_stream_goes_silent`). Here, with
    NO escalation armed, an idle stream correctly holds the per-tile FIR."""
    g, _ = _gate()
    g.mark_decode_error(0)                       # loss corrupted the tile
    s = _session_with_gate(g)
    s._dpb_forceall_pending = False              # nothing armed
    s._last_video_pkt_t = time.monotonic() - 2.0  # static screen: 2s of silence
    s._drain_pending_fir()
    assert s.sent_tiles == []                     # plain per-tile FIR held
    assert 0 in g._keyframe_required              # gate still wants it
