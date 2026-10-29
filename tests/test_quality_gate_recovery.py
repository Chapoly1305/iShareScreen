"""Unit tests for FrameQualityGate's recovery-clear semantics.

Pure-logic tests over a real `FrameQualityGate` with a controllable clock
(we replace `gate._time` with a fake exposing `monotonic()`), so the
`_RECOVERY_QUIET_S` window is exercised deterministically without sleeps.

Focus: the Windows over-FIR regression. On software (libav) decode the
"Could not find ref" *log* path frequently never reaches the gate
("no libav concealment captured"), so the only marks during a recovery
cycle are the *reliable* per-frame `decode_error_flags` drain noise. Those
must NOT hold the recovery quiet-window open — an error-free frame after an
IDR is genuine recovery and must clear the keyframe-required flag at once.
The quiet-window guard exists only for the *unreliable* libav-log signal
that fires during a silently-grey (no per-frame error) decoder wedge.
"""
from __future__ import annotations

from isharescreen.proxy.media.quality_gate import FrameQualityGate


class _FakeClock:
    """Monotonic clock we advance by hand. Drop-in for the `time` module
    as far as the gate uses it (`monotonic()` only)."""

    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _gate(num_tiles: int = 4) -> tuple[FrameQualityGate, _FakeClock]:
    g = FrameQualityGate(num_tiles)
    clk = _FakeClock()
    g._time = clk
    return g, clk


def _drive_recovery(g: FrameQualityGate, clk: _FakeClock, tile: int = 0) -> None:
    """Break a tile, then deliver a fresh IDR + a clean post-IDR frame,
    advancing past the post-IDR grace so the clean mark isn't suppressed."""
    g.mark_decode_error(tile)                 # reliable per-frame error
    assert tile in g._keyframe_required
    clk.advance(g._POST_IDR_GRACE_S + 0.05)   # past the post-IDR grace
    g.mark_idr_observed(tile)                  # IDR landed → condition 1
    clk.advance(0.01)
    g.mark_clean(tile)                         # error-free decode → condition 2


def test_reliable_only_recovery_clears_immediately():
    """The Windows case: only reliable (per-frame) marks were ever seen, so
    a clean error-free frame after the IDR clears recovery WITHOUT waiting
    out `_RECOVERY_QUIET_S` — no 7-second over-FIR loop."""
    g, clk = _gate()
    _drive_recovery(g, clk, tile=0)
    assert 0 not in g._keyframe_required        # cleared at once
    assert 0 not in g._idr_observed
    assert 0 not in g.bad_tiles


def test_reliable_post_idr_drain_does_not_block_clear():
    """Pre-IDR P-frames draining against the reset DPB keep firing reliable
    per-frame errors right up to the clean frame. Those must NOT re-arm the
    quiet window (the exact regression: reliable drain noise blocked the
    clear for ~7 s on Windows)."""
    g, clk = _gate()
    g.mark_decode_error(0)
    clk.advance(g._POST_IDR_GRACE_S + 0.05)
    g.mark_idr_observed(0)
    # A burst of reliable per-frame errors *after* the IDR (drain noise),
    # each landing just before the clean frame.
    for _ in range(5):
        clk.advance(0.02)
        g.mark_decode_error(0)                  # reliable=True (default)
    clk.advance(0.02)
    g.mark_clean(0)
    assert 0 not in g._keyframe_required         # still clears despite drain


def test_unreliable_concealment_holds_quiet_window():
    """Mac/d3d11va silent-grey case: the libav 'Could not find ref' log
    (reliable=False) keeps firing during grey. A clean-looking frame must
    NOT clear until that unreliable signal has been quiet for
    `_RECOVERY_QUIET_S` — preserving the existing silent-wedge guard."""
    g, clk = _gate()
    g.mark_decode_error(0)
    clk.advance(g._POST_IDR_GRACE_S + 0.05)
    g.mark_idr_observed(0)
    # An unreliable (libav-log) concealment lands AFTER the IDR — the screen
    # is still silently grey even though an IDR was fed.
    clk.advance(0.01)
    g.mark_decode_error(0, reliable=False)
    clk.advance(0.01)
    g.mark_clean(0)
    assert 0 in g._keyframe_required             # held: quiet window not elapsed

    # Once the unreliable signal has been quiet for the full window, the
    # next clean frame clears it.
    clk.advance(g._RECOVERY_QUIET_S + 0.01)
    g.mark_clean(0)
    assert 0 not in g._keyframe_required


def test_unreliable_then_reliable_still_holds():
    """A reliable per-frame error arriving after an unreliable one must not
    *shorten* the quiet window: the window is keyed off the unreliable
    timestamp only, so a later reliable mark can't make us clear early, but
    it also can't extend the hold past the unreliable-quiet point."""
    g, clk = _gate()
    g.mark_decode_error(0)
    clk.advance(g._POST_IDR_GRACE_S + 0.05)
    g.mark_idr_observed(0)
    g.mark_decode_error(0, reliable=False)       # silent-grey marker
    clk.advance(g._RECOVERY_QUIET_S + 0.01)      # unreliable signal now quiet
    g.mark_decode_error(0)                        # reliable drain noise after
    clk.advance(0.01)
    g.mark_clean(0)
    assert 0 not in g._keyframe_required          # reliable noise didn't re-hold
