"""Event-time diagnostics for rare AVC reference-chain failures."""
from __future__ import annotations

import logging
import queue
import time
import types

from isharescreen.proxy.media.quality_gate import FrameQualityGate
from isharescreen.proxy.session import Session


def test_dpb_diagnostic_captures_decoder_transport_and_geometry(caplog):
    session = Session.__new__(Session)
    session._decoder = types.SimpleNamespace(
        recovery_diagnostics={
            "decoder": "d3d11va",
            "nalus_fed": 4321,
            "keyframes_seen": 7,
            "frames_since_keyframe": 900,
            "frames_since_context_reset": 6800,
            "last_keyframe_age_s": 42.5,
            "restarts": 2,
            "sps_patch": True,
            "await_key": False,
            "reference_reset_pending": True,
            "reference_resets": 3,
        },
    )
    session._video_codec = "avc"
    session._connect_wall_ts = time.time() - 3600
    session._last_video_pkt_t = time.monotonic() - 0.02
    session._lost_pkts = 12
    session._last_dpb_diag_loss_total = 5
    session._video_q = queue.Queue()
    session._video_q.put_nowait(b"packet")
    session._video_q_dropped = 3
    session._ltr_enabled = False
    session._ltr_acks_sent = 0
    session._runtime_canvas_w = 3072
    session._runtime_canvas_h = 1728
    session._negotiation = None
    session._avc_needs_reconfig = False

    with caplog.at_level(logging.WARNING):
        session._log_dpb_diagnostics(
            "number of reference frames (8+9) exceeds max (16)",
            time.monotonic(),
        )

    line = caplog.messages[-1]
    assert "decoder=d3d11va" in line
    assert "frames_since_key=900" in line
    assert "context_frames=6800" in line
    assert "ref_reset_pending=True ref_resets=3" in line
    assert "loss_total=12 loss_since_dpb=7" in line
    assert "video_q=1/16384 video_q_drop=3" in line
    assert "canvas=3072x1728" in line
    assert "number of reference frames (8+9) exceeds max (16)" in line


def test_single_tile_dpb_log_arms_recovery_before_frame_flag(monkeypatch):
    """The context-wide AVC overflow is enough to identify tile 0.

    libav may log it from decode() before get_frame() drains the per-frame error
    flag, so recovery must not depend on bad_streak already being nonzero.
    """
    gate = FrameQualityGate(1)
    session = Session.__new__(Session)
    armed: list[str] = []
    session._decoder = types.SimpleNamespace(
        _gate=gate,
        mark_reference_chain_broken=armed.append,
    )
    session._observed_tile_count = 1
    session._dpb_error_window = __import__("collections").deque()
    session._last_decoder_restart_t = 0.0
    session._last_dpb_error_t = 0.0
    session._dpb_fir_count = 0
    session._dpb_forceall_pending = False
    monkeypatch.setattr(session, "_log_dpb_diagnostics", lambda *_args: None)

    session._on_libav_concealment(
        "number of reference frames (6+11) exceeds max (16), discarding one",
    )

    assert gate._keyframe_required == {0}
    assert gate._states[0].bad_streak == 1
    assert session._dpb_forceall_pending is True
    assert armed == [
        "number of reference frames (6+11) exceeds max (16), discarding one",
    ]


def test_dpb_log_arms_decoder_even_while_fir_fast_path_is_cooling_down(
        monkeypatch):
    """Decoder gating is immediate; only the outbound FIR is rate-limited."""
    gate = FrameQualityGate(1)
    armed: list[str] = []
    session = Session.__new__(Session)
    session._decoder = types.SimpleNamespace(
        _gate=gate,
        mark_reference_chain_broken=armed.append,
    )
    session._observed_tile_count = 1
    session._dpb_error_window = __import__("collections").deque()
    session._last_decoder_restart_t = 0.0
    session._last_dpb_error_t = time.monotonic()
    session._last_dpb_fast_recovery_t = time.monotonic()
    session._dpb_fir_count = 1
    session._dpb_forceall_pending = False
    session._last_publish_t = time.monotonic()
    monkeypatch.setattr(session, "_log_dpb_diagnostics", lambda *_args: None)

    msg = "reference picture missing during reorder"
    session._on_libav_concealment(msg)

    assert armed == [msg]
    # No second fast-path FIR was armed inside the one-second cooldown.
    assert gate._keyframe_required == set()
    assert session._dpb_fir_count == 1
