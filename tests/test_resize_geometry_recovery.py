"""Geometry recovery regressions for AVC dynamic resize and layout gaps."""
from __future__ import annotations

import threading
import types

from isharescreen.proxy.protocol.rfb import DisplayRect
from isharescreen.proxy.session import Session
import isharescreen.proxy.session as session_mod


def _bare_geometry_session(
    canvas: tuple[int, int], rects: list[DisplayRect],
) -> Session:
    session = Session.__new__(Session)
    session._runtime_canvas_w, session._runtime_canvas_h = canvas
    session._negotiation = None
    session._display_rects = rects
    return session


def test_full_display_layout_needs_no_corrective_crop() -> None:
    session = _bare_geometry_session(
        (1920, 1080), [DisplayRect(81, 0, 0, 1920, 1080)])
    assert session.display_content_rect is None


def test_short_display_layout_crops_uncovered_bottom() -> None:
    session = _bare_geometry_session(
        (1920, 2160), [DisplayRect(81, 0, 0, 1920, 1080)])
    assert session.display_content_rect == DisplayRect(0, 0, 0, 1920, 1080)


def test_avc_resize_drops_slices_until_new_avcc_arrives(monkeypatch) -> None:
    """A 0x451 layout can precede the new avcC. New-size slices must not be
    forwarded or decoded under the old SPS; that painted old-height content
    at the top of the new canvas with black/green rows below."""
    session = Session.__new__(Session)
    key = (0x1234, 99)
    session._decoder = types.SimpleNamespace(
        feed_nalu=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transitional slice reached decoder")))
    session._pending_groups = {key: [(1, True, b"not-an-avcc")]}
    session._group_arrival = {key: 0.0}
    session._recently_flushed = {}
    session._ssrc_to_tile = {0x1234: 0}
    session._tile_bytes = {}
    session._ltr_enabled = False
    session._video_codec = "avc"
    session._avc_needs_reconfig = True
    session._video_au_callback = lambda *_args: (_ for _ in ()).throw(
        AssertionError("transitional slice reached browser"))
    session._last_publish_t = 0.0
    session._fresh_evt = threading.Event()

    monkeypatch.setattr(
        session, "_maybe_reharvest_avc_config", lambda _payloads: None)
    monkeypatch.setattr(
        session_mod, "reassemble_h264",
        lambda _payloads: (_ for _ in ()).throw(
            AssertionError("transitional slice was reassembled")))

    session._flush_group(key)

    assert session._avc_needs_reconfig is True
    assert key not in session._pending_groups
