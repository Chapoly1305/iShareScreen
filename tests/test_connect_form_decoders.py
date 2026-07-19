"""The connect-form decoder dropdown must reflect the REAL hardware probe on
the very first page load — not fall back to the platform filter just because
the browser won the race against the startup probe thread.

`_decoder_options_html` waits on `_PROBE_DONE`; these tests drive that seam
directly, with the probes monkeypatched at the registry's module-global seams
(same technique as test_decoder_registry.py).
"""
from __future__ import annotations

import threading

from isharescreen.gui import connect as C
from isharescreen.proxy.media import registry as R


def _darwin_no_vt(monkeypatch):
    """Pretend: macOS where VideoToolbox HEVC-444 is NOT actually available."""
    monkeypatch.setattr(R.sys, "platform", "darwin")
    monkeypatch.setattr(R, "_hevc444_method", lambda: None)
    monkeypatch.setattr(R, "_vt_available", lambda: False)


def test_first_render_waits_for_probe(monkeypatch):
    """Probe finishes AFTER the request arrives: the render must block until
    the result is in and serve the filtered list, not the platform fallback."""
    _darwin_no_vt(monkeypatch)
    monkeypatch.setattr(C, "_PROBE_DONE", threading.Event())
    t = threading.Timer(0.2, C._PROBE_DONE.set)
    t.start()
    try:
        html = C._decoder_options_html()
    finally:
        t.cancel()
    assert C._PROBE_DONE.is_set(), "render returned before the probe finished"
    assert "vt-hevc444" not in html          # probed-unavailable → hidden
    assert 'value="auto"' in html


def test_probed_out_decoder_hidden_when_ready(monkeypatch):
    """With the probe already complete, unavailable decoders never appear."""
    _darwin_no_vt(monkeypatch)
    done = threading.Event()
    done.set()
    monkeypatch.setattr(C, "_PROBE_DONE", done)
    html = C._decoder_options_html()
    assert "vt-hevc444" not in html
    # the software fallbacks remain offerable
    assert "libav-hevc444-sw" in html


def test_probe_timeout_falls_back_to_platform_filter(monkeypatch):
    """If the probe never finishes (pathological), the form must still render
    — platform-filtered — rather than hang. Wait cap shrunk for the test."""
    _darwin_no_vt(monkeypatch)
    stuck = threading.Event()                 # never set
    monkeypatch.setattr(C, "_PROBE_DONE", stuck)
    orig_wait = stuck.wait
    monkeypatch.setattr(stuck, "wait", lambda timeout=None: orig_wait(0.05))
    html = C._decoder_options_html()
    assert 'value="auto"' in html
    assert "vt-hevc444" in html               # platform filter: vt is darwin-plausible
