"""HiDPI / fixed-advertise resolution regressions.

Covers the dynamic-resolution port fixes:
  * the CLI must NOT blindly 2× a fixed --advertise WxH under `--hidpi auto`
    (that produced a 4×-bytes backing and an oversized window); it seeds a
    safe 1× and defers the local-display decision to the frontend.
  * explicit `--hidpi on`/`off` still force 2×/1×.
  * the frontend's `_resolve_hidpi_request` maps (mode, window, client scale)
    to the right backing scale and advertised logical size — including the
    window opening at the LOGICAL advertised size, not the backing size.
"""
import pytest

from isharescreen.cli import _build_session_config, _make_parser


def _config(*flags):
    base = ["--host", "h", "-u", "u", "--password-stdin"]
    args = _make_parser().parse_args(base + list(flags))
    # _password_from_args reads stdin for --password-stdin; inject directly.
    import isharescreen.cli as cli
    orig = cli._password_from_args
    cli._password_from_args = lambda _a: "pw"
    try:
        return _build_session_config(args)
    finally:
        cli._password_from_args = orig


# ── Fix 2: CLI fixed-advertise hidpi resolution ──────────────────────

def test_fixed_advertise_hidpi_auto_seeds_1x():
    """`--advertise 1366x768 --hidpi auto` must NOT blindly become 2×.

    The CLI can't see the display, so it seeds 1× and lets the frontend
    re-resolve. (The old bug forced 2× → 2732×1536 backing = 4× the bytes.)
    """
    cfg = _config("--advertise", "1366x768", "--hidpi", "auto")
    assert cfg.advertise is not None
    assert (cfg.advertise.width, cfg.advertise.height) == (1366, 768)
    assert cfg.advertise.hidpi_scale == 1
    assert cfg.hidpi == "auto"


def test_fixed_advertise_hidpi_on_forces_2x():
    cfg = _config("--advertise", "1366x768", "--hidpi", "on")
    assert cfg.advertise.hidpi_scale == 2


def test_fixed_advertise_hidpi_off_forces_1x():
    cfg = _config("--advertise", "1366x768", "--hidpi", "off")
    assert cfg.advertise.hidpi_scale == 1


def test_auto_advertise_defers_to_frontend():
    """`--advertise auto` (no fixed geometry) → advertise None, frontend sizes."""
    cfg = _config("--advertise", "auto", "--hidpi", "auto")
    assert cfg.advertise is None


# ── Fix 2/3: frontend resolution + logical window sizing ─────────────

def test_resolve_hidpi_request_auto_matches_local_display():
    app = pytest.importorskip("isharescreen.frontend.desktop.app")
    # Non-Retina client (scale 1) → 1× backing, logical size unchanged.
    assert app._resolve_hidpi_request("auto", 1366, 768, 1) == (1366, 768, 1)
    # Retina client (scale 2) → 2× backing, logical size unchanged.
    assert app._resolve_hidpi_request("auto", 1366, 768, 2) == (1366, 768, 2)


def test_resolve_hidpi_request_explicit_modes():
    app = pytest.importorskip("isharescreen.frontend.desktop.app")
    # on forces 2× even on a non-Retina client; off forces 1× on Retina.
    assert app._resolve_hidpi_request("on", 1366, 768, 1)[2] == 2
    assert app._resolve_hidpi_request("off", 1366, 768, 2)[2] == 1


def test_resolve_hidpi_auto_stays_retina_and_caps_logical():
    app = pytest.importorskip("isharescreen.frontend.desktop.app")
    # A Retina client at a large logical size: auto STAYS 2× (elements stay
    # normal-sized, like Screen Sharing.app) and caps the logical request so
    # the 2× backing still fits the host's 3840×2160 ceiling — rather than
    # dropping to a 1× full-pixel desktop where the host UI renders microscopic.
    req_w, req_h, scale = app._resolve_hidpi_request("auto", 2560, 1440, 2)
    assert scale == 2                        # Retina kept, not downgraded to 1×
    assert req_w * scale <= 3840             # 2× backing fits the host cap
    assert req_h * scale <= 2160
