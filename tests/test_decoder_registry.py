"""Unit tests for the pluggable decoder registry (media/registry.py).

These exercise the selection LOGIC without a GPU: platform filtering, the
hardware-then-priority ordering, override aliasing, software-fallback gating,
and codec resolution. The availability probes are monkeypatched at their
module-global seams (`_libav_hw`, `_qsv_hw`, `_vt_available`) — the spec lambdas
resolve those names at call time, so patching the module attribute steers
selection.
"""
from __future__ import annotations

from isharescreen.proxy.media import registry as R


def test_normalize_override_aliases():
    assert R.normalize_override("auto", "hevc") is None
    assert R.normalize_override("", "hevc") is None
    assert R.normalize_override(None, "hevc") is None
    assert R.normalize_override("vt", "hevc") == "vt-hevc444"
    assert R.normalize_override("videotoolbox", "hevc") == "vt-hevc444"
    assert R.normalize_override("qsv", "hevc") == "qsv-hevc444"
    assert R.normalize_override("libav", "hevc") == "libav-hevc444"
    assert R.normalize_override("libav", "avc") == "libav-avc420"
    assert R.normalize_override("qsv-hevc444", "hevc") == "qsv-hevc444"


def test_candidates_platform_filtered_and_sorted(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "win32")
    names = [s.name for s in R.candidates("hevc")]
    assert "vt-hevc444" not in names                 # darwin-only filtered
    # vendor QSV (90) > generic libav HW (60) > software (20)
    assert names == ["qsv-hevc444", "libav-hevc444", "libav-hevc444-sw"]

    monkeypatch.setattr(R.sys, "platform", "darwin")
    names = [s.name for s in R.candidates("hevc")]
    assert names[0] == "vt-hevc444"                  # prio 100 first
    assert "qsv-hevc444" not in names                # win/linux only


def test_windows_prefers_qsv_over_generic_libav(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "win32")
    # Both vendor QSV and generic libav work → vendor wins (lower-latency intent)
    monkeypatch.setattr(R, "_qsv_hw", lambda: True)
    monkeypatch.setattr(R, "_libav_hw", lambda: True)
    assert R.select("hevc").name == "qsv-hevc444"
    assert R.resolve_codec("auto") == "hevc"

    # QSV absent, generic works → libav-hevc444
    monkeypatch.setattr(R, "_qsv_hw", lambda: False)
    assert R.select("hevc").name == "libav-hevc444"

    # No HW path → codec falls back to AVC (software does NOT drive negotiation)
    monkeypatch.setattr(R, "_libav_hw", lambda: False)
    assert R.can_decode("hevc", "444", hardware_only=True) is False
    assert R.resolve_codec("auto") == "avc"
    # …but select("hevc") still yields the software last-resort if forced into HEVC
    assert R.select("hevc").name == "libav-hevc444-sw"


def test_software_444_outranks_avc_but_does_not_force_hevc(monkeypatch):
    # On a GPU with no HW 4:4:4, codec negotiation must still pick AVC even
    # though an always-available software HEVC decoder exists.
    monkeypatch.setattr(R.sys, "platform", "win32")
    monkeypatch.setattr(R, "_qsv_hw", lambda: False)
    monkeypatch.setattr(R, "_libav_hw", lambda: False)
    assert R.resolve_codec("auto") == "avc"          # not slow CPU HEVC
    assert R.can_decode("hevc", "444") is True        # sw exists (non-hw query)
    assert R.can_decode("hevc", "444", hardware_only=True) is False


def test_macos_prefers_vt_then_libav_then_software(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "darwin")
    monkeypatch.setattr(R, "_libav_hw", lambda: True)
    monkeypatch.setattr(R, "_vt_available", lambda: True)
    assert R.select("hevc").name == "vt-hevc444"      # 100 over libav 60

    monkeypatch.setattr(R, "_vt_available", lambda: False)
    assert R.select("hevc").name == "libav-hevc444"

    monkeypatch.setattr(R, "_libav_hw", lambda: False)
    assert R.select("hevc").name == "libav-hevc444-sw"  # last-resort fallback


def test_override_honored_unsupported_or_unknown_falls_to_auto(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "win32")
    monkeypatch.setattr(R, "_qsv_hw", lambda: False)
    monkeypatch.setattr(R, "_libav_hw", lambda: True)
    assert R.select("hevc", override="qsv-hevc444").name == "qsv-hevc444"
    assert R.select("hevc", override="libav-hevc444-sw").name == "libav-hevc444-sw"
    # vt-hevc444 is darwin-only → unsupported on win32 → auto (libav)
    assert R.select("hevc", override="vt-hevc444").name == "libav-hevc444"
    # unknown name → auto
    assert R.select("hevc", override="nope").name == "libav-hevc444"


def test_avc_lowest_priority_but_always_available(monkeypatch):
    for plat in ("darwin", "win32", "linux"):
        monkeypatch.setattr(R.sys, "platform", plat)
        assert R.can_decode("avc", "420") is True
        spec = R.select("avc")
        assert spec.name == "libav-avc420" and spec.priority == 1
        assert R.resolve_codec("avc") == "avc"
