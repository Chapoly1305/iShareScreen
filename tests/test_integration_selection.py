"""Selection regressions covered while reconciling the personal branch."""
from __future__ import annotations

import types

import av
import av.codec.hwaccel as hwaccel_mod
import pytest

from isharescreen.proxy.media import hwcaps as H, registry as R


@pytest.fixture(autouse=True)
def clean_probes(monkeypatch):
    monkeypatch.delenv("ISS_DECODER", raising=False)
    monkeypatch.delenv("ISS_HEVC444", raising=False)
    monkeypatch.setattr(H, "_libav_cache", {})
    monkeypatch.setattr(H, "_qsv_cache", {})


def test_both_hardware_paths_remain_available_without_changing_priority(monkeypatch):
    monkeypatch.setattr(H.sys, "platform", "win32")
    monkeypatch.setattr(H, "available_hwaccels", lambda _: ("d3d11va",))
    monkeypatch.setattr(H, "_probe_one", lambda _: True)
    calls = []
    monkeypatch.setattr(H, "_qsv_hevc444", lambda: calls.append(1) or True)
    assert H.hevc444_decode_method() == "libav"
    assert H.qsv_hevc444_hwdecode() is True
    assert H.qsv_hevc444_hwdecode() is True
    assert calls == [1]
    assert R.select("hevc").name == "libav-hevc444"
    assert R.select("hevc", override="qsv-hevc444").name == "qsv-hevc444"


@pytest.mark.parametrize("override, codec", [
    ("libav-avc420", "avc"), ("qsv", "hevc"),
    ("libav-hevc444-sw", "hevc"),
])
def test_auto_codec_follows_supported_decoder(monkeypatch, override, codec):
    monkeypatch.setattr(R.sys, "platform", "win32")
    monkeypatch.setenv("ISS_DECODER", override)
    monkeypatch.setattr(R, "can_decode", lambda *a, **k: codec != "hevc")
    assert R.resolve_codec("auto") == codec


def test_explicit_codec_rejects_incompatible_decoder(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "win32")
    monkeypatch.setenv("ISS_DECODER", "qsv")
    assert R.resolve_codec("avc") == "avc"
    assert R.select("avc", override="qsv-hevc444").name == "libav-avc420"


def test_unsupported_decoder_does_not_change_auto_codec(monkeypatch):
    monkeypatch.setattr(R.sys, "platform", "win32")
    monkeypatch.setenv("ISS_DECODER", "vt")
    monkeypatch.setattr(R, "can_decode", lambda *a, **k: False)
    assert R.resolve_codec("auto") == "avc"


def test_disabled_hevc_does_not_probe_either_path(monkeypatch):
    monkeypatch.setenv("ISS_HEVC444", "0")
    monkeypatch.setattr(H, "_probe_one", lambda _: pytest.fail("probe disabled"))
    monkeypatch.setattr(H, "_qsv_hevc444", lambda: pytest.fail("probe disabled"))
    assert H.hevc444_decode_method() is None


@pytest.mark.parametrize("bound, emits, expected", [
    (False, True, False), (True, True, True),
    (True, False, False), (None, True, True),
])
def test_probe_requires_frame_and_rejects_explicit_software(
    monkeypatch, bound, emits, expected,
):
    frame = types.SimpleNamespace(format=types.SimpleNamespace(name="yuv444p"))
    ctx = types.SimpleNamespace(
        is_hwaccel=bound,
        decode=lambda packet: [frame] if packet is not None and emits else [],
    )
    monkeypatch.setattr(hwaccel_mod, "HWAccel", lambda **kwargs: object())
    monkeypatch.setattr(av, "CodecContext", types.SimpleNamespace(create=lambda *a, **k: ctx))
    assert H._probe_one("cuda") is expected


def test_probe_sample_really_decodes_as_160px_444():
    ctx = av.CodecContext.create("hevc", "r")
    frames = list(ctx.decode(av.Packet(H._HEVC444_SAMPLE))) + list(ctx.decode(None))
    assert len(frames) == 1
    assert (frames[0].width, frames[0].height, frames[0].format.name) == (160, 160, "yuv444p")
