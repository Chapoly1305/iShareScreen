"""The live evaluation connection profile is enforced beyond the browser UI."""
from io import StringIO
from types import SimpleNamespace

import pytest

from isharescreen.gui import connect


def test_evaluation_profile_overrides_submitted_fields_without_mutating_input(monkeypatch):
    from isharescreen.proxy.media import registry
    monkeypatch.setattr(registry, "select", lambda codec: SimpleNamespace(name="vt-hevc444") if codec == "hevc" else None)
    submitted = dict(sr_evaluation="on", frontend="browser", advertise="auto", scale="4.0",
                     decoder="libav-avc420", audio="on", curtain="off", host="mac.test", user="viewer", password="secret")
    normalized = connect._evaluation_values(submitted)
    assert normalized == dict(submitted, frontend="desktop", advertise="960x540", scale="1.0",
                              decoder="vt-hevc444", audio="off", curtain="on")
    assert submitted["advertise"] == "auto" and submitted["decoder"] == "libav-avc420"
    assert connect._evaluation_values({"advertise": "auto"}) == {"advertise": "auto"}


def test_evaluation_does_not_silently_fall_back_to_avc(monkeypatch):
    from isharescreen.proxy.media import registry
    monkeypatch.setattr(registry, "select", lambda codec: None)
    with pytest.raises(ValueError, match="HEVC decoder"):
        connect._evaluation_values({"sr_evaluation": "on"})


def test_launch_pins_actual_child_flags_and_precision_and_reconnect_values(monkeypatch, tmp_path):
    from isharescreen import cli
    from isharescreen.tui import storage
    monkeypatch.setattr(storage, "_state_dir", lambda: tmp_path)
    from isharescreen.proxy.media import registry
    monkeypatch.setattr(registry, "select", lambda codec: SimpleNamespace(name="vt-hevc444"))
    monkeypatch.setattr(connect, "_kill_current", lambda: None)
    monkeypatch.setattr(connect, "_free_bridge_port", lambda port: None)
    monkeypatch.setattr(connect, "_emit", lambda *args: None)
    monkeypatch.setattr(connect, "_STATE", {})
    monkeypatch.setattr(connect, "_LATEST", {"stale": True})
    monkeypatch.setattr(connect, "_HELLO", {"canvas": {"w": 3840, "h": 2160}})
    monkeypatch.setattr(connect, "_LAST", {})
    monkeypatch.setattr(connect, "_PROC", None)
    monkeypatch.setattr(connect, "_LAUNCH_N", 0)
    monkeypatch.setattr(connect.threading, "Thread", lambda *args, **kwargs: SimpleNamespace(start=lambda: None))
    monkeypatch.setenv("ISS_SR_PRECISION", "amp-fp16")
    monkeypatch.setenv("ISS_SR_GPU_ONLY", "0")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    monkeypatch.setenv("ISS_VIDEO_CODEC", "avc")
    launched = []

    def popen(cmd, **kwargs):
        launched.append((cmd, kwargs))
        return SimpleNamespace(pid=123, stdin=StringIO())

    monkeypatch.setattr(connect.subprocess, "Popen", popen)
    # Disabled controls are omitted by browsers. Server defaults must still pin them.
    connect._launch(dict(host="mac.test", user="viewer", password="secret", sr_evaluation="on"))
    cmd, kwargs = launched[0]
    assert "secret" not in cmd
    assert cmd[cmd.index("--frontend") + 1] == "desktop"
    assert cmd[cmd.index("--codec") + 1] == "hevc"
    assert cmd[cmd.index("--decoder") + 1] == "vt-hevc444"
    assert cmd[cmd.index("--display") + 1] == "combined"
    assert "--no-dynamic" in cmd and "--no-audio" in cmd
    assert "--curtain" in cmd and "--no-curtain" not in cmd
    assert kwargs["env"]["ISS_SR_PRECISION"] == "fp32"
    assert kwargs["env"]["ISS_SR_GPU_ONLY"] == "1"
    assert kwargs["env"]["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"
    assert connect._STATE["evaluation"] is True
    assert not connect._LATEST and not connect._HELLO
    assert connect._LAST["sr_evaluation"] == "on" and connect._LAST["advertise"] == "960x540"
    # Feed the generated arguments through the real CLI configuration builder.
    args = cli._make_parser().parse_args(cmd[3:])
    monkeypatch.setattr(cli, "_password_from_args", lambda args: "secret")
    config = cli._build_session_config(args)
    assert (config.advertise.width, config.advertise.height, config.advertise.hidpi_scale) == (960, 540, 1.0)
    assert not config.dynamic_resolution and not config.audio and config.curtain


def test_form_and_dashboard_expose_locks_and_canvas_mismatch_warning():
    assert 'name="sr_evaluation"' in connect._FORM
    assert "field.disabled=enabled" in connect._FORM
    assert "dynamic resolution off" in connect._FORM
    assert "blanks the host screen" in connect._FORM
    dashboard = connect._dash_page().decode()
    assert 'id="evaluation-profile"' in dashboard
    assert "awaiting canvas verification" in dashboard
    assert "input does not match the profile" in dashboard
