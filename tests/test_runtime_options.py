"""Runtime CLI compatibility, validation, and import ordering without a host."""
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from isharescreen.cli import _make_parser
from isharescreen.runtime_options import (
    OPTIONS, DEVELOPER_OPTIONS, LAUNCHER_OPTIONS, EXISTING_OPTIONS, apply_arguments,
)


def test_catalog_covers_every_runtime_iss_setting():
    root = Path(__file__).resolve().parents[1] / "src" / "isharescreen"
    settings = set()
    for path in root.rglob("*.py"):
        if path.name != "runtime_options.py":
            settings.update(re.findall(r'\bISS_[A-Z0-9_]+\b', path.read_text()))
    assert settings == {option.env for option in OPTIONS + DEVELOPER_OPTIONS + LAUNCHER_OPTIONS} | EXISTING_OPTIONS.keys()


@pytest.mark.parametrize("option", [o for o in OPTIONS if o.kind is bool], ids=lambda o: o.flag)
def test_boolean_flags_override_both_environment_values(monkeypatch, option):
    monkeypatch.setenv(option.env, "0")
    apply_arguments(_make_parser().parse_args(["--" + option.flag]))
    assert os.environ[option.env] == "1"
    apply_arguments(_make_parser().parse_args(["--no-" + option.flag]))
    assert os.environ[option.env] == "0"


def test_omitted_flags_preserve_environment(monkeypatch):
    monkeypatch.setenv("ISS_RCTL", "0")
    monkeypatch.setenv("ISS_RCTL_BWE", "23000")
    before = dict(os.environ)
    apply_arguments(_make_parser().parse_args([]))
    assert dict(os.environ) == before


@pytest.mark.parametrize("flags", [
    ["--rctl-bwe", "0"], ["--rctl-bwe", "65536"],
    ["--rctl-bwe", "20.5"], ["--dev-option", "drop-pct=101"],
    ["--dev-option", "drop-pct=nan"], ["--present-fps", "inf"],
    ["--sharpen", "-1"], ["--dev-option", "tiles-per-frame=0"],
    ["--dev-option", "audio-f4=0x100000000"], ["--dev-option", "media-feedback=unknown"],
])
def test_invalid_options_rejected_before_applying_environment(monkeypatch, flags):
    monkeypatch.setenv("ISS_RCTL", "1")
    with pytest.raises(SystemExit):
        _make_parser().parse_args(["--no-rctl", *flags])
    assert os.environ["ISS_RCTL"] == "1"


def test_numeric_flags_and_bandwidth_reset(monkeypatch):
    monkeypatch.setenv("ISS_VIDEO_BITRATE_BPS", "9000000")
    for option in OPTIONS + DEVELOPER_OPTIONS + LAUNCHER_OPTIONS:
        if option.env != "ISS_VIDEO_BITRATE_BPS":
            monkeypatch.delenv(option.env, raising=False)
    args = _make_parser().parse_args([
        "--rctl-bwe", "30000", "--video-bitrate-bps", "0",
        "--dev-option", "audio-f4=0x1234", "--present-fps", "240",
    ])
    apply_arguments(args)
    assert os.environ["ISS_RCTL_BWE"] == "30000"
    assert os.environ["ISS_AUDIO_F4"] == "4660"
    assert float(os.environ["ISS_PRESENT_FPS"]) == 240
    assert "ISS_VIDEO_BITRATE_BPS" not in os.environ


def test_cli_defers_native_module_imports_until_flags_are_applied():
    code = '''
import sys
from isharescreen import cli
assert "isharescreen.proxy.session" not in sys.modules
assert "isharescreen.proxy.media.hevc" not in sys.modules
args = cli._make_parser().parse_args(["--dev-option", "wt-profile=on", "--dev-option", "pertile-recovery=off", "--no-rctl"])
cli.apply_arguments(args)
from isharescreen.proxy import session
from isharescreen.proxy.media import hevc
assert session._PROFILE_RX is True
assert hevc._PERTILE_RECOVERY is False
s = session.Session(session.SessionConfig(host="test", username="test", password="test"))
assert s._rctl_enabled is False
'''
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("option", DEVELOPER_OPTIONS, ids=lambda o: o.flag)
def test_developer_settings_are_not_standalone_parameters(option):
    parser = _make_parser()
    assert "--" + option.flag not in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--" + option.flag])


@pytest.mark.parametrize("option", [o for o in DEVELOPER_OPTIONS if o.kind is bool], ids=lambda o: o.flag)
def test_developer_booleans_are_validated_and_overridable(monkeypatch, option):
    monkeypatch.setenv(option.env, "1")
    args = _make_parser().parse_args(["--dev-option", option.flag + "=off"])
    apply_arguments(args)
    assert os.environ[option.env] == "0"
    args = _make_parser().parse_args(["--dev-option", option.flag + "=on"])
    apply_arguments(args)
    assert os.environ[option.env] == "1"


@pytest.mark.parametrize("value", [
    "legacy-chroma=on", "unknown=on", "rctl=on", "drop-pct", "drop-pct=",
    "wt-profile=maybe", "drop-pct=infinity", "tiles-per-frame=1.5",
])
def test_developer_option_rejects_typos_and_invalid_values(value):
    with pytest.raises(SystemExit):
        _make_parser().parse_args(["--dev-option", value])


def test_repeated_developer_option_last_value_wins(monkeypatch):
    monkeypatch.delenv("ISS_DROP_PCT", raising=False)
    apply_arguments(_make_parser().parse_args([
        "--dev-option", "drop-pct=5", "--dev-option", "drop-pct=2",
    ]))
    assert float(os.environ["ISS_DROP_PCT"]) == 2


def test_launcher_callback_still_parses_without_cluttering_help(monkeypatch):
    monkeypatch.delenv("ISS_SESSION_CHOICE_URL", raising=False)
    parser = _make_parser()
    assert "--session-choice-url" not in parser.format_help()
    apply_arguments(parser.parse_args(["--session-choice-url", "http://127.0.0.1:1234/session-choice"]))
    assert os.environ["ISS_SESSION_CHOICE_URL"] == "http://127.0.0.1:1234/session-choice"


def test_dead_legacy_chroma_flag_is_rejected():
    with pytest.raises(SystemExit):
        _make_parser().parse_args(["--legacy-chroma"])
