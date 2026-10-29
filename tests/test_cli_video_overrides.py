"""Video CLI precedence, without a GPU or remote connection."""
import os
import sys
from types import SimpleNamespace

import pytest

from isharescreen import cli


@pytest.mark.parametrize("option,env_name,old,new", [
    ("--codec", "ISS_VIDEO_CODEC", "avc", "hevc"),
    ("--decoder", "ISS_DECODER", "libav-hevc444", "libav-avc420"),
])
def test_cli_video_precedence(monkeypatch, option, env_name, old, new):
    monkeypatch.setenv(env_name, old)
    # An omitted flag inherits; explicit auto resets; a concrete flag wins.
    cli._apply_video_overrides(cli._make_parser().parse_args([]))
    assert os.environ[env_name] == old
    cli._apply_video_overrides(cli._make_parser().parse_args([option, "auto"]))
    assert env_name not in os.environ
    cli._apply_video_overrides(cli._make_parser().parse_args([option, new]))
    assert os.environ[env_name] == new


def test_unknown_decoder_rejected_before_mutating_settings(monkeypatch):
    monkeypatch.setenv("ISS_VIDEO_CODEC", "hevc")
    monkeypatch.setenv("ISS_DECODER", "libav-hevc444")
    args = cli._make_parser().parse_args(["--codec", "avc", "--decoder", "d3d11va"])
    with pytest.raises(ValueError, match="libav-avc420"):
        cli._apply_video_overrides(args)
    assert os.environ["ISS_VIDEO_CODEC"] == "hevc"
    assert os.environ["ISS_DECODER"] == "libav-hevc444"


@pytest.mark.parametrize("name", ["vt", "videotoolbox", "qsv", "libav", "libav-avc420"])
def test_existing_decoder_names_and_aliases_remain_accepted(monkeypatch, name):
    monkeypatch.delenv("ISS_DECODER", raising=False)
    cli._apply_video_overrides(cli._make_parser().parse_args(["--decoder", name]))
    assert os.environ["ISS_DECODER"] == name


@pytest.mark.parametrize("flags,expected", [([], "hevc"), (["--codec", "auto"], "avc")])
def test_browser_preserves_inherited_codec_but_honors_explicit_auto(monkeypatch, flags, expected):
    monkeypatch.setenv("ISS_VIDEO_CODEC", "hevc")
    monkeypatch.delenv("ISS_TILES_PER_FRAME", raising=False)
    args = cli._make_parser().parse_args(["--frontend", "browser", *flags])
    cli._apply_video_overrides(args)
    seen = []
    monkeypatch.setitem(sys.modules, "isharescreen.frontend.wt.server", SimpleNamespace(
        run=lambda *a, **kw: seen.append(os.environ["ISS_VIDEO_CODEC"]) or 0,
    ))
    assert cli._run_frontend(None, args) == 0
    assert seen == [expected]
