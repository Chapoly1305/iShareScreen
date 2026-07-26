"""CLI help must survive argparse's percent-style interpolation."""
from __future__ import annotations

from isharescreen.cli import _make_parser


def test_cli_help_formats_without_interpolation_errors():
    help_text = _make_parser().format_help()

    assert "scale' (%)." in help_text
