"""Fresh-install logging setup."""
from __future__ import annotations

import logging
import types

from isharescreen.cli import _setup_logging


def test_setup_logging_creates_missing_parent_directory(tmp_path):
    root = logging.getLogger()
    old_handlers = list(root.handlers)
    old_level = root.level
    log_path = tmp_path / "fresh-home" / ".iss" / "gui-1-1.log"
    args = types.SimpleNamespace(
        verbose=False,
        quiet=False,
        log_file=str(log_path),
    )

    try:
        _setup_logging(args)
        assert log_path.is_file()
        assert any(
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename == str(log_path)
            for handler in root.handlers
        )
    finally:
        new_handlers = list(root.handlers)
        root.handlers[:] = old_handlers
        root.setLevel(old_level)
        for handler in new_handlers:
            if handler not in old_handlers:
                handler.close()
