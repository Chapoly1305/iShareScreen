"""Read/write the local OS clipboard. Thin wrapper around pyperclip.

pyperclip handles the per-platform ugliness for us: ctypes Win32 on
Windows (no PowerShell startup), pbcopy/pbpaste on macOS, GPaste D-Bus
on GNOME-Wayland, and xclip/xsel/wl-clipboard fallback elsewhere.

`available()` is meant to be called once at startup so we can disable
clipboard sync cleanly + log an actionable hint instead of warning on
every poll cycle. On Linux without xclip/wl-clipboard installed,
pyperclip raises `PyperclipException`; we treat that as "no clipboard"
and silently no-op the rest of the session.
"""
from __future__ import annotations

import logging
import sys
from typing import Optional

import pyperclip


log = logging.getLogger(__name__)


def available() -> bool:
    """Probe the platform clipboard. True if reads/writes will work.

    Logs a one-line install hint on Linux when no helper is on PATH so
    the user knows what to install."""
    try:
        pyperclip.paste()
        return True
    except pyperclip.PyperclipException as e:
        if sys.platform.startswith("linux"):
            log.warning(
                "clipboard sync disabled — install a helper: "
                "`apt install xclip` (X11) or `apt install wl-clipboard` "
                "(Wayland). Underlying error: %s", e,
            )
        else:
            log.warning("clipboard sync disabled: %s", e)
        return False


def read_text() -> Optional[str]:
    """Return the current local clipboard text, or None on failure."""
    try:
        return pyperclip.paste()
    except pyperclip.PyperclipException:
        return None


def push_text(text: str) -> bool:
    """Write `text` to the local OS clipboard. Returns True on success."""
    try:
        pyperclip.copy(text)
        return True
    except pyperclip.PyperclipException as e:
        log.debug("local clipboard push failed: %s", e)
        return False


__all__ = ["available", "push_text", "read_text"]
