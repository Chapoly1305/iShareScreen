"""`iss` console-script entry. Routes between the browser connect GUI
(default) and the script-friendly CLI (`--host …`, `--headless`, `--help`,
`--version`).

Bare `iss` opens the browser connect form + live diagnostics dashboard. A
full connection on the command line (`iss --host mac.local -u me
--password-stdin`) runs the session directly — that's also how the GUI form
launches a session under the hood.
"""
from __future__ import annotations

import sys


# Flags that route to `cli.py` instead of the browser connect GUI.
_CLI_ROUTING_FLAGS = {"--headless", "--help", "-h", "--version", "--list-decoders"}


def main() -> int:
    argv = sys.argv[1:]
    if "--tui" in argv:
        print("iss: the terminal TUI has been removed; opening the browser "
              "connect UI instead.", file=sys.stderr)
    # A connection / session on the command line (`--host …`), or the
    # script/help flags, run the cli directly with no GUI — this is also how
    # the GUI form launches the actual session under the hood.
    if "--host" in argv or any(a in _CLI_ROUTING_FLAGS for a in argv):
        from isharescreen.cli import main as cli_main
        return cli_main()
    # Default: the browser connect form + live diagnostics dashboard.
    from isharescreen.gui.connect import main as gui_main
    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
