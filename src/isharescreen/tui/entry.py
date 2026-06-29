"""`iss` console-script entry. Routes between the browser connect GUI
(default), the script-friendly CLI (`--host …`, `--headless`, `--help`,
`--version`), and the legacy terminal TUI (`--tui`).

Bare `iss` opens the browser connect form + live diagnostics dashboard. A
full connection on the command line (`iss --host mac.local -u me
--password-stdin`) runs the session directly — that's also how the GUI form
launches a session under the hood. `iss --tui [...]` launches the old TUI
with any typed values pre-filled into its connect form.
"""
from __future__ import annotations

import sys
from typing import Any, Optional


# Flags that route to `cli.py` instead of the TUI. Anything else goes
# to the TUI (with the typed values pre-filled into the connect form).
_CLI_ROUTING_FLAGS = {"--headless", "--help", "-h", "--version", "--list-decoders"}


def main() -> int:
    argv = sys.argv[1:]
    # The terminal TUI is opt-in now — the default UI is the browser connect
    # form + live diagnostics dashboard. `iss --tui [...]` still launches the
    # TUI, with any typed values pre-filled into its connect form.
    if "--tui" in argv:
        from isharescreen.tui.app import main as tui_main
        return tui_main(_build_cli_overrides([a for a in argv if a != "--tui"]))
    # A connection / session on the command line (`--host …`), or the
    # script/help flags, run the cli directly with no GUI — this is also how
    # the GUI form launches the actual session under the hood.
    if "--host" in argv or any(a in _CLI_ROUTING_FLAGS for a in argv):
        from isharescreen.cli import main as cli_main
        return cli_main()
    # Default: the browser connect form + live diagnostics dashboard.
    from isharescreen.gui.connect import main as gui_main
    return gui_main()


def _build_cli_overrides(argv: list[str]) -> Optional[dict[str, Any]]:
    """Parse `argv` with cli.py's argparse and return the just-typed
    fields as an override dict for the TUI's connect form. Returns None
    when argv is empty (TUI falls back to last-session / defaults only).

    Only fields the user actually mentioned end up in the dict —
    argparse defaults would otherwise overwrite saved preferences. For
    BooleanOptionalAction flags (`--audio`/`--no-audio`,
    `--curtain`/`--no-curtain`), either form counts as "user set"."""
    if not argv:
        return None
    from isharescreen.cli import _make_parser, _parse_advertise, _password_from_args

    args = _make_parser().parse_args(argv)
    typed = set(argv)
    overrides: dict[str, Any] = {}

    if args.host:
        overrides["host"] = args.host
    if args.user:
        overrides["user"] = args.user
    # --password-stdin lets a launcher pipe in a password while still
    # showing the TUI. Without it, the password field stays whatever the
    # last-session / default flow produced (i.e., empty).
    pwd = _password_from_args(args)
    if pwd:
        overrides["password"] = pwd
    if args.advertise:
        d = _parse_advertise(args.advertise)
        if d is not None:
            overrides["advertise"] = (
                f"{d.width}x{d.height}"
                + (f"@{d.hidpi_scale}" if d.hidpi_scale != 1 else "")
            )
    if "--audio" in typed or "--no-audio" in typed:
        overrides["audio"] = args.audio
    if "--curtain" in typed or "--no-curtain" in typed:
        overrides["curtain"] = args.curtain
    if "--hdr" in typed:
        overrides["hdr"] = args.hdr
    if "--share-console" in typed:
        overrides["share_console"] = args.share_console
    if "--alt-session" in typed:
        overrides["alt_session"] = args.alt_session
    # --verbose / --log-file are not connect-form fields; they live on
    # ViewerArgs so the supervisor can forward them to the worker.
    if args.verbose:
        overrides["verbose"] = args.verbose
    if args.log_file:
        overrides["log_file"] = args.log_file
    if "--codec" in typed:
        overrides["codec"] = args.codec
    return overrides or None


if __name__ == "__main__":
    raise SystemExit(main())
