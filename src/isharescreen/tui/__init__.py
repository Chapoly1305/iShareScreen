"""Terminal-side UI for iShareScreen.

The default `iss` entry point lands here: a Textual app that owns the
terminal, runs the connect form, spawns the actual desktop viewer as a
child process, and renders live stats it subscribes to over the child's
control socket. `iss --headless` bypasses the TUI entirely.

Sub-modules:

  control_client  - async subscriber to a Session's ControlServer.
  supervisor      - spawns + manages the child iss process.
  connect_screen  - the pre-flight form (replaces the old stdin prompt).
  session_screen  - the live-session view (stats, settings, log).
  app             - the Textual App that wires the screens together.
  entry           - the console-script entry point + headless router.
"""
