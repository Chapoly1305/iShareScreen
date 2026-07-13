"""`iss` console-script entry + connect-form persistence.

The default `iss` invocation opens the browser connect GUI
(`isharescreen.gui.connect`); `iss --host …` / `iss --headless` run the
session directly via the CLI. This package holds only the pieces that
survived the removal of the old Textual terminal UI:

  entry    - the console-script entry point + CLI/GUI router.
  storage  - `~/.iss/last.json` persistence for the connect form.
"""
