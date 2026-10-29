"""Optional connect GUI for iShareScreen.

A browser-based connect form: a tiny local HTTP server (Python stdlib) serves an
HTML form for the connection details and opens it in the default browser, then
launches the session. The browser gives real native text selection / copy-paste
and a modern look, with **zero extra dependencies** — nothing to add to
``pyproject.toml`` and nothing to ``brew``/``apt`` install on any platform.

Run it with ``python -m isharescreen.gui.connect`` (or wire it to ``iss --gui``).
"""
