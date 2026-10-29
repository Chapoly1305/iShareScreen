from io import StringIO
from types import SimpleNamespace

from isharescreen import cli
from isharescreen.gui import connect as C
from isharescreen.tui import storage


def test_gui_passes_auth_options_and_keeps_secrets_out_of_argv_and_profile(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_state_dir", lambda: tmp_path)
    monkeypatch.setattr(C, "_kill_current", lambda: None)
    monkeypatch.setattr(C, "_free_bridge_port", lambda port: None)
    monkeypatch.setattr(C, "_emit", lambda *args: None)
    monkeypatch.setattr(C, "_STATE", {})
    monkeypatch.setattr(C, "_LAST", {})
    monkeypatch.setattr(C, "_PROC", None)
    monkeypatch.setattr(C, "_LAUNCH_N", 0)
    monkeypatch.setattr(C.threading, "Thread", lambda *a, **kw: SimpleNamespace(start=lambda: None))
    launched = []

    class Input(StringIO):
        def close(self):
            self.contents = self.getvalue()
            super().close()

    stream = Input()
    def popen(cmd, **kwargs):
        launched.append((cmd, kwargs))
        return SimpleNamespace(pid=123, stdin=stream)
    monkeypatch.setattr(C.subprocess, "Popen", popen)
    C._launch(dict(host="mac.local", port="5901", user="alice", password="secret-value",
                   auth="kerberos", krb_realm="EXAMPLE.COM", krb_service="vnc/mac.local@EXAMPLE.COM",
                   remember_password="on", decoder="libav-avc420"))
    cmd, kwargs = launched[0]
    assert "secret-value" not in repr(cmd) + repr(kwargs["env"])
    assert stream.contents == "secret-value\n"
    assert cmd[cmd.index("--auth") + 1] == "kerberos"
    assert cmd[cmd.index("--port") + 1] == "5901"
    assert "--remember-password" in cmd
    assert not any(flag.startswith("--krb-") for flag in cmd)
    assert "secret-value" not in (tmp_path / "last.json").read_text()
    args = cli._make_parser().parse_args(cmd[3:])
    monkeypatch.setattr(cli.sys, "stdin", StringIO(stream.contents))
    cfg = cli._build_session_config(args)
    assert cfg.password == "secret-value" and cfg.auth_mode == "kerberos" and cfg.port == 5901
    assert cfg.krb_realm == cfg.krb_principal == cfg.krb_service == ""
    assert storage.load_last().krb_realm == ""


def test_saved_profile_renders_auth_selection_but_never_password(monkeypatch):
    monkeypatch.setattr(C, "load_last", lambda: SimpleNamespace(
        host='mac"<.local', user="alice", password="never-render", auth="kerberos",
        remember_password=True, port=5901, krb_realm='EXAMPLE"<.COM',
    ))
    monkeypatch.setattr(C, "_decoder_options_html", lambda: "")
    html = C._form_page().decode()
    assert 'value="kerberos" selected' in html
    assert 'value="5901"' in html
    assert 'id="remember-password" checked' in html
    assert "mac&quot;&lt;.local" in html
    assert "EXAMPLE" not in html
    assert "never-render" not in html


def test_first_kerberos_form_does_not_require_manual_realm(monkeypatch):
    monkeypatch.setattr(C, "load_last", lambda: None)
    monkeypatch.setattr(C, "_decoder_options_html", lambda: "")
    html = C._form_page().decode()
    assert all(f'name="{field}"' not in html for field in ("krb_realm", "krb_principal", "krb_service"))
    assert "First connection:" not in html
    assert "authMode" not in html


def test_gui_rejects_incompatible_video_before_stopping_session(monkeypatch):
    monkeypatch.setattr(C, "_kill_current", lambda: (_ for _ in ()).throw(AssertionError("must preserve running session")))
    messages = []
    monkeypatch.setattr(C, "_emit", lambda kind, text: messages.append(text))
    for settings in (
        dict(frontend="desktop", codec="avc", decoder="qsv-hevc444"),
        dict(frontend="browser", codec="hevc"),
        dict(frontend="desktop", decoder="unknown"),
    ):
        C._launch(dict(host="mac.local", user="alice", **settings))
    assert len(messages) == 3


def test_form_has_advanced_controls_and_secret_free_storage_allowlist(monkeypatch):
    monkeypatch.setattr(C, "load_last", lambda: None)
    monkeypatch.setattr(C, "_decoder_options_html", lambda: "")
    html = C._form_page().decode()
    assert '<details id="advanced">' in html
    assert 'name="codec"' in html and 'name="decoder"' in html
    fields = html.split("const fields = ", 1)[1].split(";", 1)[0]
    assert "'password'" not in fields
    assert "'codec'" in fields and "'decoder'" in fields
    assert "localStorage.setItem" in html


def test_new_login_defaults_to_kerberos_without_remembering_password(monkeypatch):
    monkeypatch.setattr(C, "load_last", lambda: None)
    monkeypatch.setattr(C, "_decoder_options_html", lambda: "")
    html = C._form_page().decode()
    assert 'value="kerberos" selected' in html
    assert 'id="remember-password" checked' not in html
