"""Credential isolation, input precedence, and opt-in password persistence."""
from __future__ import annotations

import io
import json

import pytest

from isharescreen import cli, credentials as C
from isharescreen.tui import storage


@pytest.fixture
def store(monkeypatch):
    entries = {}

    class Store:
        def get_password(self, service, account):
            return entries.get((service, account))

        def set_password(self, service, account, value):
            entries[service, account] = value

        def delete_password(self, service, account):
            del entries[service, account]

    monkeypatch.setattr(C, "_backend", Store)
    return entries


def config(monkeypatch, *flags, stdin=""):
    monkeypatch.setattr(cli.sys, "stdin", io.StringIO(stdin))
    args = cli._make_parser().parse_args(["--host", "mac.local", "-u", "alice", *flags])
    return cli._build_session_config(args)


def test_store_scoped_to_host_port_and_account(store):
    C.set_password("Mac.local.", 5900, "alice", "secret")
    assert C.get_password("mac.local", 5900, "alice") == "secret"
    assert C.get_password("mac.local", 5901, "alice") is None
    assert C.get_password("mac.local", 5900, "bob") is None
    assert C.get_password("other.local", 5900, "alice") is None
    C.delete_password("mac.local", 5900, "alice")
    C.delete_password("mac.local", 5900, "alice")  # idempotent
    assert not store


def test_explicit_password_overrides_saved_password(monkeypatch, store):
    C.set_password("mac.local", 5900, "alice", "old")
    cfg = config(monkeypatch, "--password-stdin", stdin="new\n")
    assert cfg.password == "new"
    assert C.get_password("mac.local", 5900, "alice") == "old"
    assert cfg.on_authenticated is None


def test_blank_gui_password_uses_saved_password(monkeypatch, store):
    C.set_password("mac.local", 5900, "alice", "saved")
    assert config(monkeypatch, "--password-stdin", stdin="\n").password == "saved"


def test_save_occurs_only_on_success_callback(monkeypatch, store):
    C.set_password("mac.local", 5900, "alice", "old")
    cfg = config(monkeypatch, "--password-stdin", "--remember-password", stdin="new\n")
    assert C.get_password("mac.local", 5900, "alice") == "old"
    cfg.on_authenticated()
    assert C.get_password("mac.local", 5900, "alice") == "new"
    C.set_password("mac.local", 5900, "alice", "later")
    cfg.on_authenticated()  # reconnect must not overwrite a later update
    assert C.get_password("mac.local", 5900, "alice") == "later"


def test_no_keyring_never_touches_store(monkeypatch):
    def forbidden():
        pytest.fail("credential store was accessed")
    monkeypatch.setattr(C, "_backend", forbidden)
    assert config(monkeypatch, "--no-keyring", "--password-stdin", stdin="pw\n").password == "pw"
    with pytest.raises(SystemExit, match="password"):
        config(monkeypatch, "--no-keyring")


def test_kerberos_does_not_prompt_for_password(monkeypatch):
    import getpass
    monkeypatch.setattr(getpass, "getpass", lambda *a: pytest.fail("password prompt"))
    cfg = config(monkeypatch, "--auth", "kerberos", "--no-keyring", "--krb-realm", "EXAMPLE.COM")
    assert cfg.password == ""
    assert cfg.krb_realm == "EXAMPLE.COM"


def test_store_failure_does_not_log_secrets(monkeypatch, caplog):
    class Broken:
        def set_password(self, *a):
            raise RuntimeError("secret-password backend details")
    monkeypatch.setattr(C, "_backend", Broken)
    C.save_after_auth("mac.local", 5900, "alice", "secret-password")()
    assert "not saved" in caplog.text
    assert "secret-password" not in caplog.text


def test_forget_exits_without_opening_session(monkeypatch, store):
    C.set_password("mac.local", 5900, "alice", "pw")
    monkeypatch.setattr(cli, "_build_session_config", lambda *a: pytest.fail("connection started"))
    assert cli.main(["--host", "mac.local", "-u", "alice", "--forget-password"]) == 0
    assert not store


def test_profile_roundtrip_never_persists_password(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_state_dir", lambda: tmp_path)
    values = storage.ConnectFormValues(
        host="mac.local", user="alice", password="not-in-json", advertise="auto",
        audio=True, curtain=True, hdr=False, share_console=False, alt_session=False,
        remember_password=True, port=5901, auth="kerberos", krb_realm="EXAMPLE.COM",
    )
    storage.save_last(values)
    contents = (tmp_path / "last.json").read_text()
    assert "not-in-json" not in contents
    assert "password" not in json.loads(contents)
    loaded = storage.load_last()
    assert loaded.password == ""
    assert loaded.remember_password and loaded.port == 5901
    assert loaded.auth == "kerberos" and loaded.krb_realm == "EXAMPLE.COM"
    (tmp_path / "last.json").write_text('{"port":"bad"}')
    assert storage.load_last() is None


@pytest.mark.parametrize("succeeds", [False, True])
def test_session_only_saves_after_negotiation_success(monkeypatch, store, succeeds):
    from types import SimpleNamespace
    from isharescreen.proxy import session as S
    from isharescreen.proxy.protocol.auth import AuthError
    cfg = S.SessionConfig("mac.local", "alice", "new",
                          on_authenticated=C.save_after_auth("mac.local", 5900, "alice", "new"))
    session = S.Session.__new__(S.Session)
    session._dest_host, session._recorder = "127.0.0.1", None
    session._media_offer_config = None
    session._negotiation = None
    monkeypatch.setattr(S, "create_offers", lambda **kw: (b"v", b"a"))
    monkeypatch.setattr(S, "extract_offer_ssrc", lambda *a, **kw: None)
    def negotiate(*args, **kwargs):
        assert kwargs["krb_host"] == "mac.local"
        if not succeeds:
            raise AuthError("bad password")
        # Stop immediately after the credential callback at the next access.
        return SimpleNamespace()
    monkeypatch.setattr(S, "connect_and_negotiate", negotiate)
    with pytest.raises((AuthError, AttributeError)) as error:
        session._negotiate_and_burst(cfg)
    if succeeds:
        assert "keys" in str(error.value)
    else:
        assert str(error.value) == "bad password"
    assert C.get_password("mac.local", 5900, "alice") == ("new" if succeeds else None)


@pytest.mark.parametrize("remember", [False, True])
def test_kerberos_password_storage_requires_explicit_opt_in(monkeypatch, store, remember):
    flags = ["--auth", "kerberos", "--password-stdin"]
    if remember:
        flags.append("--remember-password")
    cfg = config(monkeypatch, *flags, stdin="kerberos-secret\n")
    assert not store
    if remember:
        assert cfg.on_authenticated is not None
        cfg.on_authenticated()
        assert C.get_password("mac.local", 5900, "alice") == "kerberos-secret"
    else:
        assert cfg.on_authenticated is None
        assert not store
