from enum import IntFlag
from types import SimpleNamespace
import sys

import pytest
from isharescreen.proxy.protocol.auth import AuthError
from isharescreen.proxy.protocol.kerberos_spnego import SpnegoGSS


class Flags(IntFlag):
    mutual_auth = 2
    sequence_detect = 8
    confidentiality = 16
    integrity = 32


@pytest.fixture
def backend(monkeypatch):
    calls = []
    ctx = SimpleNamespace(complete=True, negotiated_protocol="kerberos", context_attr=Flags(58),
                          step=lambda token: b"request" if token is None else None,
                          unwrap=lambda token: SimpleNamespace(data=b"x"*16, encrypted=True))
    def client(**kwargs):
        calls.append(kwargs)
        return ctx
    class NoCredentialError(Exception): pass
    api = SimpleNamespace(ContextReq=Flags, NegotiateOptions=SimpleNamespace(use_sspi=1),
                          client=client, exceptions=SimpleNamespace(NoCredentialError=NoCredentialError))
    monkeypatch.setitem(sys.modules, "spnego", api)
    return SpnegoGSS(), ctx, calls


def test_explicit_password_uses_kerberos_only_and_preserves_lkdc_target(backend):
    gss, ctx, calls = backend
    with gss:
        assert gss.start("mac", "alice", "secret", "LKDC:SHA1.TEST", "", "") == b"request"
        assert calls[0]["protocol"] == "kerberos"
        assert calls[0]["hostname"] == "LKDC:SHA1.TEST@LKDC:SHA1.TEST"
        assert calls[0]["username"] == "alice@LKDC:SHA1.TEST"
        assert calls[0]["password"] == "secret"
        assert gss.step(b"reply") == b""
        assert gss.unwrap(b"wrapped") == (b"x"*16, True)
    assert gss.context is None


@pytest.mark.parametrize("change", [dict(complete=False), dict(negotiated_protocol="ntlm"), dict(context_attr=Flags.integrity)])
def test_rejects_incomplete_or_weakened_context(backend, change):
    gss, ctx, calls = backend
    gss.start("mac", "alice", "", "REALM", "", "")
    for key, value in change.items(): setattr(ctx, key, value)
    with pytest.raises(AuthError): gss.step(b"reply")


def test_native_error_does_not_expose_password(backend):
    gss, ctx, calls = backend
    def fail(token): raise RuntimeError("secret")
    ctx.step = fail
    with pytest.raises(AuthError) as error:
        gss.start("mac", "alice", "secret", "REALM", "", "")
    assert "secret" not in str(error.value)
