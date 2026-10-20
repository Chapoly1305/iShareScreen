"""Apple type-35 framing and failure behavior, without a KDC or real secrets."""
from __future__ import annotations

import struct

import pytest

from isharescreen.proxy.protocol import kerberos as K, negotiation as N
from isharescreen.proxy.protocol.auth import AuthError


def blob(data):
    return struct.pack(">I", len(data)) + data


class Socket:
    def __init__(self, data=b""):
        self.data = data
        self.sent = []
        self.closed = False

    def recv(self, n):
        # Deliberately fragmented: length fields and tokens can span reads.
        part, self.data = self.data[:min(n, 2)], self.data[min(n, 2):]
        return part

    def sendall(self, data):
        self.sent.append(data)

    def close(self):
        self.closed = True

    def settimeout(self, *a):
        pass

    def setsockopt(self, *a):
        pass


class GSS:
    confidential = True
    key = b"0123456789abcdef"
    closed = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.closed = True

    def start(self, *a):
        self.args = a
        return b"AP-REQ"

    def step(self, token):
        assert token == b"AP-REP"
        return b""

    def unwrap(self, token):
        assert token == b"wrapped-key"
        return self.key, self.confidential


@pytest.fixture
def gss(monkeypatch):
    context = GSS()
    monkeypatch.setattr(K, "_NativeGSS", lambda: context)
    return context


def transcript(result=0):
    return b"\x00" * 4 + blob(b"AP-REP") + blob(b"wrapped-key") + struct.pack(">I", result)


def test_type35_uses_unwrapped_key_and_consumes_only_security_result(gss):
    sock = Socket(transcript() + b"NEXT")
    assert K.do_kerberos_auth(sock, "alice", "", host="mac.local") == gss.key
    assert sock.sent == [b"\x23\x00\x00\x00\x00", blob(b"AP-REQ")]
    assert sock.data == b"NEXT"
    assert gss.args[:3] == ("mac.local", "alice", "")
    assert gss.closed


@pytest.mark.parametrize("confidential,key", [(False, b"k" * 16), (True, b"k" * 15), (True, b"k" * 32)])
def test_rejects_unprotected_or_wrong_length_key(gss, confidential, key):
    gss.confidential, gss.key = confidential, key
    with pytest.raises(AuthError, match="confidential 16-byte"):
        K.do_kerberos_auth(Socket(transcript()), "alice", "", host="mac.local")
    assert gss.closed


def test_rejects_failed_security_result(gss):
    with pytest.raises(AuthError, match="result=1"):
        K.do_kerberos_auth(Socket(transcript(1)), "alice", "", host="mac.local")


@pytest.mark.parametrize("length", [0, K._MAX_TOKEN + 1, 0xffffffff])
def test_token_length_is_bounded_before_reading_body(length):
    sock = Socket(struct.pack(">I", length) + b"unread")
    with pytest.raises(AuthError, match="token length"):
        K._read_token(sock)
    assert sock.data == b"unread"


def test_kerberos_failure_closes_socket_without_password_fallback(monkeypatch):
    sock = Socket()
    monkeypatch.setattr(N, "_open_socket_and_handshake", lambda *a, **kw: sock)
    monkeypatch.setattr(K, "do_kerberos_auth", lambda *a, **kw: (_ for _ in ()).throw(AuthError("no tickets")))
    monkeypatch.setattr(N, "do_srp_auth", lambda *a: pytest.fail("password fallback"))
    with pytest.raises(AuthError, match="no tickets"):
        N._phase_auth("127.0.0.1", 5900, "alice", "", "kerberos")
    assert sock.closed


def test_checks_server_offers_type35_before_sending_auth(monkeypatch):
    sock = Socket()
    monkeypatch.setattr(N.socket, "create_connection", lambda *a, **kw: sock)
    monkeypatch.setattr(N, "do_protocol_handshake", lambda *a: b"\x21\x24")
    with pytest.raises(AuthError, match="type 35"):
        N._open_socket_and_handshake("mac.local", 5900, security_type=35)
    assert sock.closed and not sock.sent


def test_original_hostname_used_for_kerberos_identity(monkeypatch):
    sock = Socket()
    seen = {}
    monkeypatch.setattr(N, "_open_socket_and_handshake", lambda *a, **kw: sock)
    def authenticate(*args, **kwargs):
        seen.update(kwargs)
        return b"k" * 16
    monkeypatch.setattr(K, "do_kerberos_auth", authenticate)
    assert N._phase_auth("127.0.0.1", 5900, "alice", "", "kerberos", krb_host="mac.local")[0] is sock
    assert seen["host"] == "mac.local"


def test_ticket_only_alt_session_requires_password_before_cmd2(monkeypatch):
    sock = Socket()
    monkeypatch.setattr(N, "_phase_auth", lambda *a, **kw: (sock, b"k" * 16))
    monkeypatch.setattr(N, "_phase_client_init", lambda *a: (1920, 1080))
    monkeypatch.setattr(N, "_peek_ss_prompt", lambda *a: True)
    with pytest.raises(AuthError, match="separate user session requires a password"):
        N.connect_and_negotiate("mac.local", 5900, "alice", "", auth_mode="kerberos",
                                alt_session=True, video_offer=b"v", audio_offer=b"a")
    assert sock.closed


@pytest.fixture
def native(monkeypatch):
    """Exercise real identity/acquisition logic with GSS calls stubbed only."""
    from types import SimpleNamespace
    gss = K._NativeGSS.__new__(K._NativeGSS)
    gss.credential, gss.context = K.C.c_void_p(), K.C.c_void_p()
    gss.imported, gss.initial = [], []
    def name(value, oid):
        gss.imported.append(value)
        return value
    gss._name = name
    def acquire(*args):
        args[5]._obj.value = 1
        return 0
    def release(minor, pointer, *args):
        pointer._obj.value = None
    gss.lib = SimpleNamespace(gss_acquire_cred=acquire, gss_release_cred=release,
                              gss_delete_sec_context=release)
    def initial(*args):
        gss.initial.append(args)
        gss.credential.value = 2
    gss._initial_apple_credential = initial
    gss.step = lambda *args: b"AP-REQ"
    return gss


def test_blank_realm_uses_discovery_for_client_and_service(monkeypatch, native):
    realm = "LKDC:SHA1." + "A" * 40
    hosts = []
    def discover(host):
        hosts.append(host)
        return realm
    monkeypatch.setattr(K, "discover_lkdc_realm", discover)
    assert native.start("192.0.2.1", "alice", "", "", "", "") == b"AP-REQ"
    assert hosts == ["192.0.2.1"]
    assert native.imported == ["alice@" + realm, f"vnc/{realm}@{realm}"]
    assert not native.initial


@pytest.mark.parametrize("user,realm,principal,service,client,target", [
    ("alice", "EXAMPLE.COM", "", "", "alice@EXAMPLE.COM", "vnc@mac.local"),
    ("alice@EXAMPLE.COM", "", "", "", "alice@EXAMPLE.COM", "vnc@mac.local"),
    ("alice", "", "alice@EXAMPLE.COM", "", "alice@EXAMPLE.COM", "vnc@mac.local"),
    ("alice", "", "", "vnc/host@EXAMPLE.COM", "alice@EXAMPLE.COM", "vnc/host@EXAMPLE.COM"),
])
def test_explicit_identity_does_not_query_remote_realm(monkeypatch, native, user, realm, principal, service, client, target):
    monkeypatch.setattr(K, "discover_lkdc_realm", lambda *a: pytest.fail("unexpected discovery"))
    native.start("mac.local", user, "", realm, principal, service)
    assert native.imported == [client, target]


def test_first_password_login_discovers_then_acquires_ticket(monkeypatch, native):
    monkeypatch.setattr(K.sys, "platform", "darwin")
    realm = "LKDC:SHA1." + "B" * 40
    monkeypatch.setattr(K, "discover_lkdc_realm", lambda *a: realm)
    native.lib.gss_acquire_cred = lambda *a: 0x70000
    assert native.start("mac.local", "alice", "secret", "", "", "") == b"AP-REQ"
    assert native.initial == [("alice@" + realm, "mac.local", "secret")]


def test_first_login_without_password_requests_mac_password(monkeypatch, native):
    native.lib.gss_acquire_cred = lambda *a: 0x70000
    with pytest.raises(AuthError, match="enter the target Mac account password"):
        native.start("mac.local", "alice", "", "EXAMPLE.COM", "", "")
    assert not native.initial


def test_unusable_cached_ticket_reacquired_once_with_password(monkeypatch, native):
    monkeypatch.setattr(K.sys, "platform", "darwin")
    calls = []
    def step(*a):
        calls.append(native.credential.value)
        if len(calls) == 1:
            native.context.value = 3
            raise AuthError("cached ticket expired")
        assert native.context.value is None
        return b"AP-REQ"
    native.step = step
    assert native.start("mac.local", "alice", "secret", "EXAMPLE.COM", "", "") == b"AP-REQ"
    assert calls == [1, 2] and len(native.initial) == 1


def test_failed_fresh_ticket_not_retried_indefinitely(monkeypatch, native):
    monkeypatch.setattr(K.sys, "platform", "darwin")
    def step(*a):
        raise AuthError("ticket failed")
    native.step = step
    with pytest.raises(AuthError, match="ticket failed"):
        native.start("mac.local", "alice", "secret", "EXAMPLE.COM", "", "")
    assert len(native.initial) == 1


def test_first_login_cli_to_discovery_to_password_acquisition_to_type35(monkeypatch, native):
    """Only sockets and native GSS calls are simulated; orchestration is real."""
    from io import StringIO
    from pathlib import Path
    from isharescreen import cli
    from isharescreen.proxy.protocol import lkdc as L
    monkeypatch.setattr(K.sys, "platform", "darwin")
    monkeypatch.setattr(cli.sys, "stdin", StringIO("first-password\n"))
    cfg = cli._build_session_config(cli._make_parser().parse_args([
        "--host", "mac.local", "-u", "alice", "--auth", "kerberos",
        "--password-stdin", "--no-keyring",
    ]))
    assert cfg.krb_realm == cfg.krb_principal == cfg.krb_service == ""
    lines = (Path(__file__).parent / "fixtures/lkdc_referral.hex").read_text().splitlines()
    reply = bytes.fromhex("".join(line for line in lines if not line.startswith("#")))
    realm = L._referral_realm(reply)
    class KDCSocket(Socket):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
    kdc = KDCSocket(blob(reply))
    rfb = Socket(b"RFB 003.889\n\x01\x23" + transcript())
    destinations = []
    def connect(address, **kwargs):
        destinations.append(address)
        if address == ("192.0.2.1", 5900):
            return rfb
        assert address == ("mac.local", 88)
        return kdc
    monkeypatch.setattr(N.socket, "create_connection", connect)
    native.names = []
    native.lib.gss_acquire_cred = lambda *a: 0x70000
    native.step = lambda token=None: b"AP-REQ" if token is None else b""
    native.unwrap = lambda token: (b"k" * 16, True)
    monkeypatch.setattr(K, "_NativeGSS", lambda: native)
    sock, key = N._phase_auth("192.0.2.1", cfg.port, cfg.username, cfg.password,
                               cfg.auth_mode, krb_host=cfg.host)
    assert key == b"k" * 16 and sock is rfb
    assert native.initial == [("alice@" + realm, "mac.local", "first-password")]
    assert native.imported == ["alice@" + realm, f"vnc/{realm}@{realm}"]
    assert destinations == [("192.0.2.1", 5900), ("mac.local", 88)]
    assert rfb.sent == [b"RFB 003.889\n", b"\x23\x00\x00\x00\x00", blob(b"AP-REQ")]
    assert b"first-password" not in b"".join(kdc.sent + rfb.sent)
    assert kdc.closed and native.credential.value is None
    sock.close()
