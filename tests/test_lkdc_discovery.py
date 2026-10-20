"""Live-response fixture, malformed referrals, and bounded KDC transports."""
from pathlib import Path
import socket
import struct

import pytest

from isharescreen.proxy.protocol import lkdc as L
from isharescreen.proxy.protocol.auth import AuthError

REALM = "LKDC:SHA1.000102030405060708090A0B0C0D0E0F10111213"


@pytest.fixture
def reply():
    lines = (Path(__file__).parent / "fixtures/lkdc_referral.hex").read_text().splitlines()
    return bytes.fromhex("".join(line for line in lines if not line.startswith("#")))


def change_fields(reply, transform):
    fields = list(L._items(L._only(L._only(reply, 0x7e), 0x30)))
    return L._der(0x7e, L._der(0x30, b"".join(L._der(t, v) for t, v in transform(fields))))


def test_parses_sanitized_live_apple_response(reply):
    assert L._referral_realm(reply) == REALM


def test_uses_client_realm_not_server_realm(reply):
    assert L._referral_realm(change_fields(reply, lambda fs: [
        (t, L._der(0x1b, b"WELLKNOWN:COM.APPLE.LKDC") if t == 0xa9 else v)
        for t, v in fs
    ])) == REALM


def test_missing_client_realm_cannot_be_replaced_by_realm_in_error_text(reply):
    response = change_fields(reply, lambda fs: [
        (t, L._der(0x1b, REALM.encode()) if t == 0xab else v)
        for t, v in fs if t != 0xa7
    ])
    with pytest.raises(AuthError, match="invalid LKDC realm referral"):
        L._referral_realm(response)


def test_rejects_duplicate_realm(reply):
    response = change_fields(reply, lambda fs: fs + [(0xa7, L._der(0x1b, REALM.encode()))])
    with pytest.raises(AuthError, match="invalid LKDC realm referral"):
        L._referral_realm(response)


@pytest.mark.parametrize("realm", [b"EXAMPLE.COM", b"LKDC:SHA1.bad", b"LKDC:SHA1." + b"A" * 40 + b"\n", b"\xff"])
def test_rejects_invalid_realms(reply, realm):
    response = change_fields(reply, lambda fs: [
        (t, L._der(0x1b, realm) if t == 0xa7 else v) for t, v in fs
    ])
    with pytest.raises(AuthError, match="invalid LKDC realm referral"):
        L._referral_realm(response)


def test_non_referral_error_is_reported_without_using_server_text(reply):
    response = change_fields(reply, lambda fs: [
        (t, L._integer(6) if t == 0xa6 else v) for t, v in fs
    ])
    with pytest.raises(AuthError, match=r"error 6"):
        L._referral_realm(response)


@pytest.mark.parametrize("value", [b"", b"\x7e", b"\x7e\x80\x00\x00", b"\x7e\x84\xff\xff\xff\xff", b"\x7e\x81\x01\x00"])
def test_malformed_der_is_rejected(value):
    with pytest.raises(AuthError, match="invalid LKDC realm referral"):
        L._referral_realm(value)


def test_truncated_and_trailing_response_rejected(reply):
    for value in (reply[:-1], reply + b"\x00\x00"):
        with pytest.raises(AuthError):
            L._referral_realm(value)


def test_probe_is_canonicalizing_as_req_without_account_credentials():
    # Verify the required RFC 4120 fields, not a self-generated expected blob.
    fields = dict(L._items(L._only(L._only(L._realm_request(), 0x6a), 0x30)))
    assert L._number(fields[0xa1]) == 5 and L._number(fields[0xa2]) == 10
    body = dict(L._items(L._only(fields[0xa4], 0x30)))
    assert L._only(body[0xa2], 0x1b) == b"WELLKNOWN:COM.APPLE.LKDC"
    flags = L._only(body[0xa0], 3)
    assert flags[0] == 0 and int.from_bytes(flags[1:], "big") == 1 << 16
    assert 0xa3 not in fields  # no pre-auth / password material
    assert L._only(body[0xa5], 0x18).endswith(b"Z")


class TCP:
    def __init__(self, payload):
        self.payload, self.sent, self.closed = payload, b"", False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def settimeout(self, timeout):
        assert 0 < timeout <= L._TIMEOUT

    def sendall(self, data):
        self.sent += data

    def recv(self, n):
        size = min(n, 3)
        part, self.payload = self.payload[:size], self.payload[size:]
        return part


def test_tcp_discovery_reassembles_fragments_and_closes_socket(monkeypatch, reply):
    sock = TCP(struct.pack(">I", len(reply)) + reply)
    def connect(address, **kw):
        assert address == ("mac.local", 88)
        return sock
    monkeypatch.setattr(L.socket, "create_connection", connect)
    assert L.discover_lkdc_realm("mac.local") == REALM
    assert struct.unpack(">I", sock.sent[:4])[0] == len(sock.sent) - 4
    assert sock.sent[4] == 0x6a
    assert sock.closed


@pytest.mark.parametrize("size", [0, L._MAX_REPLY + 1, 0xffffffff])
def test_tcp_frame_length_checked_before_reading_body(monkeypatch, size):
    sock = TCP(struct.pack(">I", size) + b"unread")
    monkeypatch.setattr(L.socket, "create_connection", lambda *a, **kw: sock)
    with pytest.raises(AuthError, match="response length"):
        L.discover_lkdc_realm("mac.local")
    assert sock.payload == b"unread" and sock.closed


def test_udp_fallback_is_connected_to_the_same_target(monkeypatch, reply):
    def refused(*a, **kw):
        raise ConnectionRefusedError
    monkeypatch.setattr(L.socket, "create_connection", refused)
    monkeypatch.setattr(L.socket, "getaddrinfo", lambda *a, **kw: [
        (socket.AF_INET, socket.SOCK_DGRAM, 17, "", ("192.0.2.1", 88))])
    class UDP(TCP):
        def connect(self, target):
            assert target == ("192.0.2.1", 88)
        def send(self, data):
            self.sent += data
        def recv(self, n):
            assert n == L._MAX_REPLY + 1
            return self.payload
    sock = UDP(reply)
    monkeypatch.setattr(L.socket, "socket", lambda *a: sock)
    assert L.discover_lkdc_realm("mac.local") == REALM
    assert sock.sent.startswith(b"\x6a") and sock.closed


def test_unreachable_kdc_returns_actionable_error(monkeypatch):
    def unavailable(*a, **kw):
        raise TimeoutError
    monkeypatch.setattr(L.socket, "create_connection", unavailable)
    monkeypatch.setattr(L.socket, "getaddrinfo", unavailable)
    with pytest.raises(AuthError, match="TCP/UDP 88"):
        L.discover_lkdc_realm("mac.local")


def test_tcp_deadline_covers_all_reads(monkeypatch):
    times = iter([10.0, 11.0])
    monkeypatch.setattr(L.time, "monotonic", lambda: next(times))
    sock = TCP(b"aaaa")
    with pytest.raises(TimeoutError):
        L._recv(sock, 4, deadline=10.5)
    assert sock.payload == b"a"
