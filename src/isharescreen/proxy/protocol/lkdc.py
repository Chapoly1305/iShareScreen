"""Discover a remote Mac's LKDC realm without credentials or remote login.

Send a canonicalizing AS-REQ for WELLKNOWN:COM.APPLE.LKDC. Apple's KDC
returns KDC_ERR_WRONG_REALM (68), with the real realm in KRB-ERROR.crealm.
See apple-oss-distributions/Heimdal kdc/kerberos5.c and RFC 4120 section 5.9.1.
Only this small DER subset is decoded; responses and network waits are bounded.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import re
import secrets
import socket
import struct
import time

from .auth import AuthError

_WELLKNOWN = b"WELLKNOWN:COM.APPLE.LKDC"
_MAX_REPLY = 65536
_TIMEOUT = 3.0  # each transport, including all of its socket reads


def _der(tag: int, value: bytes) -> bytes:
    n = len(value)
    if n < 128:
        length = bytes([n])
    else:
        count = (n.bit_length() + 7) // 8
        length = bytes([0x80 | count]) + n.to_bytes(count, "big")
    return bytes([tag]) + length + value


def _integer(value: int) -> bytes:
    raw = value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")
    if raw[0] & 0x80:
        raw = b"\x00" + raw
    return _der(2, raw)


def _principal(kind: int, *parts: bytes) -> bytes:
    return _der(0x30, _der(0xa0, _integer(kind)) +
                _der(0xa1, _der(0x30, b"".join(_der(0x1b, part) for part in parts))))


def _realm_request() -> bytes:
    till = (datetime.now(timezone.utc) + timedelta(minutes=5)).strftime("%Y%m%d%H%M%SZ").encode("ascii")
    body = b"".join([
        _der(0xa0, _der(3, b"\x00\x00\x01\x00\x00")),  # canonicalize (bit 15)
        _der(0xa1, _principal(1, b"iss-realm-discovery")),
        _der(0xa2, _der(0x1b, _WELLKNOWN)),
        _der(0xa3, _principal(2, b"krbtgt", _WELLKNOWN)),
        _der(0xa5, _der(0x18, till)),
        _der(0xa7, _integer(secrets.randbits(31))),
        _der(0xa8, _der(0x30, _integer(18) + _integer(17))),
    ])
    return _der(0x6a, _der(0x30, _der(0xa1, _integer(5)) +
                           _der(0xa2, _integer(10)) + _der(0xa4, _der(0x30, body))))


def _items(data: bytes):
    """Read one level of definite-length DER, with strict bounds."""
    pos = 0
    while pos < len(data):
        if pos + 2 > len(data):
            raise ValueError("truncated DER header")
        tag, length = data[pos], data[pos + 1]
        pos += 2
        if tag & 0x1f == 0x1f:
            raise ValueError("unsupported DER tag")
        if length & 0x80:
            count = length & 0x7f
            if not 1 <= count <= 4 or pos + count > len(data) or data[pos] == 0:
                raise ValueError("invalid DER length")
            length = int.from_bytes(data[pos:pos + count], "big")
            pos += count
            if length < 128:
                raise ValueError("non-minimal DER length")
        end = pos + length
        if end > len(data):
            raise ValueError("truncated DER value")
        yield tag, data[pos:end]
        pos = end


def _only(data: bytes, tag: int) -> bytes:
    items = list(_items(data))
    if len(items) != 1 or items[0][0] != tag:
        raise ValueError("unexpected DER field")
    return items[0][1]


def _number(data: bytes) -> int:
    value = _only(data, 2)
    if not 1 <= len(value) <= 4 or value[0] & 0x80:
        raise ValueError("invalid DER integer")
    return int.from_bytes(value, "big")


def _referral_realm(reply: bytes) -> str:
    if len(reply) > _MAX_REPLY:
        raise AuthError("LKDC discovery response is too large")
    try:
        fields = {}
        for tag, value in _items(_only(_only(reply, 0x7e), 0x30)):
            if not 0xa0 <= tag <= 0xac or tag in fields:
                raise ValueError("invalid or duplicate KRB-ERROR field")
            fields[tag] = value
        if _number(fields[0xa0]) != 5 or _number(fields[0xa1]) != 30:
            raise ValueError("not a Kerberos 5 error")
        code = _number(fields[0xa6])
        if code != 68:
            raise AuthError(f"Target KDC did not return an LKDC referral (error {code}); "
                            "set the Kerberos realm manually for a domain KDC")
        realm = _only(fields[0xa7], 0x1b)
        if re.fullmatch(rb"LKDC:SHA1\.[0-9A-Fa-f]{40}", realm) is None:
            raise ValueError("invalid LKDC realm")
        return realm.decode("ascii")
    except (ValueError, KeyError):
        raise AuthError("Target returned an invalid LKDC realm referral") from None


def _recv(sock, length: int, deadline: float) -> bytes:
    buf = bytearray()
    while len(buf) < length:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("LKDC discovery timed out")
        sock.settimeout(remaining)
        part = sock.recv(length - len(buf))
        if not part:
            raise OSError("LKDC discovery connection closed")
        buf.extend(part)
    return bytes(buf)


def discover_lkdc_realm(host: str) -> str:
    """Query only the given host, first TCP/88 then UDP/88 on transport error."""
    request = _realm_request()
    try:
        deadline = time.monotonic() + _TIMEOUT
        with socket.create_connection((host, 88), timeout=_TIMEOUT) as sock:
            sock.settimeout(max(0.001, deadline - time.monotonic()))
            sock.sendall(struct.pack(">I", len(request)) + request)
            size = struct.unpack(">I", _recv(sock, 4, deadline))[0]
            if not 0 < size <= _MAX_REPLY:
                raise AuthError("Invalid LKDC discovery response length")
            return _referral_realm(_recv(sock, size, deadline))
    except OSError:
        pass
    try:
        addresses = socket.getaddrinfo(host, 88, type=socket.SOCK_DGRAM)
        family, kind, protocol, _, address = addresses[0]
        with socket.socket(family, kind, protocol) as sock:
            sock.settimeout(_TIMEOUT)
            sock.connect(address)  # accept datagrams only from the target KDC
            sock.send(request)
            return _referral_realm(sock.recv(_MAX_REPLY + 1))
    except (OSError, IndexError):
        raise AuthError("Could not discover the target Mac's Kerberos realm; "
                        "check connectivity to its KDC on TCP/UDP 88, "
                        "or enter the realm manually") from None
