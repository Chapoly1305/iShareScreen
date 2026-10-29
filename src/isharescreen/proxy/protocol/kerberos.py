"""Apple RFB security type 35 using the system GSS library.

Wire reference: Chapoly1305/libvncserver, ard.c HandleARDAuth35.
Reuse cached credentials first. macOS can additionally obtain initial LKDC
credentials with gss_aapl_initial_cred when a password was supplied.
The record-layer key comes only from the confidential GSS wrap token, never
from a truncated Kerberos context key.
"""
from __future__ import annotations

import ctypes as C
import ctypes.util
import logging
import struct
import sys

from .auth import AuthError
from .rfb import recv_exact
from .lkdc import discover_lkdc_realm

log = logging.getLogger("iss.auth.kerberos")
_MAX_TOKEN = 1024 * 1024
_FLAGS = 2 | 8 | 16 | 32  # mutual, sequence, confidentiality, integrity


class _Buffer(C.Structure):
    _fields_ = [("length", C.c_size_t), ("value", C.c_void_p)]


class _OID(C.Structure):
    _fields_ = [("length", C.c_uint32), ("elements", C.c_void_p)]


class _OIDSet(C.Structure):
    _fields_ = [("count", C.c_size_t), ("elements", C.POINTER(_OID))]


def _oid(raw: bytes) -> _OID:
    return _OID(len(raw), C.cast(C.c_char_p(raw), C.c_void_p))


_KERBEROS = _oid(b"\x2a\x86\x48\x86\xf7\x12\x01\x02\x02")
_PRINCIPAL = _oid(b"\x2a\x86\x48\x86\xf7\x12\x01\x02\x02\x01")
_HOST_SERVICE = _oid(b"\x2a\x86\x48\x86\xf7\x12\x01\x02\x01\x04")


def _buffer(raw: bytes) -> _Buffer:
    return _Buffer(len(raw), C.cast(C.c_char_p(raw), C.c_void_p))


class _NativeGSS:
    def __init__(self):
        if sys.platform == "darwin":
            path = "/System/Library/Frameworks/GSS.framework/GSS"
        elif sys.platform.startswith("linux"):
            path = ctypes.util.find_library("gssapi_krb5")
        else:
            path = None
        if not path:
            raise AuthError("Kerberos requires macOS GSS or Linux libgssapi-krb5")
        try:
            self.lib = C.CDLL(path)
            self._bind()
        except (OSError, AttributeError):
            raise AuthError("Could not load the system Kerberos GSS library") from None
        self.context = C.c_void_p()
        self.credential = C.c_void_p()
        self.names: list[C.c_void_p] = []

    def _bind(self):
        p, u, b, o = C.c_void_p, C.POINTER(C.c_uint32), C.POINTER(_Buffer), C.POINTER(_OID)
        signatures = {
            "gss_import_name": [u, b, o, C.POINTER(p)],
            "gss_acquire_cred": [u, p, C.c_uint32, C.POINTER(_OIDSet), C.c_int,
                                 C.POINTER(p), p, p],
            "gss_init_sec_context": [u, p, C.POINTER(p), p, o, C.c_uint32,
                                     C.c_uint32, p, b, p, b, u, p],
            "gss_unwrap": [u, p, b, b, C.POINTER(C.c_int), u],
            "gss_release_buffer": [u, b],
            "gss_release_name": [u, C.POINTER(p)],
            "gss_release_cred": [u, C.POINTER(p)],
            "gss_delete_sec_context": [u, C.POINTER(p), b],
        }
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, C.c_uint32

    @staticmethod
    def _check(major, minor, stage, *, continuing=False):
        if major == 0 or (continuing and major == 1):
            return
        raise AuthError(
            f"Kerberos {stage} failed (GSS {major:#x}/{minor.value:#x}); "
            "check tickets, principal, realm and service name"
        )

    def _name(self, text, oid):
        raw = text.encode("utf-8")
        buf, name, minor = _buffer(raw), C.c_void_p(), C.c_uint32()
        major = self.lib.gss_import_name(C.byref(minor), C.byref(buf), C.byref(oid), C.byref(name))
        if name:
            self.names.append(name)
        self._check(major, minor, "import name")
        return name

    def start(self, host, username, password, realm, principal, service):
        realm = (realm or "").strip()
        principal = (principal or "").strip()
        service = (service or "").strip()
        # Explicit identities take precedence over discovery, including a
        # fully qualified account or service in a conventional domain realm.
        if not realm:
            identity = principal or username
            if "@" in identity:
                realm = identity.rsplit("@", 1)[1]
            elif "@" in service and "/" in service:
                realm = service.rsplit("@", 1)[1]
            else:
                log.info("Discovering target Mac's LKDC realm")
                realm = discover_lkdc_realm(host)
                log.info("Discovered Kerberos realm: %s", realm)
        client = principal or (username if "@" in username else
                               f"{username}@{realm}" if realm else username)
        realm = realm or (client.rsplit("@", 1)[1] if "@" in client else "")
        target = service or (f"vnc/{realm}@{realm}" if realm.startswith("LKDC:")
                             else f"vnc@{host}")
        user = self._name(client, _PRINCIPAL)
        self.target = self._name(target, _PRINCIPAL if "/" in target else _HOST_SERVICE)
        mechanisms, minor = _OIDSet(1, C.pointer(_KERBEROS)), C.c_uint32()
        major = self.lib.gss_acquire_cred(C.byref(minor), user, 0, C.byref(mechanisms), 1,
                                         C.byref(self.credential), None, None)
        if major != 0 and password and sys.platform == "darwin":
            if self.credential:
                self.lib.gss_release_cred(C.byref(minor), C.byref(self.credential))
            self._initial_apple_credential(user, host, password)
            acquired_initial = True
        else:
            if major != 0 and not password:
                raise AuthError(
                    f"No usable cached Kerberos ticket (GSS {major:#x}/{minor.value:#x}); "
                    "enter the target Mac account password for the first connection"
                )
            self._check(major, minor, "acquire cached credential (use kinit first)")
            acquired_initial = False
        if not self.credential:
            raise AuthError("Kerberos did not return an initiating credential")
        try:
            return self.step()
        except AuthError:
            # A cached TGT can be acquired successfully yet expire/be rejected
            # while requesting the service ticket. Reacquire once before any
            # AP-REQ is sent to RFB; never retry a rejected server AP-REP.
            if acquired_initial or not password or sys.platform != "darwin":
                raise
            self._release_context()
            self.lib.gss_release_cred(C.byref(minor), C.byref(self.credential))
            log.info("Cached Kerberos credential unusable; obtaining fresh tickets")
            self._initial_apple_credential(user, host, password)
            return self.step()

    def _initial_apple_credential(self, user, host, password):
        cf = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
        p = C.c_void_p
        cf.CFStringCreateWithCString.argtypes = [p, C.c_char_p, C.c_uint32]
        cf.CFStringCreateWithCString.restype = p
        cf.CFDictionaryCreate.argtypes = [p, C.POINTER(p), C.POINTER(p), C.c_long, p, p]
        cf.CFDictionaryCreate.restype = p
        cf.CFRelease.argtypes, cf.CFRelease.restype = [p], None
        cf.CFErrorGetCode.argtypes, cf.CFErrorGetCode.restype = [p], C.c_long
        cf.CFErrorCopyDescription.argtypes, cf.CFErrorCopyDescription.restype = [p], p
        cf.CFStringGetCString.argtypes = [p, C.c_char_p, C.c_long, C.c_uint32]
        cf.CFStringGetCString.restype = C.c_bool
        fn = self.lib.gss_aapl_initial_cred
        fn.argtypes = [p, C.POINTER(_OID), p, C.POINTER(p), C.POINTER(p)]
        fn.restype = C.c_uint32
        refs = []
        error = C.c_void_p()
        try:
            def string(value):
                ref = cf.CFStringCreateWithCString(None, value.encode("utf-8"), 0x08000100)
                if not ref:
                    raise AuthError("Could not allocate Kerberos credential attributes")
                refs.append(ref)
                return ref

            # GSS queries with its own CFSTR keys. CFType callbacks are needed
            # for string-content equality, in addition to retaining values;
            # NULL callbacks compare pointers and hide the password from GSS.
            keys = (p * 3)(*[string(v) for v in
                            ("kGSSICPassword", "kGSSCredentialUsage", "kGSSICLKDCHostname")])
            values = (p * 3)(*[string(v) for v in (password, "kGSS_C_INITIATE", host)])
            key_callbacks = C.addressof(C.c_byte.in_dll(cf, "kCFTypeDictionaryKeyCallBacks"))
            value_callbacks = C.addressof(C.c_byte.in_dll(cf, "kCFTypeDictionaryValueCallBacks"))
            attrs = cf.CFDictionaryCreate(None, keys, values, 3, key_callbacks, value_callbacks)
            if not attrs:
                raise AuthError("Could not allocate Kerberos credential dictionary")
            refs.append(attrs)
            major = fn(user, C.byref(_KERBEROS), attrs, C.byref(self.credential), C.byref(error))
            if major != 0 and error:
                code = cf.CFErrorGetCode(error)
                description = cf.CFErrorCopyDescription(error)
                detail = ""
                try:
                    if description:
                        buf = C.create_string_buffer(4096)
                        if cf.CFStringGetCString(description, buf, len(buf), 0x08000100):
                            detail = buf.value.decode("utf-8", errors="replace")
                            if password:
                                detail = detail.replace(password, "[redacted]")
                            detail = " ".join(detail.split())[:512]
                finally:
                    if description:
                        cf.CFRelease(description)
                raise AuthError(
                    f"Kerberos obtain initial LKDC credential failed "
                    f"(GSS {major:#x}; CFError {code})" + (f": {detail}" if detail else "")
                )
            self._check(major, C.c_uint32(), "obtain initial LKDC credential")
        finally:
            if error:
                cf.CFRelease(error)
            for ref in reversed(refs):
                cf.CFRelease(ref)

    def step(self, token=None):
        minor, flags, output = C.c_uint32(), C.c_uint32(), _Buffer()
        input_buffer = _buffer(token) if token is not None else None
        try:
            major = self.lib.gss_init_sec_context(
                C.byref(minor), self.credential, C.byref(self.context), self.target,
                C.byref(_KERBEROS), _FLAGS, 0, None,
                C.byref(input_buffer) if input_buffer is not None else None,
                None, C.byref(output), C.byref(flags), None,
            )
            self._check(major, minor, "establish context", continuing=token is None)
            if token is not None and flags.value & _FLAGS != _FLAGS:
                raise AuthError("Kerberos context lacks required security flags")
            return C.string_at(output.value, output.length) if output.length else b""
        finally:
            if output.value:
                self.lib.gss_release_buffer(C.byref(minor), C.byref(output))

    def unwrap(self, token):
        minor, confidential, output, inp = C.c_uint32(), C.c_int(), _Buffer(), _buffer(token)
        try:
            major = self.lib.gss_unwrap(C.byref(minor), self.context, C.byref(inp),
                                         C.byref(output), C.byref(confidential), None)
            self._check(major, minor, "unwrap session key")
            return (C.string_at(output.value, output.length) if output.length else b"",
                    bool(confidential.value))
        finally:
            if output.value:
                self.lib.gss_release_buffer(C.byref(minor), C.byref(output))

    def __enter__(self):
        return self

    def _release_context(self):
        minor = C.c_uint32()
        if self.context:
            self.lib.gss_delete_sec_context(C.byref(minor), C.byref(self.context), None)

    def __exit__(self, *exc):
        minor = C.c_uint32()
        self._release_context()
        if self.credential:
            self.lib.gss_release_cred(C.byref(minor), C.byref(self.credential))
        for name in self.names:
            self.lib.gss_release_name(C.byref(minor), C.byref(name))


def _read_token(sock):
    length = struct.unpack(">I", recv_exact(sock, 4))[0]
    if not 0 < length <= _MAX_TOKEN:
        raise AuthError(f"Invalid Kerberos token length: {length}")
    return recv_exact(sock, length)


def _gss_backend():
    if sys.platform == "win32":
        from .kerberos_spnego import SpnegoGSS
        return SpnegoGSS()
    return _NativeGSS()


def do_kerberos_auth(sock, username, password, *, host, realm="", principal="", service=""):
    """Return the server-generated 16-byte Apple record-layer wrap key."""
    with _gss_backend() as gss:
        request = gss.start(host, username, password, realm, principal, service)
        if not request or len(request) > _MAX_TOKEN:
            raise AuthError("Invalid Kerberos AP-REQ")
        sock.sendall(b"\x23\x00\x00\x00\x00")
        if recv_exact(sock, 4) != b"\x00" * 4:
            raise AuthError("Kerberos server rejected preface")
        sock.sendall(struct.pack(">I", len(request)) + request)
        if gss.step(_read_token(sock)):
            raise AuthError("Unexpected Kerberos AP-REP output token")
        key, confidential = gss.unwrap(_read_token(sock))
        if not confidential or len(key) != 16:
            raise AuthError("Kerberos server did not send a confidential 16-byte session key")
        # Match the reference client: one standard RFB SecurityResult follows
        # the wrap token. Do not consume another zero word as ClientInit data.
        result = struct.unpack(">I", recv_exact(sock, 4))[0]
        if result:
            raise AuthError(f"Kerberos authentication rejected: result={result}")
        log.info("AUTH OK (Kerberos)")
        return key
