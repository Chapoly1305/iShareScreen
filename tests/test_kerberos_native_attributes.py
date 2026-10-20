"""Use real macOS CF dictionaries; stub only the final GSS network call."""
import ctypes as C
import sys
from types import SimpleNamespace

import pytest

from isharescreen.proxy.protocol.kerberos import _NativeGSS
from isharescreen.proxy.protocol.auth import AuthError

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="native macOS CoreFoundation")


@pytest.fixture
def cf():
    lib = C.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
    p = C.c_void_p
    lib.CFStringCreateWithCString.argtypes = [p, C.c_char_p, C.c_uint32]
    lib.CFStringCreateWithCString.restype = p
    lib.CFDictionaryGetValue.argtypes, lib.CFDictionaryGetValue.restype = [p, p], p
    lib.CFDictionaryCreate.argtypes = [p, C.POINTER(p), C.POINTER(p), C.c_long, p, p]
    lib.CFDictionaryCreate.restype = p
    lib.CFStringGetCString.argtypes = [p, C.c_char_p, C.c_long, C.c_uint32]
    lib.CFStringGetCString.restype = C.c_bool
    lib.CFRelease.argtypes, lib.CFRelease.restype = [p], None
    lib.CFErrorCreate.argtypes = [p, p, C.c_long, p]
    lib.CFErrorCreate.restype = p
    return lib


def string(cf, text):
    return cf.CFStringCreateWithCString(None, text.encode(), 0x08000100)


def lookup(cf, attrs, key):
    # A distinct CFString with the same content reproduces GSS's CFSTR lookup.
    ref = string(cf, key)
    try:
        found = cf.CFDictionaryGetValue(attrs, ref)
        assert found, f"GSS cannot read {key} from the native dictionary"
        buf = C.create_string_buffer(1024)
        assert cf.CFStringGetCString(found, buf, len(buf), 0x08000100)
        return buf.value.decode()
    finally:
        cf.CFRelease(ref)


def native_with_call(call):
    gss = _NativeGSS.__new__(_NativeGSS)
    gss.credential = C.c_void_p()
    gss.lib = SimpleNamespace(gss_aapl_initial_cred=call)
    return gss


def test_gss_can_read_password_usage_and_host_by_key_content(cf):
    calls = []
    def acquire(user, mechanism, attrs, credential, error):
        calls.append(True)
        assert lookup(cf, attrs, "kGSSICPassword") == "synthetic-密码"
        assert lookup(cf, attrs, "kGSSCredentialUsage") == "kGSS_C_INITIATE"
        assert lookup(cf, attrs, "kGSSICLKDCHostname") == "mac.test"
        credential._obj.value = 123
        return 0
    gss = native_with_call(acquire)
    gss._initial_apple_credential(C.c_void_p(1), "mac.test", "synthetic-密码")
    assert calls == [True] and gss.credential.value == 123


def test_apple_error_code_and_description_surface_without_password(cf):
    def acquire(user, mechanism, attrs, credential, error):
        # A real CFError verifies ownership, description conversion and cleanup.
        refs = [string(cf, "NSLocalizedDescription"), string(cf, "pre-auth failed synthetic-secret"),
                string(cf, "iss-test")]
        info = None
        try:
            p = C.c_void_p
            info = cf.CFDictionaryCreate(None, (p * 1)(refs[0]), (p * 1)(refs[1]), 1,
                C.addressof(C.c_byte.in_dll(cf, "kCFTypeDictionaryKeyCallBacks")),
                C.addressof(C.c_byte.in_dll(cf, "kCFTypeDictionaryValueCallBacks")))
            error._obj.value = cf.CFErrorCreate(None, refs[2], 12345, info)
        finally:
            if info:
                cf.CFRelease(info)
            for ref in refs:
                cf.CFRelease(ref)
        return 0xd0000
    gss = native_with_call(acquire)
    with pytest.raises(AuthError) as caught:
        gss._initial_apple_credential(C.c_void_p(1), "mac.test", "synthetic-secret")
    message = str(caught.value)
    assert "CFError 12345" in message
    assert "pre-auth failed" in message and "[redacted]" in message
    assert "synthetic-secret" not in message
