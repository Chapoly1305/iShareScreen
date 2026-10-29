"""Win32 ABI and callback lifetime, using a fake user32 (no installed hook)."""
from __future__ import annotations

import ctypes
import sys
import types

import pytest

from isharescreen.frontend.desktop import keyboard_grab as K

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Win32 ABI")


class Function:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


def test_hook_keeps_one_callback_and_pointer_sized_results(monkeypatch):
    import glfw

    callbacks = []
    wide_result = 0x100000001 if ctypes.sizeof(ctypes.c_void_p) == 8 else 123
    api = types.SimpleNamespace(
        SetWindowsHookExW=Function(lambda kind, callback, *rest: callbacks.append(callback) or 42),
        UnhookWindowsHookEx=Function(lambda _: True),
        CallNextHookEx=Function(lambda *args: wide_result),
        GetForegroundWindow=Function(lambda: 123),
    )
    monkeypatch.setattr(ctypes, "WinDLL", lambda *a, **k: api)
    monkeypatch.setattr(glfw, "get_win32_window", lambda _: 123)
    monkeypatch.setattr(K.atexit, "register", lambda _: None)
    grab = K._Win32Grab(object(), lambda *args: None)
    assert grab.enable()
    callback = callbacks[-1]
    assert ctypes.sizeof(grab._HOOKPROC._restype_) == ctypes.sizeof(ctypes.c_void_p)
    assert callback(-1, 0, 0) == wide_result
    assert api.CallNextHookEx.restype is ctypes.c_ssize_t
    grab.disable()
    assert grab._cb_ref is callback
    assert grab.enable()
    assert callbacks[-1] is callback
    grab.disable()
