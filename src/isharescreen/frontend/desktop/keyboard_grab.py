"""Per-platform "grab the keyboard while iss has focus" so OS-level
shortcuts (GNOME Activities on bare Super, Windows Start menu on Win,
Alt+Tab) don't intercept keys destined for the remote Mac.

Same pattern remote-desktop clients use: grab on focus-in, release on
focus-out. iss wires this into glfwSetWindowFocusCallback in app.py.

Implementations:
  X11      XGrabKeyboard via ctypes->libX11 (TigerVNC pattern)
  Win32    SetWindowsHookEx(WH_KEYBOARD_LL) via ctypes->user32
  Wayland  unimplemented; would need pywayland +
           zwp_keyboard_shortcuts_inhibit_v1. Falls through to no-op
           with a warning. Use an X11 session for full grab on Linux.

Default off — opt in with ISS_GRAB_KEYBOARD=1. The Win32 hook in
particular intercepts every keystroke system-wide; it self-gates on
foreground-window match to keep that cheap, but users who don't need
it shouldn't pay for it.
"""
from __future__ import annotations

import atexit
import ctypes
import logging
import sys
import time
from typing import Optional, Protocol


log = logging.getLogger(__name__)


class KeyboardGrab(Protocol):
    """Minimal interface every platform impl satisfies."""

    def enable(self) -> bool: ...
    def disable(self) -> None: ...


# ── factory ──────────────────────────────────────────────────────────

def make_grab(glfw_window, key_sender=None) -> KeyboardGrab:
    """Pick the right impl for the current platform.

    `key_sender` is a callable `(is_down: bool, x11_keysym: int) -> None`
    used by the Win32 backend to deliver intercepted keys directly to
    iss's input pipeline (bypassing GLFW for hooked keys, which is what
    makes Win+letter combos actually reach the host as Cmd+letter
    instead of being eaten by Windows' own Win+C / Win+V / etc. OS
    shortcuts). The X11 backend doesn't need it — XGrabKeyboard routes
    every key including Win+combos through GLFW's normal callback.

    Returns a no-op on platforms we don't support (Mac, Linux/Wayland)
    so callers can treat the result uniformly."""
    if sys.platform == "win32":
        if key_sender is None:
            return _NoopGrab("Win32 keyboard grab needs a key_sender callable")
        try:
            return _Win32Grab(glfw_window, key_sender)
        except Exception as e:
            log.warning("Win32 keyboard grab unavailable: %s", e)
            return _NoopGrab(str(e))

    if sys.platform.startswith("linux"):
        # Detect X11 vs Wayland by attempting the X11 native helpers.
        # If they raise or return 0 the session is Wayland (or headless).
        try:
            import glfw  # type: ignore
            disp = glfw.get_x11_display()
            if disp:
                return _X11Grab(glfw_window)
        except Exception:
            pass
        return _NoopGrab(
            "Wayland keyboard grab not implemented; switch to an X11 "
            "session at login if you need iss to capture Win/Super combos"
        )

    return _NoopGrab("only Linux (X11) and Windows are supported")


# ── no-op fallback ───────────────────────────────────────────────────

class _NoopGrab:
    def __init__(self, why: str) -> None:
        self._why = why
        self._warned = False

    def enable(self) -> bool:
        if not self._warned:
            log.info("keyboard grab disabled: %s", self._why)
            self._warned = True
        return False

    def disable(self) -> None:
        return


# ── X11 implementation ───────────────────────────────────────────────

# X11 constants we need (from <X11/X.h>).
_GRAB_MODE_ASYNC = 1
_CURRENT_TIME = 0
# Return codes from XGrabKeyboard (<X11/X.h>).
_GRAB_SUCCESS = 0
_GRAB_ALREADY_GRABBED = 1


class _X11Grab:
    """XGrabKeyboard while iss has focus. Pattern copied from TigerVNC's
    vncviewer DesktopWindow.cxx — short retry loop on AlreadyGrabbed
    (some WMs hold transient grabs during their own animations) and an
    atexit safety release so a Python crash with the keyboard grabbed
    doesn't lock the user out of their X session."""

    def __init__(self, glfw_window) -> None:
        import glfw  # type: ignore
        try:
            self._libx = ctypes.CDLL("libX11.so.6")
        except OSError as e:
            raise RuntimeError(f"libX11.so.6 not found: {e}") from e

        self._libx.XGrabKeyboard.argtypes = [
            ctypes.c_void_p,    # display
            ctypes.c_ulong,     # window (XID)
            ctypes.c_int,       # owner_events (Bool)
            ctypes.c_int,       # pointer_mode
            ctypes.c_int,       # keyboard_mode
            ctypes.c_ulong,     # time
        ]
        self._libx.XGrabKeyboard.restype = ctypes.c_int
        self._libx.XUngrabKeyboard.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        self._libx.XUngrabKeyboard.restype = ctypes.c_int
        self._libx.XFlush.argtypes = [ctypes.c_void_p]
        self._libx.XFlush.restype = ctypes.c_int

        self._display = glfw.get_x11_display()
        self._window = glfw.get_x11_window(glfw_window)
        if not self._display or not self._window:
            raise RuntimeError(
                "glfw X11 native handles unavailable (window not yet realised?)"
            )
        self._held = False
        atexit.register(self._safety_release)

    def enable(self) -> bool:
        if self._held:
            return True
        for attempt in range(5):
            r = self._libx.XGrabKeyboard(
                self._display, self._window,
                1,                      # owner_events=True so iss still
                                        # gets normal X events for our window
                _GRAB_MODE_ASYNC,
                _GRAB_MODE_ASYNC,
                _CURRENT_TIME,
            )
            if r == _GRAB_SUCCESS:
                self._held = True
                self._libx.XFlush(self._display)
                log.info("keyboard grab acquired (X11)")
                return True
            if r != _GRAB_ALREADY_GRABBED:
                log.warning(
                    "keyboard grab failed: XGrabKeyboard returned %d", r,
                )
                return False
            time.sleep(0.1)
        log.warning(
            "keyboard grab gave up after 5 retries (something else is "
            "holding the X11 keyboard grab)",
        )
        return False

    def disable(self) -> None:
        if not self._held:
            return
        self._libx.XUngrabKeyboard(self._display, _CURRENT_TIME)
        self._libx.XFlush(self._display)
        self._held = False
        log.info("keyboard grab released (X11)")

    def _safety_release(self) -> None:
        # Best-effort release on interpreter shutdown so a stack trace
        # doesn't leave the user with a wedged X server.
        try:
            self.disable()
        except Exception:
            pass


# ── Win32 implementation ─────────────────────────────────────────────
#
# Approach: Moonlight / SDL2 pattern.
#
# Naive LL hook that just `return 1`s on Win/Alt suppresses the keys
# system-wide INCLUDING from GLFW — iss never sees Super held, Win+C
# arrives as bare 'c'. Re-posting via PostMessage doesn't help because
# GLFW reads modifier state from GetKeyState() which only reflects
# real hardware events. Re-injecting via SendInput re-triggers the
# OS Start-menu / Copilot / Clipboard-history logic.
#
# What actually works (per Moonlight's input.cpp + SDL's
# WIN_KeyboardHookProc): install the LL hook, intercept ALL keys
# while iss is foreground, maintain our own modifier bitmap inside
# the hook, and dispatch each event into iss's input pipeline
# directly — bypassing GLFW for hooked keys. GLFW's key callback
# never fires for them (we suppress at OS level) but that's fine
# because we delivered the event ourselves.
#
# Bare unmodified character keys are NOT intercepted: we let them
# through to GLFW so its on_char path still does layout-correct
# character composition / IME / dead-key handling. Only keys held
# with a modifier (or modifier keys themselves) get the bypass.

# WM / VK / hook constants (from <winuser.h>).
_WH_KEYBOARD_LL = 13
_HC_ACTION = 0
_VK_BACK = 0x08
_VK_TAB = 0x09
_VK_RETURN = 0x0D
_VK_SHIFT = 0x10
_VK_CONTROL = 0x11
_VK_MENU = 0x12        # Alt
_VK_ESCAPE = 0x1B
_VK_SPACE = 0x20
_VK_LEFT = 0x25
_VK_UP = 0x26
_VK_RIGHT = 0x27
_VK_DOWN = 0x28
_VK_DELETE = 0x2E
_VK_LWIN = 0x5B
_VK_RWIN = 0x5C
_VK_LSHIFT = 0xA0
_VK_RSHIFT = 0xA1
_VK_LCONTROL = 0xA2
_VK_RCONTROL = 0xA3
_VK_LMENU = 0xA4
_VK_RMENU = 0xA5
_LLKHF_ALTDOWN = 0x20
_LLKHF_EXTENDED = 0x01
_WM_KEYDOWN = 0x0100
_WM_KEYUP = 0x0101
_WM_SYSKEYDOWN = 0x0104
_WM_SYSKEYUP = 0x0105

# X11 keysyms iss already speaks (matches keymap.py).
_KS_BACKSPACE = 0xff08
_KS_TAB = 0xff09
_KS_RETURN = 0xff0d
_KS_ESCAPE = 0xff1b
_KS_SPACE = 0x0020
_KS_DELETE = 0xffff
_KS_LEFT = 0xff51
_KS_UP = 0xff52
_KS_RIGHT = 0xff53
_KS_DOWN = 0xff54
_KS_SHIFT_L = 0xffe1
_KS_SHIFT_R = 0xffe2
_KS_CONTROL_L = 0xffe3
_KS_CONTROL_R = 0xffe4
_KS_ALT_L = 0xffe9
_KS_ALT_R = 0xffea
_KS_SUPER_L = 0xffeb
_KS_SUPER_R = 0xffec


def _vk_to_keysym(vk: int) -> int:
    """Map a Windows VK code to an X11 keysym iss can put on the wire.
    Returns 0 for keys we don't know how to translate."""
    # ASCII letters and digits — VK matches uppercase ASCII.
    if 0x30 <= vk <= 0x39:                     # '0'..'9'
        return vk
    if 0x41 <= vk <= 0x5A:                     # 'A'..'Z' → lowercase keysym
        return vk + 0x20
    # F1..F24 → X11 keysyms 0xffbe..0xffd5
    if 0x70 <= vk <= 0x87:                     # VK_F1..VK_F24
        return 0xffbe + (vk - 0x70)
    return _VK_KEYSYM.get(vk, 0)


_VK_KEYSYM = {
    _VK_BACK: _KS_BACKSPACE,
    _VK_TAB: _KS_TAB,
    _VK_RETURN: _KS_RETURN,
    _VK_ESCAPE: _KS_ESCAPE,
    _VK_SPACE: _KS_SPACE,
    _VK_DELETE: _KS_DELETE,
    _VK_LEFT: _KS_LEFT,
    _VK_UP: _KS_UP,
    _VK_RIGHT: _KS_RIGHT,
    _VK_DOWN: _KS_DOWN,
    _VK_LSHIFT: _KS_SHIFT_L,
    _VK_RSHIFT: _KS_SHIFT_R,
    _VK_SHIFT: _KS_SHIFT_L,
    _VK_LCONTROL: _KS_CONTROL_L,
    _VK_RCONTROL: _KS_CONTROL_R,
    _VK_CONTROL: _KS_CONTROL_L,
    _VK_LMENU: _KS_ALT_L,
    _VK_RMENU: _KS_ALT_R,
    _VK_MENU: _KS_ALT_L,
    _VK_LWIN: _KS_SUPER_L,
    _VK_RWIN: _KS_SUPER_R,
    # Common US-layout punctuation. Layouts that put these chars on
    # different physical keys still produce the right VK on Windows
    # because Win32 reports "OEM" VKs for the physical-key positions
    # (US layout's 0xBA = ';', etc). The keysym we send is the US
    # base value — host-side layout handles localisation.
    0xBA: ord(';'),   # VK_OEM_1
    0xBB: ord('='),   # VK_OEM_PLUS
    0xBC: ord(','),   # VK_OEM_COMMA
    0xBD: ord('-'),   # VK_OEM_MINUS
    0xBE: ord('.'),   # VK_OEM_PERIOD
    0xBF: ord('/'),   # VK_OEM_2
    0xC0: ord('`'),   # VK_OEM_3
    0xDB: ord('['),   # VK_OEM_4
    0xDC: ord('\\'),  # VK_OEM_5
    0xDD: ord(']'),   # VK_OEM_6
    0xDE: ord("'"),   # VK_OEM_7
    # Numpad digits — share the digit keysyms.
    0x60: ord('0'), 0x61: ord('1'), 0x62: ord('2'), 0x63: ord('3'),
    0x64: ord('4'), 0x65: ord('5'), 0x66: ord('6'), 0x67: ord('7'),
    0x68: ord('8'), 0x69: ord('9'),
    # Numpad arithmetic.
    0x6A: ord('*'), 0x6B: ord('+'), 0x6D: ord('-'),
    0x6E: ord('.'), 0x6F: ord('/'),
}


# Modifiers we intercept at the OS level (suppress + track in our own
# bitmap + deliver to iss). Win/Ctrl/Left-Alt only — these are the
# "command" modifiers that combine with letters to form OS shortcuts
# (Win+L, Ctrl+Alt+Del, Alt+F4) we want to suppress, AND that the
# Mac receives as discrete modifier-down/key/modifier-up sequences.
#
# Shift and AltGr (right-Alt on European layouts) are deliberately
# NOT intercepted: they're "character" modifiers that change which
# glyph a key produces (Shift+a → 'A'; AltGr+e → 'é'). Apple's RFB
# key handler expects the final character keysym, not base-char +
# Shift. We let those flow through to GLFW so its on_char path
# delivers the layout-correct codepoint.
_MODIFIER_VKS = frozenset({
    _VK_LWIN, _VK_RWIN,
    _VK_LCONTROL, _VK_RCONTROL, _VK_CONTROL,
    _VK_LMENU,   # left Alt only; right Alt = AltGr → let through
    _VK_MENU,    # generic Alt VK (rare; conservatively intercept)
})


class _Win32Grab:
    """LL keyboard hook that delivers events to iss directly while iss
    has focus, bypassing GLFW for any key held with a modifier (or any
    modifier key itself). Maintains its own modifier bitmap.

    Args to __init__:
      glfw_window  the GLFW window handle (for foreground-window self-gating)
      key_sender   callable (is_down: bool, x11_keysym: int) -> None
                   that puts the event on the wire. Typically
                   `session.input.key_event` bound from the frontend."""

    def __init__(self, glfw_window, key_sender) -> None:
        import glfw  # type: ignore
        from ctypes import wintypes

        if not callable(key_sender):
            raise TypeError("key_sender must be callable (is_down, keysym)")
        self._send = key_sender

        self._u32 = ctypes.WinDLL("user32", use_last_error=True)

        class _KBDLLHOOKSTRUCT(ctypes.Structure):
            _fields_ = [
                ("vkCode", wintypes.DWORD),
                ("scanCode", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ctypes.c_void_p),
            ]
        self._KBDLL = _KBDLLHOOKSTRUCT

        self._HOOKPROC = ctypes.WINFUNCTYPE(
            # LRESULT = LONG_PTR: pointer-sized on all Windows targets (32 or 64-bit).
            # ctypes.c_long is only 32-bit on Win64, which would leave the upper 32
            # bits of RAX undefined and could cause Windows to misread the return value.
            ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM,
        )
        self._u32.SetWindowsHookExW.argtypes = [
            ctypes.c_int, self._HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD,
        ]
        self._u32.SetWindowsHookExW.restype = wintypes.HHOOK
        self._u32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
        self._u32.UnhookWindowsHookEx.restype = wintypes.BOOL
        self._u32.CallNextHookEx.argtypes = [
            wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM,
        ]
        self._u32.CallNextHookEx.restype = ctypes.c_long
        self._u32.GetForegroundWindow.argtypes = []
        self._u32.GetForegroundWindow.restype = wintypes.HWND

        self._our_hwnd = glfw.get_win32_window(glfw_window)
        if not self._our_hwnd:
            raise RuntimeError("glfw.get_win32_window returned NULL")

        self._hook: Optional[int] = None
        self._cb_ref = None  # keep callback alive
        # Our own modifier bitmap. Maintained from the LL hook events
        # since GetAsyncKeyState won't reflect keys we suppress.
        self._mods_held: set = set()  # set of VK codes currently held
        atexit.register(self._safety_release)

    def enable(self) -> bool:
        if self._hook:
            return True
        u32 = self._u32
        KBDLL = self._KBDLL
        our_hwnd = self._our_hwnd
        send = self._send
        mods_held = self._mods_held

        @self._HOOKPROC
        def _proc(nCode: int, wParam: int, lParam: int) -> int:
            if nCode != _HC_ACTION:
                return u32.CallNextHookEx(None, nCode, wParam, lParam)
            if u32.GetForegroundWindow() != our_hwnd:
                return u32.CallNextHookEx(None, nCode, wParam, lParam)

            kbd = ctypes.cast(lParam, ctypes.POINTER(KBDLL)).contents
            vk = kbd.vkCode
            is_down = wParam in (_WM_KEYDOWN, _WM_SYSKEYDOWN)

            # Modifier key — always intercept, update bitmap, deliver.
            if vk in _MODIFIER_VKS:
                if is_down:
                    mods_held.add(vk)
                else:
                    mods_held.discard(vk)
                ks = _vk_to_keysym(vk)
                if ks:
                    try:
                        send(is_down, ks)
                    except Exception as e:
                        log.debug("kbd-grab modifier send failed: %s", e)
                return 1

            # Non-modifier key WITH a tracked modifier held (Win,
            # Ctrl, left-Alt) → intercept, deliver to iss directly.
            # Catches Win+C, Alt+F4, Copilot Win+C, etc. — keys the
            # OS would otherwise grab.
            #
            # Bare keys, and keys with only Shift / AltGr held, are
            # NOT intercepted: GLFW handles them via on_char which
            # delivers the layout-correct character codepoint.
            if mods_held:
                ks = _vk_to_keysym(vk)
                if ks:
                    try:
                        send(is_down, ks)
                    except Exception as e:
                        log.debug("kbd-grab combo send failed: %s", e)
                # Whether or not we recognised the keysym, eat this
                # event so the OS doesn't fire its own combo.
                return 1

            # Bare character key — let GLFW handle it normally.
            return u32.CallNextHookEx(None, nCode, wParam, lParam)

        self._cb_ref = _proc
        hook = u32.SetWindowsHookExW(_WH_KEYBOARD_LL, _proc, None, 0)
        if not hook:
            err = ctypes.get_last_error()
            log.warning(
                "keyboard grab failed: SetWindowsHookEx returned NULL "
                "(GetLastError=%d)", err,
            )
            self._cb_ref = None
            return False
        self._hook = hook
        log.info(
            "keyboard grab acquired (Win32 LL hook + direct input "
            "delivery; iss-foreground-gated)"
        )
        return True

    def disable(self) -> None:
        if not self._hook:
            return
        self._u32.UnhookWindowsHookEx(self._hook)
        self._hook = None
        # Do NOT clear self._cb_ref here. WH_KEYBOARD_LL carries an OS-enforced
        # ~200 ms removal delay: Windows may still dispatch to the hook proc after
        # UnhookWindowsHookEx returns. Freeing the ctypes callback object now
        # releases the function pointer while it could still be called →
        # STATUS_ACCESS_VIOLATION. The ref is replaced safely in enable() (focus
        # has been lost for at least one focus-in/out cycle, >> 200 ms) or
        # released when this object is finalised.
        # Release any modifier we believe is held but won't get a key-up
        # for (because we're tearing down before the user lets go).
        for vk in list(self._mods_held):
            ks = _vk_to_keysym(vk)
            if ks:
                try:
                    self._send(False, ks)
                except Exception:
                    pass
        self._mods_held.clear()
        log.info("keyboard grab released (Win32)")

    def _safety_release(self) -> None:
        try:
            self.disable()
        except Exception:
            pass


__all__ = ["KeyboardGrab", "make_grab"]
