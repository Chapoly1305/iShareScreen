"""InputController — sends user input back to the host over the RFB control sock.

Frontends instantiate one and route their UI events through it. The
controller serialises wire events with the active stream cipher and
ships them over the shared TCP socket.

Coordinates arrive in the source-stream pixel space
(0..server_w, 0..server_h); no scaling is performed here.

All mouse events are wrapped in msg 0x10
HandleEncryptedEventMessage — Apple's stock client sends zero
msg 0x05 PointerEvents in either cmd=1 (share-console) or cmd=2
(alt-user) HP modes; every pointer event is msg 0x10. The msg 0x05
path still works for our uid on cmd=1 but bypasses cursor-tracker
side-effects that the msg 0x10 path triggers, hurting cursor-shape
responsiveness. Keyboard input still uses msg 0x04 KeyEvent.
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
from typing import Optional

from Crypto.Cipher import AES

from .protocol.clipboard import (
    build_auto_pasteboard_msg,
    build_clipboard_request,
    build_clipboard_send,
)
from .protocol.enc1103 import StreamCipher
from .protocol.rfb import (
    BTN_SCROLL_DOWN,
    BTN_SCROLL_UP,
    build_key_event,
    build_pointer_event,
)


log = logging.getLogger(__name__)


def _build_msg10_pointer(
    cbc_key: bytes, buttons: int, x: int, y: int,
) -> bytes:
    """Build an 18-byte msg 0x10 HandleEncryptedEventMessage wrapping
    one pointer event. Reverse-engineered from screensharingd
    HandleEncryptedEventMessage.

    Wire format (18 bytes):
        [0]        0x10                msg type
        [1]        pad
        [2..17]    16B AES-128-ECB ciphertext

    Plaintext (16 bytes), encrypted with cbc_key (the post-toggle
    cryptor that replaced the SRP-K-derived ecb_key inside
    HandleSetEncryptionMessage when iss sent msg 0x12):
        [0..9]     pad/timing — 10 bytes (daemon byte-swaps as 2×u32
                                but doesn't enforce values)
        [10]       0xff sentinel  (REQUIRED — the daemon checks this
                                byte and drops the event if not 0xff)
        [11]       button mask    (Apple-style: bit0=L, bit1=R, bit2=M)
        [12..13]   u16 BE x       (in server-screen pixels)
        [14..15]   u16 BE y
    """
    plaintext = (
        b"\x00" * 10
        + b"\xff"
        + bytes([buttons & 0xff])
        + struct.pack(">HH", x & 0xffff, y & 0xffff)
    )
    assert len(plaintext) == 16, len(plaintext)
    aes = AES.new(cbc_key[:16], AES.MODE_ECB)
    ct = aes.encrypt(plaintext)
    return bytes([0x10, 0x00]) + ct


class InputController:
    """Thread-safe wrapper around the RFB control socket.

    All three event methods are non-blocking from the caller's perspective and
    swallow per-call socket errors — a stale socket during hot reconnect must
    not propagate up into the render loop. Resolution is only used to clamp
    out-of-range pointer coordinates; accurate clamping prevents wrap-around
    bugs in the macOS pointer handler.
    """

    def __init__(
        self,
        sock: socket.socket,
        cipher: Optional[StreamCipher],
        *,
        server_width: int,
        server_height: int,
        alt_session: bool = False,
    ) -> None:
        self._sock = sock
        self._cipher = cipher
        self._w = server_width
        self._h = server_height
        self._alt_session = alt_session
        self._lock = threading.Lock()
        self._closed = False
        # Cumulative TX packet counter; Session reads it for the
        # periodic profile log so the operator can confirm iss is
        # actually sending input (mouse/key/clipboard) to the host.
        self.tx_pkts = 0

    def close(self) -> None:
        with self._lock:
            self._closed = True

    # ── public event API ──────────────────────────────────────────────

    def pointer_event(self, buttons: int, x: int, y: int) -> None:
        cx = max(0, min(self._w - 1, int(x)))
        cy = max(0, min(self._h - 1, int(y)))
        if self._cipher is not None:
            # Apple's stock HP client (both cmd=1 console-share and
            # cmd=2 alt-user) wraps every pointer event in msg 0x10
            # HandleEncryptedEventMessage rather than the standard
            # msg 0x05 PointerEvent. The msg 0x05 path may still work
            # for our uid (it drives the host fine) but bypasses
            # cursor-tracker side-effects that fire on the msg 0x10
            # path; matching the wrapper gets us cursor-shape
            # responsiveness parity.
            msg = _build_msg10_pointer(
                self._cipher.cbc_key, buttons, cx, cy,
            )
        else:
            msg = build_pointer_event(buttons=buttons, x=cx, y=cy)
        self._send(msg)

    def scroll_event(self, x: int, y: int, dx: int, dy: int) -> None:
        """Apple emulates a wheel via pointer events with bits 3/4 of buttons.

        Each `(press, release)` pair = one wheel tick. `abs(dy)` is the number
        of ticks; the sign picks up vs down. The host's HID handler edge-
        triggers, so back-to-back press/release pairs at line rate are fine —
        Mac UI elements (Safari, Finder) eat them at >100 Hz.

        Horizontal scroll isn't natively supported in RFB; we ignore dx."""
        if dy == 0:
            return
        bit = BTN_SCROLL_UP if dy < 0 else BTN_SCROLL_DOWN
        for _ in range(abs(int(dy))):
            self.pointer_event(bit, x, y)
            self.pointer_event(0, x, y)

    def key_event(self, down: bool, keysym: int) -> None:
        if not keysym:
            return
        self._send(build_key_event(down=bool(down), keysym=int(keysym)))

    def request_framebuffer_update(
        self, *, incremental: bool = True,
        x: int = 0, y: int = 0, w: int = 1, h: int = 1,
    ) -> None:
        """msg 0x03 FramebufferUpdateRequest — used as the cursor-pipeline
        keepalive. The daemon re-arms its cursor (enc 1104) sender after each
        rect it sends and needs a fresh request to keep going; SS.app polls
        continuously so its cursor never freezes, iss historically requested
        once at startup and stopped. We poll a MINIMAL 1x1 region: the cursor
        pseudo-encoding is region-independent (sent whenever the cursor
        changes against any outstanding request), so a 1x1 request re-arms it
        WITHOUT making the daemon answer with full-screen video rects — a
        full-screen request does pull video over the RFB channel and
        destabilises the HP RTP stream (why the earlier full-screen attempt
        was reverted)."""
        self._send(struct.pack(
            ">BBHHHH", 0x03, 1 if incremental else 0, x, y, w, h,
        ))

    def send_clipboard_enable(self) -> None:
        """msg 0x15 — ask screensharingd to enable autopasteboard. Mode 1 =
        "start monitoring local pasteboard" per ScreensharingAgent's
        agent_SSAgent_AutoPasteboardCommand_rpc handler. Mode 2 gets
        rejected by the agent ("unknown command") so monitoring never
        starts and proactive change notifications never fire."""
        self._send(build_auto_pasteboard_msg(mode=1))

    def send_clipboard_fetch(self) -> None:
        """msg 0x0b — request the remote clipboard contents in full."""
        self._send(build_clipboard_request(promise_only=False))

    def cut_text(self, text: str) -> None:
        """Send local clipboard text to the host so it lands on the Mac's
        NSPasteboard with a `public.utf8-plain-text` flavor.

        Uses Apple's rich-format msg 0x1f path (HandleViewerClipboardSend)
        rather than RFB's standard msg 0x06 ClientCutText. Apple's daemon
        routes msg 0x06 to the legacy Pasteboard Manager scrap only —
        modern Mac apps reading NSPasteboard for `public.utf8-plain-text`
        find nothing and silently fail to paste. msg 0x1f writes the
        proper utf-8 flavor so paste actually works."""
        if not text:
            return
        self._send(build_clipboard_send(text))

    # ── internals ─────────────────────────────────────────────────────

    def _send(self, payload: bytes) -> None:
        with self._lock:
            if self._closed:
                return
            import os as _os
            if _os.environ.get("ISS_LOG_RFB_OUT") == "1":
                ctr = getattr(self._cipher, "_enc_ctr", None)
                log.info("TX type=0x%02x len=%d ctr=%s body=%s",
                         payload[0] if payload else 0, len(payload), ctr,
                         payload.hex())
            try:
                self._sock.sendall(self._cipher.encrypt_message(payload) if self._cipher else payload)
                self.tx_pkts += 1
            except OSError as e:
                log.debug("input send dropped: %s", e)
