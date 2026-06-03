"""Apple RFB clipboard extension wire protocol.

Reverse-engineered from screensharingd 15.3. Two control messages and one
server-push message implement bidirectional NSPasteboard sync over the
encrypted RFB control channel:

  msg 0x15  RFBAutoPasteboardMessage      client→server  (8 B)
            Body byte 2 = mode (2 = enable rich pasteboard sync).
            Must be sent once per session before the server will respond
            to clipboard requests.

  msg 0x0b  HandleViewerClipboardRequest  client→server  (9 B)
            Body byte 1 bit 0 = "promises only" flag.
              1 → server replies with metadata only — call this a "poll".
              0 → server replies with full data — call this a "fetch".
            The remaining 7 body bytes are unused (observed all-zero).

  msg 0x1f  HandleViewerClipboardSend     server→client  (variable)
            Header layout (16 bytes):
              [0]      0x1f
              [1]      0x00
              [2]      promise byte (echoed from request)
              [3]      0x00
              [4..8]   u32 BE  reserved (observed zero)
              [8..12]  u32 BE  uncompressed size
              [12..16] u32 BE  compressed size
              [16..16+compressed_size] zlib-deflate stream

            Multi-frame transport: a single logical 0x1f msg can span
            many encrypted RFB cipher frames. Only the first frame carries
            the 16-byte header — subsequent frames are pure payload
            continuation. Reassemble until 16 + compressed_size bytes
            accumulated.

            Apple uses Z_SYNC_FLUSH framing — `zlib.decompress(d, 15)`
            rejects this as "incomplete or truncated stream"; route
            through a `decompressobj` and call `.flush()` to get the
            final output.

Decompressed inner payload format:

    u32 BE   item_count
    per item:
        u32 BE name_len, bytes(primary_uti)        # e.g. "public.utf8-plain-text"
        u32 BE reserved (observed 0)
        u32 BE alias_count
        per alias:
            u32 BE name_len, bytes(alias_uti)
            u32 BE val_len,  bytes(alias_value)
        u32 BE primary_data_len, bytes(primary_data)

For text content `primary_uti` is "public.utf8-plain-text" and
`primary_data` is the UTF-8 bytes.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


MSG_AUTO_PASTEBOARD = 0x15
MSG_CLIPBOARD_REQUEST = 0x0b
MSG_CLIPBOARD_SEND = 0x1f
AUTO_PASTEBOARD_MODE_DEFAULT = 1   # see build_auto_pasteboard_msg docstring


@dataclass
class ClipboardItem:
    primary_uti: str
    primary_data: bytes
    aliases: List[Tuple[str, bytes]] = field(default_factory=list)


def build_auto_pasteboard_msg(mode: int = AUTO_PASTEBOARD_MODE_DEFAULT) -> bytes:
    """msg 0x15 (8 B) — sent once per session to enable clipboard sync.

    The agent's pasteboard handler only branches on cmd==1: that's
    "start monitoring pasteboard" — the agent then watches the local
    NSPasteboard and emits a change notification to the daemon on any
    generation bump. Any other cmd value gets logged as "unknown command"
    and silently ignored, so mode=1 is the only value that unlocks
    proactive change notifications."""
    return b"\x15\x00" + struct.pack(">H", mode) + b"\x00\x00\x00\x00"


def build_clipboard_request(promise_only: bool = False) -> bytes:
    """msg 0x0b (8 B) — fetch (promise_only=False) or poll (True).

    The body is exactly 8 bytes including the type byte. Sending 9 leaves
    one stray byte in the daemon's input buffer, which it reads as
    msg_type=0 (SetPixelFormat) on the next iteration — every subsequent
    0x0b is then silently dropped, so only the prime fetch ever reaches
    the clipboard handler."""
    return b"\x0b" + bytes([1 if promise_only else 0]) + b"\x00" * 6


def build_clipboard_send(text: str) -> bytes:
    """msg 0x1f (rich-format outbound) carrying a single utf-8 text item.

    Why not the standard RFB msg 0x06 ClientCutText? Because Apple's
    screensharingd's HandleViewerCutTextMessage path writes to the
    legacy Pasteboard Manager scrap only — modern Mac apps that read
    `public.utf8-plain-text` from NSPasteboard see nothing, and
    pasting silently fails. Routing through HandleViewerClipboardSend
    (msg 0x1f) — the same msg the daemon uses inbound for rich
    pasteboard pushes — gets the text into NSPasteboard with the
    proper utf-8 flavor, which all modern apps recognize.

    Wire format mirrors the inbound 0x1f exactly: 16-byte header +
    zlib-deflate stream of an inner archive with one item.
    """
    text_bytes = text.encode("utf-8")
    uti = b"public.utf8-plain-text"
    # Inner archive: u32 item_count, then per-item:
    #   u32 name_len, name bytes, u32 reserved=0, u32 alias_count=0,
    #   u32 primary_data_len, primary_data bytes
    inner = (
        struct.pack(">I", 1)
        + struct.pack(">I", len(uti)) + uti
        + struct.pack(">I", 0)               # reserved
        + struct.pack(">I", 0)               # alias_count
        + struct.pack(">I", len(text_bytes)) + text_bytes
    )
    # Z_SYNC_FLUSH framing matches what the daemon emits inbound, which
    # makes parser symmetry trivial.
    co = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, 15)
    compressed = co.compress(inner) + co.flush(zlib.Z_SYNC_FLUSH)
    header = (
        struct.pack(">B", MSG_CLIPBOARD_SEND)  # 0x1f
        + b"\x00"                              # pad
        + b"\x00"                              # promise (full data, not promise-only)
        + b"\x00"                              # pad
        + struct.pack(">I", 0)                 # reserved
        + struct.pack(">I", len(inner))        # uncompressed_size
        + struct.pack(">I", len(compressed))   # compressed_size
    )
    return header + compressed


def parse_clipboard_send_header(data: bytes) -> Optional[Tuple[int, int, int, int]]:
    """Parse the 16-byte msg 0x1f header. Returns
    (promise_flag, reserved, uncompressed_size, compressed_size) or None
    if the buffer is too short / not a 0x1f message."""
    if len(data) < 16 or data[0] != MSG_CLIPBOARD_SEND:
        return None
    promise = data[2]
    reserved = struct.unpack(">I", data[4:8])[0]
    uncompressed = struct.unpack(">I", data[8:12])[0]
    compressed = struct.unpack(">I", data[12:16])[0]
    return promise, reserved, uncompressed, compressed


def decompress_clipboard_payload(payload: bytes) -> bytes:
    """Decompress the deflate stream that follows the 16-byte header.
    Apple uses Z_SYNC_FLUSH so the stream lacks a final end-of-stream
    marker — route through a `decompressobj` and flush."""
    obj = zlib.decompressobj(15)
    out = obj.decompress(payload)
    out += obj.flush()
    return out


def parse_clipboard_items(decompressed: bytes) -> List[ClipboardItem]:
    """Decode the inner NSPasteboard archive. See module docstring."""
    p = 0

    def u32() -> int:
        nonlocal p
        v = struct.unpack(">I", decompressed[p:p + 4])[0]
        p += 4
        return v

    def lp_bytes() -> bytes:
        nonlocal p
        n = u32()
        b = decompressed[p:p + n]
        p += n
        return b

    items: List[ClipboardItem] = []
    item_count = u32()
    for _ in range(item_count):
        primary_uti = lp_bytes().decode("utf-8", errors="replace")
        _reserved = u32()
        alias_count = u32()
        aliases: List[Tuple[str, bytes]] = []
        for _ in range(alias_count):
            an = lp_bytes().decode("utf-8", errors="replace")
            av = lp_bytes()
            aliases.append((an, av))
        primary_data = lp_bytes()
        items.append(ClipboardItem(primary_uti, primary_data, aliases))
    return items


class ClipboardReassembler:
    """State machine for stitching multi-cipher-frame 0x1f sends.

    Feed every inbound plaintext message via `feed(msg)`. Returns the
    fully assembled 0x1f bytes (header + compressed payload) once
    complete, else None. The caller is responsible for routing only
    appropriate frames to this reassembler — see `in_progress`."""

    def __init__(self) -> None:
        self._buf = bytearray()
        self._expected = 0
        self._frames = 0
        self._last_frame_count = 0

    @property
    def in_progress(self) -> bool:
        return bool(self._buf)

    @property
    def last_frame_count(self) -> int:
        """Number of cipher frames consumed by the most recently completed
        0x1f. Useful for callers compensating any daemon-side counter drift
        triggered by the multi-frame send."""
        return self._last_frame_count

    def feed(self, msg: bytes) -> Optional[bytes]:
        if not self._buf:
            if not msg or msg[0] != MSG_CLIPBOARD_SEND:
                return None
            hdr = parse_clipboard_send_header(msg)
            if hdr is None:
                return None
            _, _, _, compressed = hdr
            self._expected = 16 + compressed
            self._buf.extend(msg)
            self._frames = 1
        else:
            self._buf.extend(msg)
            self._frames += 1
        if len(self._buf) >= self._expected:
            full = bytes(self._buf[:self._expected])
            self._buf.clear()
            self._expected = 0
            self._last_frame_count = self._frames
            self._frames = 0
            return full
        return None


def text_from_items(items: List[ClipboardItem]) -> Optional[str]:
    """Pick the first text-flavored item, return its UTF-8 string. Returns
    None if no text flavour was advertised by the source app.

    UTI fallback order:
      1. `public.utf8-plain-text` (Apple's preferred text flavor)
      2. any `public.*text*` UTI
      3. `com.apple.traditional-mac-plain-text` (legacy/latin-1 — what
         shows up after iss-side ClientCutText round-trips, since
         msg 0x06 only carries latin-1 bytes by Apple's RFB convention)
      4. `public.utf16*` (decode as UTF-16-LE / UTF-16-BE)
    """
    for it in items:
        if it.primary_uti == "public.utf8-plain-text":
            return it.primary_data.decode("utf-8", errors="replace")
    for it in items:
        if it.primary_uti.startswith("public.") and "text" in it.primary_uti:
            if "utf16" in it.primary_uti:
                # The 'external' variant is BE; the bare one is platform-
                # native LE on macOS.
                enc = "utf-16-be" if "external" in it.primary_uti else "utf-16-le"
                return it.primary_data.decode(enc, errors="replace")
            return it.primary_data.decode("utf-8", errors="replace")
    for it in items:
        if it.primary_uti == "com.apple.traditional-mac-plain-text":
            return it.primary_data.decode("latin-1", errors="replace")
    return None


__all__ = [
    "AUTO_PASTEBOARD_MODE_DEFAULT",
    "ClipboardItem",
    "ClipboardReassembler",
    "MSG_AUTO_PASTEBOARD",
    "MSG_CLIPBOARD_REQUEST",
    "MSG_CLIPBOARD_SEND",
    "build_auto_pasteboard_msg",
    "build_clipboard_request",
    "decompress_clipboard_payload",
    "parse_clipboard_items",
    "parse_clipboard_send_header",
    "text_from_items",
]
