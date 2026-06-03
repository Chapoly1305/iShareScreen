"""Apple-specific magic byte sequences sent during the RFB handshake.

Reverse-engineered from a live Apple Screen Sharing handshake. Where
we understand the fields well enough to generate the bytes, we do
that — only blobs whose layout isn't fully un-reversed remain as
captured-hex constants. Each constant carries a comment with what we
do and don't know about it, so future maintainers can keep peeling.
"""
from __future__ import annotations

import struct


# 32-byte command-mask appended to the 0x21 ViewerInfo message. Each
# bit gates a viewer-side feature on the daemon (image clipboard,
# clipboard-change push notifications, periodic tickle/heartbeat
# packets, etc.). The bits Apple's shipping Screen Sharing.app sets,
# verified by byte-diffing a real Screen Sharing.app handshake against
# an iss handshake:
#     mask[0]  = 0xb0   (bits 4, 5, 7)
#     mask[2]  = 0x0c   (bits 2, 3)
#     mask[3]  = 0x03   (bits 0, 1)
#     mask[4]  = 0x90   (bits 4, 7)
#     mask[10] = 0x40   (bit 6)
# (Earlier RE notes had these at offsets 2/4/5/6/12, off by 2 — those
# offsets put the bits in the wrong byte and the daemon ignored them,
# which is why iss never received the 2.1 s tickle and clipboard
# pushes.)
def _build_apple_viewer_command_mask() -> bytes:
    m = bytearray(32)
    m[0] = 0xb0
    m[2] = 0x0c
    m[3] = 0x03
    m[4] = 0x90
    m[10] = 0x40
    return bytes(m)


APPLE_VIEWER_COMMAND_MASK: bytes = _build_apple_viewer_command_mask()


# macOS (major, minor, patch) we send in the 0x21 ViewerInfo `os_ver`
# slot regardless of the actual local OS. The host uses this to pick
# the code path for our viewer. (15, 3, 0) corresponds to a stable
# macOS Sequoia release; advertising a slightly older version keeps us
# compatible across the widest range of host macOS versions (older
# hosts won't treat us as "viewer is too new" and downgrade). All
# misc-status broadcast features we depend on are gated by the
# command_mask above, not by os_ver, so this version doesn't restrict
# what the host pushes us.
APPLE_VIEWER_OS_VER: tuple[int, int, int] = (15, 3, 0)


# 12-byte 0x12 follow-up Apple sends in the plaintext phase right
# after the 0x21 ViewerInfo message. State-3 path interprets this
# with cmd=0x100, distinct from the post-toggle cmd=1/2 SetEncryption
# codes. Field values appear load-bearing: an all-zero variant fails
# the handshake.
#
# Layout (12 bytes total):
#     [0]      u8   msg type    = 0x12
#     [1]      u8   pad         = 0
#     [2..3]   u16  field A     = 1   (BE; load-bearing, all-zero fails)
#     [4..5]   u16  field B     = 1   (BE; load-bearing)
#     [6..7]   u16  field C     = 1   (BE; load-bearing)
#     [8..11]  u32  trailer     = 1   (BE)
#
# Building this from struct.pack instead of fromhex makes the
# structure explicit and the magic-number all-1s pattern visible.
APPLE_0X12_FOLLOWUP: bytes = struct.pack(
    ">BBHHHI", 0x12, 0,  1, 1, 1,  1,
)


__all__ = [
    "APPLE_0X12_FOLLOWUP",
    "APPLE_VIEWER_COMMAND_MASK",
    "APPLE_VIEWER_OS_VER",
]
