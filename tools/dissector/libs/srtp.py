"""Self-contained SRTP receiver for Apple Screen Sharing adaptive media.

Apple's AVC suite-5 media streams use ``AES_CM_256_HMAC_SHA1_80`` — a profile
``pylibsrtp``'s high-level API does not expose. Rather than depend on a
low-level libsrtp build, this implements the RFC 3711 receive path directly
on top of ``pycryptodome`` (already a dissector dependency) plus the stdlib
``hmac``/``hashlib``: AES-256 in counter mode for both key derivation and the
payload cipher, HMAC-SHA1-80 for authentication.

The 46-byte ``0x1c`` media key blob is ``32-byte master key || 14-byte master
salt``. Key-derivation labels (RFC 3711 §4.3): 0/1/2 = RTP cipher/auth/salt.

Per-SSRC roll-over counters are tracked independently: Apple emits one SSRC per
tile (4 per quality tier) with independent sequence spaces, so a shared ROC
desynchronises after the first wrap.
"""
from __future__ import annotations

import hmac
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from Crypto.Cipher import AES
from Crypto.Util import Counter

_AUTH_TAG_LEN = 10  # HMAC-SHA1-80 truncated to 80 bits
_RTP_HEADER_MIN = 12

SRTP_KEY_BLOB_LEN = 46
SRTP_MASTER_KEY_LEN = 32
SRTP_MASTER_SALT_LEN = 14


def split_blob(blob: bytes) -> tuple[bytes, bytes]:
    if len(blob) != SRTP_KEY_BLOB_LEN:
        raise ValueError(f"SRTP key blob must be {SRTP_KEY_BLOB_LEN} bytes, got {len(blob)}")
    return blob[:SRTP_MASTER_KEY_LEN], blob[SRTP_MASTER_KEY_LEN:SRTP_KEY_BLOB_LEN]


def srtp_kdf(master_key: bytes, master_salt: bytes, label: int, out_len: int) -> bytes:
    """RFC 3711 §4.3.1 key derivation — AES-CM with the label XORed into the IV."""
    kid = bytearray(14)
    kid[7] = label
    iv0 = bytes(kid[i] ^ master_salt[i] for i in range(14))
    ecb = AES.new(master_key, AES.MODE_ECB)
    out = bytearray()
    counter = 0
    while len(out) < out_len:
        block = bytearray(iv0 + b"\x00\x00")
        c = counter
        for i in range(15, -1, -1):
            if c == 0:
                break
            c += block[i]
            block[i] = c & 0xFF
            c >>= 8
        out += ecb.encrypt(bytes(block))
        counter += 1
    return bytes(out[:out_len])


@dataclass
class _SsrcState:
    roc: int = 0
    max_seq: int = 0
    initialized: bool = False


class SRTPReceiver:
    """SRTP receiver: AES-256-CTR + HMAC-SHA1-80 with per-SSRC ROC tracking."""

    def __init__(self, master_key: bytes, master_salt: bytes) -> None:
        self._cipher_key = srtp_kdf(master_key, master_salt, 0, 32)
        self._auth_key = srtp_kdf(master_key, master_salt, 1, 20)
        self._salt = srtp_kdf(master_key, master_salt, 2, 14)
        self._salt_int = int.from_bytes(self._salt + b"\x00\x00", "big")
        self._states: dict[int, _SsrcState] = {}
        self._counts: defaultdict[int, int] = defaultdict(int)

    @classmethod
    def from_blob(cls, blob: bytes) -> "SRTPReceiver":
        key, salt = split_blob(blob)
        return cls(key, salt)

    @property
    def ssrc_counts(self) -> dict[int, int]:
        return dict(self._counts)

    def decrypt(self, pkt: bytes) -> Optional[tuple[bytes, bytes]]:
        """Authenticate + decrypt one SRTP packet.

        Returns ``(rtp_header, payload)`` or ``None`` if the HMAC tag fails for
        every candidate ROC (i.e. wrong key/direction or not SRTP)."""
        if len(pkt) < _RTP_HEADER_MIN + _AUTH_TAG_LEN:
            return None
        body_len = len(pkt) - _AUTH_TAG_LEN
        seq = (pkt[2] << 8) | pkt[3]
        ssrc = int.from_bytes(pkt[8:12], "big")

        state = self._states.get(ssrc)
        if state is None or not state.initialized:
            roc_guess = 0
        else:
            diff = seq - state.max_seq
            if diff > 0x7FFF:
                roc_guess = max(0, state.roc - 1)
            elif diff < -0x7FFF:
                roc_guess = state.roc + 1
            else:
                roc_guess = state.roc

        candidates: list[int] = []
        seen: set[int] = set()
        for r in (roc_guess, state.roc if state else 0, roc_guess + 1, max(0, roc_guess - 1)):
            if r not in seen:
                seen.add(r)
                candidates.append(r)

        for roc in candidates:
            res = self._try_decrypt(pkt, body_len, seq, ssrc, roc)
            if res is not None:
                self._update_state(ssrc, roc, seq)
                self._counts[ssrc] += 1
                return res
        return None

    def _try_decrypt(self, pkt: bytes, body_len: int, seq: int, ssrc: int, roc: int) -> Optional[tuple[bytes, bytes]]:
        h = hmac.new(self._auth_key, digestmod="sha1")
        h.update(memoryview(pkt)[:body_len])
        h.update(roc.to_bytes(4, "big"))
        if not hmac.compare_digest(h.digest()[:_AUTH_TAG_LEN], pkt[body_len:body_len + _AUTH_TAG_LEN]):
            return None

        first_byte = pkt[0]
        cc = first_byte & 0x0F
        hdr_len = _RTP_HEADER_MIN + cc * 4
        if (first_byte >> 4) & 1:  # extension present
            if hdr_len + 4 > body_len:
                return None
            ext_len = (pkt[hdr_len + 2] << 8) | pkt[hdr_len + 3]
            hdr_len += 4 + ext_len * 4
        if hdr_len > body_len:
            return None

        header = bytes(pkt[:hdr_len])
        if hdr_len == body_len:
            return header, b""

        index = (roc << 16) | seq
        iv_int = self._salt_int ^ (ssrc << 64) ^ (index << 16)
        ctr = Counter.new(128, initial_value=iv_int)
        dec = AES.new(self._cipher_key, AES.MODE_CTR, counter=ctr)
        plaintext = dec.decrypt(pkt[hdr_len:body_len])
        return header, plaintext

    def _update_state(self, ssrc: int, roc: int, seq: int) -> None:
        state = self._states.setdefault(ssrc, _SsrcState())
        if not state.initialized:
            state.roc, state.max_seq, state.initialized = roc, seq, True
            return
        if ((roc << 16) | seq) > ((state.roc << 16) | state.max_seq):
            state.roc, state.max_seq = roc, seq


__all__ = ["SRTPReceiver", "split_blob", "srtp_kdf", "SRTP_KEY_BLOB_LEN"]
