"""HEVC depayloader for Apple Screen Sharing's RTP media (the 'dynamic' path).

Apple deviates from RFC 7798 in its DONL handling. Every payload carries a
2-byte decoding-order number (DONL) right after the (FU) NAL header, and there
is no DOND between aggregated sub-NALUs:

  * Single NAL (outer type 0..47): ``hdr(2) + DONL(2) + nal_data``
  * Aggregation Packet (type 48):  ``hdr(2) + DONL(2) + [size(2)+nal_data]...``
  * Fragmentation Unit (type 49):  ``hdr(2) + fu_hdr(1) + DONL(2) + frag_data``
    — the DONL is repeated in *every* fragment, not just the start.

A single decoded picture is spread across several SSRCs (one per tile); the
global decode order is recovered by sorting NALUs by their (roll-over-unwrapped)
DONL.
"""
from __future__ import annotations

from typing import Iterable, Optional

NAL_VPS = 32
NAL_SPS = 33
NAL_PPS = 34
NAL_AGGREGATION = 48
NAL_FRAGMENTATION = 49
IDR_RANGE = range(16, 22)  # BLA_W_LP (16) .. CRA_NUT (21)

NAL_NAMES = {
    0: "TRAIL_N", 1: "TRAIL_R", 2: "TSA_N", 3: "TSA_R", 4: "STSA_N", 5: "STSA_R",
    6: "RADL_N", 7: "RADL_R", 8: "RASL_N", 9: "RASL_R",
    16: "BLA_W_LP", 17: "BLA_W_RADL", 18: "BLA_N_LP", 19: "IDR_W_RADL",
    20: "IDR_N_LP", 21: "CRA_NUT",
    32: "VPS", 33: "SPS", 34: "PPS", 35: "AUD", 36: "EOS", 37: "EOB",
    39: "SEI_PREFIX", 40: "SEI_SUFFIX", 48: "AP", 49: "FU",
}


def nal_name(t: int) -> str:
    return NAL_NAMES.get(t, f"type{t}")


def classify_payload(pay: bytes) -> Optional[dict]:
    """Classify one decrypted RTP payload without reassembling.

    Returns a dict describing the outer NAL, its DONL, and the HEVC NAL types
    it carries (sub-NALUs for an AP, the inner type for an FU start, the type
    itself for a single NAL). FU continuation/end fragments contribute no
    contained type. Returns ``None`` for runt payloads."""
    if len(pay) < 2:
        return None
    outer = (pay[0] >> 1) & 0x3F
    info: dict = {"outer_type": outer, "outer_name": nal_name(outer)}

    if outer == NAL_AGGREGATION:
        if len(pay) < 4:
            return info
        info["donl"] = (pay[2] << 8) | pay[3]
        pos, n, sub = 4, len(pay), []
        while pos + 2 <= n:
            size = (pay[pos] << 8) | pay[pos + 1]
            pos += 2
            if size == 0 or pos + size > n:
                break
            sub.append((pay[pos] >> 1) & 0x3F)
            pos += size
        info["contained"] = sub
        return info

    if outer == NAL_FRAGMENTATION:
        if len(pay) < 6:
            return info
        fu = pay[2]
        info["donl"] = (pay[3] << 8) | pay[4]
        info["fu_start"] = bool(fu & 0x80)
        info["fu_end"] = bool(fu & 0x40)
        info["fu_type"] = fu & 0x3F
        info["contained"] = [fu & 0x3F] if (fu & 0x80) else []
        return info

    # single NAL
    if len(pay) >= 4:
        info["donl"] = (pay[2] << 8) | pay[3]
    info["contained"] = [outer]
    return info


def reassemble_group(payloads: Iterable[bytes]) -> list[bytes]:
    """Turn one SSRC's decrypted RTP payloads (in arrival/seq order) into clean
    Annex-B-less NALUs. Handles AP, FU, and single-NAL packets with Apple's
    DONL conventions. Malformed entries are dropped (UDP loss makes that
    routine)."""
    out: list[bytes] = []
    fu_buf = bytearray()
    fu_active = False

    for pay in payloads:
        if len(pay) < 2:
            continue
        outer = (pay[0] >> 1) & 0x3F

        if outer == NAL_AGGREGATION:
            pos, n = 4, len(pay)
            while pos + 2 <= n:
                size = (pay[pos] << 8) | pay[pos + 1]
                pos += 2
                if size == 0 or pos + size > n:
                    break
                out.append(bytes(pay[pos:pos + size]))
                pos += size

        elif outer == NAL_FRAGMENTATION:
            if len(pay) < 6:
                continue
            fu = pay[2]
            inner = fu & 0x3F
            if fu & 0x80:  # start
                hdr0 = (pay[0] & 0x81) | (inner << 1)
                fu_buf = bytearray([hdr0, pay[1]]) + pay[5:]
                fu_active = True
            elif fu_active:
                fu_buf += pay[5:]
                if fu & 0x40:  # end
                    out.append(bytes(fu_buf))
                    fu_active = False

        else:  # single NAL with leading 2-byte DONL
            if len(pay) < 4:
                continue
            out.append(bytes(pay[:2]) + bytes(pay[4:]))

    return out


def unwrap_dons(dons: list[int]) -> list[int]:
    """Unwrap a sequence of 16-bit DONLs (arrival order) into a monotonic
    counter so NALUs from different SSRCs can be globally decode-ordered."""
    out: list[int] = []
    prev: Optional[int] = None
    base = 0
    for d in dons:
        if prev is not None and d < prev - 32768:
            base += 65536
        prev = d
        out.append(base + d)
    return out


__all__ = [
    "classify_payload", "reassemble_group", "unwrap_dons", "nal_name",
    "NAL_NAMES", "IDR_RANGE", "NAL_AGGREGATION", "NAL_FRAGMENTATION",
    "NAL_VPS", "NAL_SPS", "NAL_PPS",
]
