"""HEVC NAL reassembly from Apple's RTP payload format.

Apple deviates from RFC 7798 in two places:
  - Aggregation Packets (type 48) have a 2-byte DONL once after the NAL
    header, with **no** DOND between sub-NALUs.
  - Fragmentation Units (type 49) carry a 2-byte DONL **inside every
    fragment**, not just the start fragment.
  - Single NALUs ride with a 2-byte DONL prefix that must be stripped
    before handing the NALU to a decoder.
"""
from __future__ import annotations

import struct
from typing import Iterable, Optional


# HEVC NAL unit types we care about.
NAL_VPS = 32
NAL_SPS = 33
NAL_PPS = 34
NAL_AGGREGATION = 48
NAL_FRAGMENTATION = 49

# IDR / IRAP NAL unit types: BLA_W_LP (16) through CRA_NUT (21).
IDR_RANGE = range(16, 22)


def reassemble_group(payloads: Iterable[bytes]) -> list[bytes]:
    """Turn the RTP payloads belonging to one timestamp group into clean
    NALUs. Handles AP, FU, and single-NAL packets with Apple's DONL
    conventions. Drops malformed entries silently — UDP loss makes that
    routine, and the decoder gets to error on what survives."""
    out: list[bytes] = []
    fu_buf = bytearray()
    fu_active = False

    for pay in payloads:
        if len(pay) < 2:
            continue
        nt = (pay[0] >> 1) & 0x3F

        if nt == NAL_AGGREGATION:
            # header(2) + DONL(2) + [size(2) + data]...
            pos = 4
            n = len(pay)
            while pos + 2 <= n:
                size = struct.unpack(">H", pay[pos:pos + 2])[0]
                pos += 2
                if size == 0 or pos + size > n:
                    break
                out.append(bytes(pay[pos:pos + size]))
                pos += size

        elif nt == NAL_FRAGMENTATION:
            # header(2) + FU_hdr(1) + DONL(2) + payload
            if len(pay) < 6:
                continue
            fu_hdr = pay[2]
            start = bool(fu_hdr & 0x80)
            end = bool(fu_hdr & 0x40)
            inner_type = fu_hdr & 0x3F
            if start:
                # Reconstruct the inner NAL header from the FU NAL header.
                hdr0 = (pay[0] & 0x81) | (inner_type << 1)
                fu_buf = bytearray([hdr0, pay[1]])
                fu_buf += pay[5:]
                fu_active = True
            elif fu_active:
                fu_buf += pay[5:]
                if end:
                    out.append(bytes(fu_buf))
                    fu_active = False

        else:
            # Single NAL with leading 2-byte DONL.
            if len(pay) < 4:
                continue
            out.append(bytes(pay[:2]) + bytes(pay[4:]))

    return out


# ── H.264 / AVC (RFC 6184 + Apple DON) ────────────────────────────────
#
# When the client offers the AVC codec bank (RTP PT 123, see offers.py) the
# host streams H.264 instead of HEVC. The RTP framing differs from the HEVC
# variant above:
#   - NAL header is **1 byte** (type = byte & 0x1F), not 2.
#   - Parameter sets are NOT sent as Annex-B SPS/PPS NALs. They arrive once,
#     up front, as an Apple-wrapped `avc1` sample description embedding an
#     `avcC` box (the MP4 AVCDecoderConfigurationRecord). That packet has the
#     H.264 forbidden_zero_bit set in byte 0 (0x80), so it can never be
#     mistaken for a slice NAL — `is_avc_config()` keys off exactly that.
#   - Slices ride in FU-B (type 29) units carrying a 2-byte DON after the FU
#     header (the H.264 analogue of HEVC's DONL); FU-A (28, no DON), STAP
#     aggregation, and single-NAL packets are handled too for completeness.
# (Confirmed live 2026-06 against a real host: H.264 High@L4.2, 4:2:0.)

H264_NAL_IDR = 5
H264_NAL_SPS = 7
H264_NAL_PPS = 8
H264_NAL_STAP_A = 24
H264_NAL_STAP_B = 25
H264_NAL_FU_A = 28
H264_NAL_FU_B = 29

# H.264 IDR is a single NAL type (5), unlike HEVC's IRAP range.
H264_IDR_RANGE = range(5, 6)


def is_avc_config(payload: bytes) -> bool:
    """True if `payload` is Apple's `avc1`/`avcC` config wrapper rather than a
    slice NAL. The wrapper sets the H.264 forbidden_zero_bit (byte0 & 0x80),
    which is illegal in a real NAL header, so the test is unambiguous."""
    return bool(payload) and (payload[0] & 0x80) != 0


def parse_avc_config(payload: bytes) -> tuple[Optional[bytes], list[bytes]]:
    """Extract (SPS, [PPS, …]) from Apple's `avc1`/`avcC` config packet.

    Layout: `92 e6 c0 a3 | u32 box_size | "avc1" VisualSampleEntry … | avcC
    box`. We don't parse the whole sample entry — we locate the `avcC`
    AVCDecoderConfigurationRecord and read its SPS/PPS arrays. Each returned
    NAL keeps its 1-byte header (the form libavcodec wants in extradata).
    Returns (None, []) if no parseable avcC is found."""
    i = payload.find(b"avcC")
    if i < 0:
        return None, []
    rec = payload[i + 4:]                       # AVCDecoderConfigurationRecord
    if len(rec) < 7 or rec[0] != 1:             # configurationVersion == 1
        return None, []
    try:
        num_sps = rec[5] & 0x1F
        off = 6
        sps: Optional[bytes] = None
        for _ in range(num_sps):
            ln = struct.unpack(">H", rec[off:off + 2])[0]
            off += 2
            nal = bytes(rec[off:off + ln])
            off += ln
            if sps is None and nal:
                sps = nal                       # take the first SPS
        num_pps = rec[off]
        off += 1
        ppss: list[bytes] = []
        for _ in range(num_pps):
            ln = struct.unpack(">H", rec[off:off + 2])[0]
            off += 2
            ppss.append(bytes(rec[off:off + ln]))
            off += ln
    except (IndexError, struct.error):
        return None, []
    return sps, ppss


def reassemble_group_h264(payloads: Iterable[bytes]) -> list[bytes]:
    """H.264 sibling of `reassemble_group`. Turns one timestamp group's RTP
    payloads into clean (start-code-free) H.264 NAL units, honouring Apple's
    FU-B DON. Config wrappers (`is_avc_config`) are skipped — the caller
    extracts SPS/PPS from those separately. Malformed entries are dropped."""
    out: list[bytes] = []
    fu_buf = bytearray()
    fu_active = False

    for pay in payloads:
        if not pay or is_avc_config(pay):
            continue
        nt = pay[0] & 0x1F

        if nt in (H264_NAL_FU_A, H264_NAL_FU_B):
            # FU-A: ind(1) + fu_hdr(1) + payload
            # FU-B: ind(1) + fu_hdr(1) + DON(2) + payload
            data_off = 4 if nt == H264_NAL_FU_B else 2
            if len(pay) <= data_off:
                continue
            fu_hdr = pay[1]
            start = bool(fu_hdr & 0x80)
            end = bool(fu_hdr & 0x40)
            inner_type = fu_hdr & 0x1F
            if start:
                # Reconstruct the original 1-byte NAL header: F/NRI from the
                # FU indicator, type from the FU header.
                hdr0 = (pay[0] & 0xE0) | inner_type
                fu_buf = bytearray([hdr0])
                fu_buf += pay[data_off:]
                fu_active = True
            elif fu_active:
                fu_buf += pay[data_off:]
                if end:
                    out.append(bytes(fu_buf))
                    fu_active = False

        elif nt in (H264_NAL_STAP_A, H264_NAL_STAP_B):
            # STAP-A: ind(1) + [size(2) + NAL]…
            # STAP-B: ind(1) + DON(2) + [size(2) + NAL]…
            pos = 3 if nt == H264_NAL_STAP_B else 1
            n = len(pay)
            while pos + 2 <= n:
                size = struct.unpack(">H", pay[pos:pos + 2])[0]
                pos += 2
                if size == 0 or pos + size > n:
                    break
                out.append(bytes(pay[pos:pos + size]))
                pos += size

        else:
            # Single NAL. Apple prefixes a 2-byte DON after the 1-byte header
            # (the H.264 analogue of HEVC's single-NAL DONL); strip it.
            if len(pay) >= 4:
                out.append(bytes(pay[:1]) + bytes(pay[3:]))
            else:
                out.append(bytes(pay))

    return out


__all__ = [
    "IDR_RANGE",
    "NAL_AGGREGATION",
    "NAL_FRAGMENTATION",
    "NAL_PPS",
    "NAL_SPS",
    "NAL_VPS",
    "reassemble_group",
    "H264_IDR_RANGE",
    "H264_NAL_IDR",
    "H264_NAL_PPS",
    "H264_NAL_SPS",
    "is_avc_config",
    "parse_avc_config",
    "reassemble_group_h264",
]
