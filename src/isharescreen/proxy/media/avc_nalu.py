"""H.264 (AVC) NAL reassembly + decoder-config parse for Apple's RTP format.

Apple's server sends H.264 4:2:0 when the client advertises the field1=123
codec bank (see offers.py). Packetisation is RFC 6184 and — verified against
live captures — the screen stream is DON-LESS:

  - FU-A (type 28): indicator(1) + FU-header(1) + payload (no DON). This is how
    the IDR + P-slices actually arrive; it's the bulk of the packets.
  - Single NAL (types 1-23): the slice header follows the 1-byte NAL header
    directly, NO DON (small slices that fit in a single packet). Stripping a
    2-byte DON here corrupted those slices — see reassemble_h264().
  - FU-B (type 29): indicator(1) + FU-header(1) + DON(2) + payload, and STAP-A
    (type 24): indicator(1) + DON(2) + [size(2)+NAL]... — handled defensively;
    not observed in the screen stream.
  - Decoder config: NOT Annex-B SPS/PPS NALs. It arrives as an Apple-wrapped
    packet (first byte 0x92) carrying an ISO-BMFF `avc1` sample entry whose
    `avcC` box holds the SPS/PPS. parse_avc_config() pulls them out.
"""
from __future__ import annotations

import struct
from typing import Iterable, Optional

# H.264 NAL unit types (RFC 6184 / ITU-T H.264 Table 7-1).
NAL_SLICE_NONIDR = 1
NAL_SLICE_IDR = 5
NAL_SPS = 7
NAL_PPS = 8
NAL_STAP_A = 24
NAL_FU_A = 28
NAL_FU_B = 29

# Apple config packet: not a valid NAL header (forbidden_zero_bit set).
APPLE_CONFIG_MARKER = 0x92


def h264_nal_type(byte0: int) -> int:
    """NAL unit type from a 1-byte H.264 NAL header."""
    return byte0 & 0x1F


def reassemble_h264(payloads: Iterable[bytes]) -> list[bytes]:
    """Turn the RTP payloads of one timestamp group into clean Annex-B-less
    NALUs (raw NAL bytes, no start code). Handles FU-A/FU-B, STAP-A and single
    NALs with Apple's DON conventions. Drops malformed entries silently — UDP
    loss makes partial groups routine and the decoder errors on what survives.
    Config packets (0x92) are not NALs and are skipped here; use
    parse_avc_config()."""
    out: list[bytes] = []
    fu_buf = bytearray()
    fu_active = False

    for pay in payloads:
        if not pay:
            continue
        b0 = pay[0]
        if b0 == APPLE_CONFIG_MARKER:
            continue
        t = b0 & 0x1F

        if t in (NAL_FU_A, NAL_FU_B):
            don = 2 if t == NAL_FU_B else 0
            data_off = 2 + don
            if len(pay) < data_off:
                continue
            fu_hdr = pay[1]
            start = bool(fu_hdr & 0x80)
            end = bool(fu_hdr & 0x40)
            inner_type = fu_hdr & 0x1F
            if start:
                # Reconstruct the original 1-byte NAL header: keep
                # forbidden_zero + nal_ref_idc from the FU indicator, take the
                # type from the FU header.
                nal_hdr = (b0 & 0xE0) | inner_type
                fu_buf = bytearray([nal_hdr])
                fu_buf += pay[data_off:]
                fu_active = True
            elif fu_active:
                fu_buf += pay[data_off:]
                if end:
                    out.append(bytes(fu_buf))
                    fu_active = False

        elif t == NAL_STAP_A:
            # indicator(1) + DON(2) + [size(2) + NAL]...
            pos = 3
            n = len(pay)
            while pos + 2 <= n:
                size = struct.unpack(">H", pay[pos:pos + 2])[0]
                pos += 2
                if size == 0 or pos + size > n:
                    break
                out.append(bytes(pay[pos:pos + size]))
                pos += size

        elif 1 <= t <= 23:
            # Single NAL — Apple sends it DON-LESS (the slice header follows the
            # 1-byte NAL header directly, same as its FU-A fragments which are
            # also DON-less). The previous code stripped 2 bytes assuming a DON,
            # which corrupted every single-packet slice's header (bogus
            # first_mb/pps_id → "non-existing PPS" → decoder reset). Large slices
            # arrive as FU-A and were fine; only small single-packet slices broke.
            if len(pay) < 2:
                continue
            out.append(bytes(pay))

    return out


def au_is_keyframe(au_annexb: bytes) -> bool:
    """True if an Annex-B access unit is independently decodable (a keyframe),
    so a WebCodecs decoder can start on it. Apple's H.264 has no type-5 IDR —
    keyframes are intra (I) slices in ordinary type-1 NALs — so we parse the
    slice header's slice_type. IDR (type 5) is always key; for type-1 slices,
    slice_type in {2,7} (I / I-only-picture) is a keyframe.

    The slice header after the 1-byte NAL header is:
      first_mb_in_slice  ue(v)
      slice_type         ue(v)
    on the emulation-prevention-stripped RBSP.
    """
    from .bitstream import BitReader, remove_emulation_prevention
    for nal in au_annexb.split(b"\x00\x00\x00\x01"):
        if len(nal) < 2:
            continue
        t = nal[0] & 0x1F
        if t == NAL_SLICE_IDR:
            return True
        if t == NAL_SLICE_NONIDR:
            try:
                br = BitReader(remove_emulation_prevention(nal[1:]))
                br.read_ue()                 # first_mb_in_slice
                slice_type = br.read_ue()    # slice_type
                if slice_type in (2, 7):
                    return True
            except Exception:
                pass
    return False


def au_has_sps(au_annexb: bytes) -> bool:
    """True if the Annex-B access unit already carries an SPS NAL (type 7)
    in-band. macOS's single-tile keyframes embed SPS/PPS; prepending our
    cached copies on top of those gives the decoder DUPLICATE parameter sets,
    which WebCodecs rejects with 'Decoding error' on every keyframe."""
    for nal in au_annexb.split(b"\x00\x00\x00\x01"):
        if nal and (nal[0] & 0x1F) == 7:   # NAL_SPS
            return True
    return False


def parse_avc_config(payload: bytes) -> Optional[tuple[bytes, bytes]]:
    """Extract (sps, pps) raw NAL bytes from an Apple H.264 config packet (the
    0x92 packet wrapping an `avc1`/`avcC` box). Returns None if no avcC is
    found. SPS/PPS are returned without start codes — the caller adds Annex-B
    or builds extradata.

    avcDecoderConfigurationRecord layout after the `avcC` fourcc:
      configurationVersion(1) profile(1) compat(1) level(1)
      lengthSizeMinusOne(1) numSPS(1, &0x1f) [SPS_len(2) SPS]...
      numPPS(1) [PPS_len(2) PPS]...
    """
    # Anchor the avcC search to the avc1 sample entry that wraps it, so a stray
    # b"avcC" byte sequence inside SPS/PPS payload data can't be mistaken for
    # the box. Fall back to a bare search only if avc1 isn't present.
    a1 = payload.find(b"avc1")
    idx = payload.find(b"avcC", a1 if a1 >= 0 else 0)
    if idx < 0:
        return None
    p = idx + 4
    if p + 6 > len(payload):
        return None
    num_sps = payload[p + 5] & 0x1F
    p += 6
    sps: Optional[bytes] = None
    for _ in range(num_sps):
        if p + 2 > len(payload):
            return None
        ln = struct.unpack(">H", payload[p:p + 2])[0]
        p += 2
        if p + ln > len(payload):
            return None
        if sps is None:
            sps = bytes(payload[p:p + ln])
        p += ln
    if p + 1 > len(payload):
        return None
    num_pps = payload[p]
    p += 1
    pps: Optional[bytes] = None
    for _ in range(num_pps):
        if p + 2 > len(payload):
            return None
        ln = struct.unpack(">H", payload[p:p + 2])[0]
        p += 2
        if p + ln > len(payload):
            return None
        if pps is None:
            pps = bytes(payload[p:p + ln])
        p += ln
    if sps is None or pps is None:
        return None
    return sps, pps


__all__ = [
    "APPLE_CONFIG_MARKER",
    "NAL_FU_A",
    "NAL_FU_B",
    "NAL_PPS",
    "NAL_SLICE_IDR",
    "au_is_keyframe",
    "NAL_SLICE_NONIDR",
    "NAL_SPS",
    "NAL_STAP_A",
    "h264_nal_type",
    "parse_avc_config",
    "reassemble_h264",
]
