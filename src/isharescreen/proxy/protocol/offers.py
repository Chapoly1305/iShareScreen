"""AVCMediaStreamNegotiator offer build + answer parse.

The negotiation protobuf shape is reverse-engineered from
`-[AVCMediaStreamNegotiator createOffer]`. We rebuild it from scratch in
pure Python so the package works on Linux and Windows (Apple's framework
is macOS-only). Default settings preserve this client's existing wire bytes;
stream identity, timestamp, and CallID UUID are generated per offer.
"""
from __future__ import annotations

import logging
import platform
import plistlib
import secrets
import time
import uuid
import zlib
from typing import Optional

from .media_config import MediaOfferConfig
from .protobuf import field_bytes as _field_bytes, field_varint as _field_varint, varint as _varint

from ... import __version__


log = logging.getLogger(__name__)

import os as _os


def tiles_per_frame(config: Optional[MediaOfferConfig] = None) -> int:
    """The `tilesPerFrame` value we advertise in the media offer (field 6).
    macOS honors it, so this is also the number of video SSRCs/tiles to expect
    in the stream. Default 4 = Apple SS.app's byte-identical offer (4 parallel
    H.264 sub-streams); ISS_TILES_PER_FRAME=1 asks for a single picture per
    frame (browser-WebCodecs-decodable, no cross-tile references).

    Codec-dependent default: AVC defaults to 1, HEVC to 4. Apple's 4-tile
    stream uses CROSS-TILE references (a tile's P-frames reference POCs owned
    by other tiles). Our HEVC path decodes through native VideoToolbox, which
    matches Apple's own reference model and follows that structure fine. The
    AVC path decodes the four interleaved H.264 sub-streams in one shared
    libav context, which resolves the cross-tile references to the wrong
    tile's pictures — no decode error, so the corrupted frame is published and
    drift/ghosting ("fleas") accumulates under motion. Requesting a single
    self-contained picture per frame removes the cross-tile dimension so libav
    decodes cleanly (the same workaround the browser frontend already uses).
    An explicit config wins; otherwise ISS_TILES_PER_FRAME overrides the
    codec-dependent default."""
    if config is not None:
        return config.video.effective_tiles_per_frame
    codec = _os.environ.get("ISS_VIDEO_CODEC", "").lower()
    default = "1" if codec == "avc" else "4"
    try:
        return max(1, int(_os.environ.get("ISS_TILES_PER_FRAME", default)))
    except ValueError:
        return 1 if codec == "avc" else 4


# ── protobuf helpers ──────────────────────────────────────────────────

def _read_varint(data: bytes, pos: int) -> tuple[int, int]:
    val = 0
    shift = 0
    while pos < len(data):
        b = data[pos]
        pos += 1
        val |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            break
    return val, pos


def _ltrp_enabled_for_codec(codec: str) -> bool:
    """Legacy environment helper; explicit offers use VideoSettings instead."""
    return codec != "avc" and _os.environ.get("ISS_LTRP", "1") != "0"


def _build_remote_endpoint_info() -> bytes:
    """RemoteEndpointInfo protobuf: derive hw_model/os_build at runtime so we
    don't masquerade as the recording host. AVConference treats this as
    informational; populates opportunistically. The whole 0x1c message is
    enc1103-wrapped on the wire, so this only reaches the daemon, not
    passive observers."""
    hw_model = "Generic"
    avc_version = "1.0.0"
    os_build = "0"
    try:
        sys_name = platform.system()
        if sys_name == "Darwin":
            import subprocess
            hw_model = subprocess.check_output(
                ["sysctl", "-n", "hw.model"], text=True, timeout=1
            ).strip() or hw_model
            os_build = subprocess.check_output(
                ["sw_vers", "-buildVersion"], text=True, timeout=1
            ).strip() or os_build
        elif sys_name in ("Linux", "Windows"):
            hw_model = f"{sys_name}-{platform.machine()}"
            os_build = platform.release()
    except Exception as e:
        log.debug("RemoteEndpointInfo probe failed: %s", e)

    def _str(tag: int, s: str) -> bytes:
        b = s.encode("utf-8")[:127]
        return bytes([tag, len(b)]) + b

    return (
        b"\x08\x00"        # f1 = 0
        + b"\x10\x01"      # f2 = 1
        + _str(0x1A, hw_model)
        + _str(0x22, avc_version)
        + _str(0x2A, os_build)
    )


_REMOTE_ENDPOINT_INFO = _build_remote_endpoint_info()


# ── offer construction ───────────────────────────────────────────────

def _build_mediablob(
    mode: int, session_id: int, timestamp: int, *, audio_enabled: bool = True,
    config: Optional[MediaOfferConfig] = None,
) -> bytes:
    """Serialize editable offer settings plus fresh stream identity/time.

    mode 7 carries screenSettings (F5); mode 8 carries audioSettings (F3).
    Audio F4 is payloadFlags, not a requested bitrate. The existing on/off
    masks remain unchanged; see docs/analysis/media-negotiation-24G624.md.
    """
    config = config if config is not None else MediaOfferConfig.from_environment()
    config.validate()
    if mode == 7:
        desc_field = _field_bytes(5, config.video.to_protobuf(session_id))
    elif mode == 8:
        desc_field = _field_bytes(3, config.audio.to_protobuf(session_id, enabled=audio_enabled))
    else:
        raise ValueError(f"unsupported negotiation mode {mode}")
    return (
        _field_varint(1, int(config.allow_dynamic_max_bitrate))
        + _field_varint(2, int(config.allows_contents_change_with_aspect_preservation))
        + desc_field
        + _field_bytes(6, f"iShareScreen {__version__}".encode("ascii"))
        + _field_varint(8, config.baseband_codec_sample_rate)
        + b"".join(_field_bytes(9, entry.to_protobuf()) for entry in config.bandwidth)
        + _field_varint(13, timestamp)
        + _field_varint(14, config.blob_version)
        + _field_varint(16, config.media_control_info_version)
        + _field_varint(18, config.access_network_type)
    )


def create_offers(
    *, audio_enabled: bool = True, config: Optional[MediaOfferConfig] = None,
) -> tuple[bytes, bytes]:
    """Generate fresh (video, audio) offer plists from a configurable schema.

    An explicit config wins over environment variables. Without one, resolve
    the legacy ISS_* variables once for both streams. audio_enabled=False
    selects the existing disabled payload mask unless explicitly overridden.
    """
    from copy import deepcopy
    config = deepcopy(config) if config is not None else MediaOfferConfig.from_environment()
    config.validate()

    def _plist(mode: int) -> bytes:
        session_id = secrets.randbits(32)
        timestamp = time.time_ns()
        blob = _build_mediablob(
            mode, session_id, timestamp, audio_enabled=audio_enabled, config=config)
        plist = {
            "avcMediaStreamOptionRemoteEndpointInfo": _REMOTE_ENDPOINT_INFO,
            "avcMediaStreamNegotiatorMode": mode,
            "avcMediaStreamNegotiatorMediaBlob": zlib.compress(blob),
            "avcMediaStreamOptionCallID": str(uuid.uuid4()).upper(),
        }
        return plistlib.dumps(plist, fmt=plistlib.FMT_BINARY)

    return _plist(7), _plist(8)


# ── answer parsing ────────────────────────────────────────────────────

def extract_offer_ssrc(offer_plist: bytes, *, is_video: bool) -> Optional[int]:
    """Pull our advertised SSRC from a freshly-built offer.

    AVConference accepts only RTCP/RTP from the SSRC we negotiated; using a
    different one (e.g. 0x00DECADE) silently triggers `noRemotePacketsTimeout`
    after ~30 s. Video SSRC lives in field 5→1; audio in field 3→1.
    """
    plist = plistlib.loads(offer_plist)
    blob = zlib.decompress(plist["avcMediaStreamNegotiatorMediaBlob"])
    target = 5 if is_video else 3
    pos = 0
    while pos < len(blob):
        tag, pos = _read_varint(blob, pos)
        fn = tag >> 3
        wt = tag & 7
        if wt == 0:
            _, pos = _read_varint(blob, pos)
        elif wt == 2:
            length, pos = _read_varint(blob, pos)
            if fn == target:
                inner = blob[pos:pos + length]
                ipos = 0
                inner_tag, ipos = _read_varint(inner, ipos)
                if (inner_tag & 7) == 0 and (inner_tag >> 3) == 1:
                    ssrc, _ = _read_varint(inner, ipos)
                    return ssrc & 0xFFFFFFFF
            pos += length
        elif wt == 1:
            pos += 8
        elif wt == 5:
            pos += 4
        else:
            break
    return None


def extract_canvas_dims(answer_msg: bytes) -> tuple[int, int, int]:
    """Pull (canvas_w, canvas_h, num_tiles) from the server's 0x1c answer.

    The video media-stream answer's protobuf sits inside an embedded bplist.
    Top-level F5 carries the video config, with F4=canvas_width,
    F5=canvas_height (luma samples) and F6=tile_count. Returns zeros if not
    found — the caller should treat that as "encoder not ready, retry".
    """
    if not answer_msg or answer_msg[0] != 0x00:
        return 0, 0, 0
    idx = 0
    while True:
        idx = answer_msg.find(b"bplist", idx)
        if idx < 0:
            return 0, 0, 0
        plist_obj = None
        for end in range(idx + 1, len(answer_msg) + 1, 2):
            try:
                plist_obj = plistlib.loads(answer_msg[idx:end])
                break
            except Exception:
                plist_obj = None
        if not isinstance(plist_obj, dict):
            idx += 6
            continue
        blob = plist_obj.get("avcMediaStreamNegotiatorMediaBlob")
        if not blob:
            idx += 6
            continue
        try:
            dec = zlib.decompress(blob)
        except Exception:
            idx += 6
            continue
        cw = ch = ct = 0
        pos = 0
        while pos < len(dec):
            tag, pos = _read_varint(dec, pos)
            fn = tag >> 3
            wt = tag & 7
            if wt == 0:
                _, pos = _read_varint(dec, pos)
            elif wt == 2:
                ln, pos = _read_varint(dec, pos)
                if fn == 5:
                    sub = dec[pos:pos + ln]
                    sp = 0
                    while sp < len(sub):
                        st, sp = _read_varint(sub, sp)
                        sf = st >> 3
                        sw = st & 7
                        if sw == 0:
                            v, sp = _read_varint(sub, sp)
                            if sf == 4:
                                cw = v
                            elif sf == 5:
                                ch = v
                            elif sf == 6:
                                ct = v
                            elif sf == 7:
                                # field 7 = ltrpEnabled (negotiated result).
                                # Diagnostic: did the server accept LTRP?
                                log.info("negotiated ltrpEnabled (answer F5.f7) = %d", v)
                        elif sw == 2:
                            sl, sp = _read_varint(sub, sp)
                            sp += sl
                        elif sw == 1:
                            sp += 8
                        elif sw == 5:
                            sp += 4
                        else:
                            break
                pos += ln
            elif wt == 1:
                pos += 8
            elif wt == 5:
                pos += 4
            else:
                break
        if cw and ch:
            return cw, ch, ct
        idx += 6


__all__ = ["MediaOfferConfig", "create_offers", "extract_canvas_dims", "extract_offer_ssrc"]
