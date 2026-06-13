from __future__ import annotations

import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scapy.all import IP, IPv6, UDP, PcapReader  # type: ignore[import-untyped]

try:
    import pylibsrtp  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency
    pylibsrtp = None

try:
    if pylibsrtp is not None:
        from pylibsrtp import _binding as pylibsrtp_binding  # type: ignore[import-untyped]
    else:  # pragma: no cover - optional dependency
        pylibsrtp_binding = None
except Exception:  # pragma: no cover - optional dependency
    pylibsrtp_binding = None

# Self-contained SRTP receive + Apple HEVC depayload (no libsrtp needed). This
# is the real decode path; the pylibsrtp blocks above remain only as a
# best-effort cipher-suite probe.
try:
    from .srtp import SRTPReceiver
    from . import hevc_depay
except Exception:  # pragma: no cover - pycryptodome missing
    SRTPReceiver = None  # type: ignore[assignment]
    hevc_depay = None  # type: ignore[assignment]

AVC_SUITE5_KEY_LEN = 46
_LOWLEVEL_SRTP_INIT_ATTEMPTED = False
_LOWLEVEL_SRTP_INIT_OK = False


def _payload_entropy(data: bytes) -> float:
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for freq in counts.values():
        p = freq / total
        entropy -= p * math.log2(p)
    return entropy


def _parse_rtp(packet_payload: bytes) -> dict[str, object] | None:
    if len(packet_payload) < 12:
        return None
    b0 = packet_payload[0]
    b1 = packet_payload[1]
    version = (b0 >> 6) & 0x03
    if version != 2:
        return None

    has_padding = ((b0 >> 5) & 0x01) == 1
    has_extension = ((b0 >> 4) & 0x01) == 1
    csrc_count = b0 & 0x0F
    marker = ((b1 >> 7) & 0x01) == 1
    payload_type = b1 & 0x7F
    seq = int.from_bytes(packet_payload[2:4], "big")
    timestamp = int.from_bytes(packet_payload[4:8], "big")
    ssrc = int.from_bytes(packet_payload[8:12], "big")

    offset = 12 + (csrc_count * 4)
    if offset > len(packet_payload):
        return None

    ext_len_words = 0
    if has_extension:
        if offset + 4 > len(packet_payload):
            return None
        ext_len_words = int.from_bytes(packet_payload[offset + 2 : offset + 4], "big")
        offset += 4 + (ext_len_words * 4)
        if offset > len(packet_payload):
            return None

    payload_end = len(packet_payload)
    if has_padding:
        pad_len = packet_payload[-1]
        if pad_len <= 0 or pad_len > len(packet_payload) - offset:
            return None
        payload_end -= pad_len
    if payload_end < offset:
        return None

    rtp_payload = packet_payload[offset:payload_end]
    return {
        "version": version,
        "has_padding": has_padding,
        "has_extension": has_extension,
        "csrc_count": csrc_count,
        "marker": marker,
        "payload_type": payload_type,
        "sequence": seq,
        "timestamp": timestamp,
        "ssrc": ssrc,
        "extension_len_words": ext_len_words,
        "payload_len": len(rtp_payload),
        "payload": rtp_payload,
    }


def _summarize_lengths(lengths: list[int]) -> dict[str, object]:
    if not lengths:
        return {"count": 0}
    histogram = Counter(lengths).most_common(12)
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "avg": sum(lengths) / len(lengths),
        "top_histogram": [{"length": v, "count": c} for v, c in histogram],
    }


def _seq_gap_stats(seq_values: list[int]) -> dict[str, object]:
    if len(seq_values) < 2:
        return {
            "count": len(seq_values),
            "gaps_detected": 0,
            "out_of_order": 0,
        }
    missing_total = 0
    out_of_order = 0
    for prev, cur in zip(seq_values, seq_values[1:]):
        delta = (cur - prev) & 0xFFFF
        if delta == 0:
            out_of_order += 1
            continue
        if delta > 0x8000:
            out_of_order += 1
            continue
        if delta > 1:
            missing_total += delta - 1
    return {
        "count": len(seq_values),
        "gaps_detected": missing_total,
        "out_of_order": out_of_order,
    }


def _counter_to_str_dict(counter: Counter[int]) -> dict[str, int]:
    return {str(k): int(v) for k, v in sorted(counter.items())}


def _analyze_hevc_pt100(server_rtp: list[dict[str, object]]) -> dict[str, object]:
    payload_lens: list[int] = []
    nalu_type_counts: Counter[int] = Counter()
    fu_type_counts: Counter[int] = Counter()
    fu_start_counts: Counter[int] = Counter()
    fu_end_counts: Counter[int] = Counter()
    fu_seq_by_type: dict[int, list[int]] = defaultdict(list)
    single_keyframe_count = 0
    fu_keyframe_count = 0

    for r in server_rtp:
        if int(r["payload_type"]) != 100:
            continue
        payload = r.get("payload")
        if not isinstance(payload, (bytes, bytearray)):
            continue
        payload_bytes = bytes(payload)
        payload_lens.append(len(payload_bytes))
        if len(payload_bytes) < 2:
            continue
        nalu_type = (payload_bytes[0] >> 1) & 0x3F
        nalu_type_counts[nalu_type] += 1

        if nalu_type in (19, 20, 21):
            single_keyframe_count += 1

        if nalu_type != 49 or len(payload_bytes) < 3:
            continue
        fu_header = payload_bytes[2]
        fu_type = fu_header & 0x3F
        fu_type_counts[fu_type] += 1
        fu_seq_by_type[fu_type].append(int(r["sequence"]))
        if fu_header & 0x80:
            fu_start_counts[fu_type] += 1
        if fu_header & 0x40:
            fu_end_counts[fu_type] += 1
        if fu_type in (19, 20, 21):
            fu_keyframe_count += 1

    inter_vcl_count = int(nalu_type_counts.get(1, 0)) + int(fu_type_counts.get(1, 0))
    keyframe_vcl_count = single_keyframe_count + fu_keyframe_count
    likely_keyframe_starvation = keyframe_vcl_count <= 2 and inter_vcl_count >= 128
    initial_idr_burst_only = (
        int(fu_start_counts.get(20, 0)) <= 1
        and int(fu_end_counts.get(20, 0)) <= 1
        and inter_vcl_count >= max(64, keyframe_vcl_count * 8)
    )
    large_payload_count = sum(1 for n in payload_lens if n >= 300)
    idr_fu_seq_stats = _seq_gap_stats(fu_seq_by_type.get(20, []))

    return {
        "pt100_packet_count": len(payload_lens),
        "pt100_large_payload_count_ge_300": large_payload_count,
        "pt100_payload_lengths": _summarize_lengths(payload_lens),
        "nalu_type_counts": _counter_to_str_dict(nalu_type_counts),
        "fu_type_counts": _counter_to_str_dict(fu_type_counts),
        "fu_start_counts": _counter_to_str_dict(fu_start_counts),
        "fu_end_counts": _counter_to_str_dict(fu_end_counts),
        "idr_fu_seq_stats": idr_fu_seq_stats,
        "inter_vcl_count": inter_vcl_count,
        "keyframe_vcl_count": keyframe_vcl_count,
        "likely_keyframe_starvation": likely_keyframe_starvation,
        "initial_idr_burst_only": initial_idr_burst_only,
        "note": (
            "HEVC adaptive payload summary. FU type is parsed from payload[2] "
            "when outer NALU type is 49."
        ),
    }


def _infer_udp_port_roles(
    ports: list[int],
    by_port: dict[str, object],
) -> dict[str, object]:
    metrics: dict[int, dict[str, int]] = {}
    for port in ports:
        stream = by_port.get(str(port))
        if not isinstance(stream, dict):
            continue
        hevc = stream.get("hevc_pt100_analysis")
        if not isinstance(hevc, dict):
            continue
        large_payload = int(hevc.get("pt100_large_payload_count_ge_300") or 0)
        fu_counts = hevc.get("fu_type_counts")
        fu_packets = int(sum(int(v) for v in fu_counts.values())) if isinstance(fu_counts, dict) else 0
        nalu_counts = hevc.get("nalu_type_counts")
        nalu_packets = int(sum(int(v) for v in nalu_counts.values())) if isinstance(nalu_counts, dict) else 0
        video_score = (large_payload * 4) + fu_packets + nalu_packets
        metrics[port] = {
            "video_score": video_score,
            "large_payload_packets": large_payload,
            "fu_packets": fu_packets,
            "pt100_packets": nalu_packets,
        }

    if not metrics:
        return {
            "video_candidate_port": None,
            "audio_candidate_port": None,
            "mapping_note": "insufficient RTP/HEVC data",
            "per_port_scores": {},
        }

    sorted_ports = sorted(metrics.items(), key=lambda kv: (kv[1]["video_score"], kv[1]["large_payload_packets"]), reverse=True)
    video_candidate_port = sorted_ports[0][0] if sorted_ports[0][1]["video_score"] > 0 else None
    audio_candidate_port = None
    for port, _score in sorted_ports[1:]:
        if port != video_candidate_port:
            audio_candidate_port = port
            break

    note = "inferred from RTP payload shape (pt=100 + large FU payload density)"
    if video_candidate_port is None:
        note = "no clear video candidate; fallback mapping should use configured order"
    return {
        "video_candidate_port": video_candidate_port,
        "audio_candidate_port": audio_candidate_port,
        "mapping_note": note,
        "per_port_scores": {str(port): vals for port, vals in metrics.items()},
    }


def _srtp_profiles_for_len(key_len: int) -> list[tuple[str, int]]:
    if pylibsrtp is None:
        return []
    if key_len == 30:
        return [
            ("AES128_CM_SHA1_80", pylibsrtp.Policy.SRTP_PROFILE_AES128_CM_SHA1_80),
            ("AES128_CM_SHA1_32", pylibsrtp.Policy.SRTP_PROFILE_AES128_CM_SHA1_32),
        ]
    if key_len == 28:
        return [("AEAD_AES_128_GCM", pylibsrtp.Policy.SRTP_PROFILE_AEAD_AES_128_GCM)]
    if key_len == 44:
        return [("AEAD_AES_256_GCM", pylibsrtp.Policy.SRTP_PROFILE_AEAD_AES_256_GCM)]
    return []


def _candidate_key_materials(key_hex: str) -> list[dict[str, object]]:
    raw = bytes.fromhex(key_hex)
    # Native ScreenSharing.framework uses 46-byte media keys for AVC suite-5.
    # pylibsrtp's high-level Policy API does not expose AES_CM_256 profiles, so
    # probing windowed sub-slices here produces noisy false paths.
    if len(raw) == AVC_SUITE5_KEY_LEN:
        return []
    out: list[dict[str, object]] = []
    seen: set[tuple[int, int]] = set()
    for key_len in (30, 28, 44):
        if key_len > len(raw):
            continue
        for off in range(0, len(raw) - key_len + 1):
            marker = (off, key_len)
            if marker in seen:
                continue
            seen.add(marker)
            chunk = raw[off : off + key_len]
            for profile_name, profile_value in _srtp_profiles_for_len(key_len):
                out.append(
                    {
                        "offset": off,
                        "len": key_len,
                        "profile_name": profile_name,
                        "profile_value": profile_value,
                        "key_material": chunk,
                    }
                )
    return out


def _lowlevel_suite5_support() -> tuple[bool, str]:
    if pylibsrtp_binding is None:
        return False, "pylibsrtp low-level binding unavailable"
    ffi = pylibsrtp_binding.ffi
    lib = pylibsrtp_binding.lib
    required_symbols = (
        "srtp_init",
        "srtp_create",
        "srtp_unprotect",
        "srtp_dealloc",
        "srtp_crypto_policy_set_aes_cm_256_hmac_sha1_80",
        "srtp_crypto_policy_set_aes_cm_256_hmac_sha1_32",
    )
    missing = [name for name in required_symbols if not hasattr(lib, name)]
    if missing:
        return False, f"missing libsrtp symbols: {', '.join(missing)}"
    try:
        ffi.typeof("srtp_policy_t *")
        ffi.typeof("srtp_t *")
    except Exception as exc:  # pragma: no cover - CFFI type/runtime dependent
        return False, f"binding missing required types ({exc})"
    return True, "available"


def _lowlevel_srtp_init() -> tuple[bool, str]:
    global _LOWLEVEL_SRTP_INIT_ATTEMPTED, _LOWLEVEL_SRTP_INIT_OK
    if _LOWLEVEL_SRTP_INIT_ATTEMPTED:
        return (_LOWLEVEL_SRTP_INIT_OK, "initialized" if _LOWLEVEL_SRTP_INIT_OK else "init_failed")
    _LOWLEVEL_SRTP_INIT_ATTEMPTED = True
    if pylibsrtp_binding is None:
        _LOWLEVEL_SRTP_INIT_OK = False
        return False, "pylibsrtp low-level binding unavailable"
    lib = pylibsrtp_binding.lib
    try:
        status = int(lib.srtp_init())
    except Exception as exc:  # pragma: no cover - CFFI/runtime dependent
        _LOWLEVEL_SRTP_INIT_OK = False
        return False, f"srtp_init exception: {exc}"
    _LOWLEVEL_SRTP_INIT_OK = status == 0
    if _LOWLEVEL_SRTP_INIT_OK:
        return True, "initialized"
    return False, f"srtp_init failed status={status}"


def _lowlevel_probe_srtp_stream_suite5(
    packets: list[bytes],
    candidates: list[dict[str, object]],
    probe_packets: int,
) -> list[dict[str, object]]:
    if pylibsrtp_binding is None:
        return []
    ok, _ = _lowlevel_srtp_init()
    if not ok:
        return []

    ffi = pylibsrtp_binding.ffi
    lib = pylibsrtp_binding.lib
    subset = packets[: max(1, probe_packets)]
    results: list[dict[str, object]] = []
    for cand in candidates:
        try:
            policy = ffi.new("srtp_policy_t *")
            if bool(cand["use_short_tag"]):
                lib.srtp_crypto_policy_set_aes_cm_256_hmac_sha1_32(
                    ffi.addressof(policy[0], "rtp")
                )
                lib.srtp_crypto_policy_set_aes_cm_256_hmac_sha1_32(
                    ffi.addressof(policy[0], "rtcp")
                )
            else:
                lib.srtp_crypto_policy_set_aes_cm_256_hmac_sha1_80(
                    ffi.addressof(policy[0], "rtp")
                )
                lib.srtp_crypto_policy_set_aes_cm_256_hmac_sha1_80(
                    ffi.addressof(policy[0], "rtcp")
                )
            ssrc_any_inbound = getattr(lib, "ssrc_any_inbound", 1)
            policy[0].ssrc.type = int(ssrc_any_inbound)
            policy[0].ssrc.value = 0
            key_buf = ffi.new("unsigned char[]", bytes(cand["key_material"]))
            policy[0].key = key_buf
            policy[0].window_size = 128
            policy[0].allow_repeat_tx = 1
            policy[0].next = ffi.NULL
            session_p = ffi.new("srtp_t *")
            create_status = int(lib.srtp_create(session_p, policy))
            if create_status != 0:
                continue
            session = session_p[0]
        except Exception:
            continue

        ok_count = 0
        fail_count = 0
        for pkt in subset:
            try:
                pkt_buf = ffi.new("unsigned char[]", pkt)
                pkt_len = ffi.new("int *", len(pkt))
                status = int(lib.srtp_unprotect(session, pkt_buf, pkt_len))
                if status == 0:
                    ok_count += 1
                else:
                    fail_count += 1
                    if ok_count == 0 and fail_count >= 16:
                        break
            except Exception:
                fail_count += 1
                if ok_count == 0 and fail_count >= 16:
                    break

        try:
            lib.srtp_dealloc(session)
        except Exception:
            pass

        total = ok_count + fail_count
        results.append(
            {
                "origin": cand["origin"],
                "offset": cand["offset"],
                "key_len": cand["len"],
                "profile": cand["profile_name"],
                "ok": ok_count,
                "fail": fail_count,
                "success_rate": (ok_count / total) if total else 0.0,
                "backend": "pylibsrtp-lowlevel",
            }
        )

    results.sort(key=lambda r: (r["ok"], r["success_rate"]), reverse=True)
    return results[:8]


def _probe_srtp_stream(
    packets: list[bytes],
    candidates: list[dict[str, object]],
    probe_packets: int,
) -> list[dict[str, object]]:
    if pylibsrtp is None:
        return []
    subset = packets[: max(1, probe_packets)]
    results: list[dict[str, object]] = []
    for cand in candidates:
        try:
            policy = pylibsrtp.Policy(
                key=cand["key_material"],
                ssrc_type=pylibsrtp.Policy.SSRC_ANY_INBOUND,
                ssrc_value=0,
                srtp_profile=int(cand["profile_value"]),
            )
            session = pylibsrtp.Session(policy=policy)
        except Exception:
            continue
        ok = 0
        fail = 0
        for pkt in subset:
            try:
                session.unprotect(pkt)
                ok += 1
            except Exception:
                fail += 1
                if ok == 0 and fail >= 16:
                    break
        total = ok + fail
        results.append(
            {
                "origin": cand["origin"],
                "offset": cand["offset"],
                "key_len": cand["len"],
                "profile": cand["profile_name"],
                "ok": ok,
                "fail": fail,
                "success_rate": (ok / total) if total else 0.0,
                "backend": "pylibsrtp-policy",
            }
        )
    results.sort(key=lambda r: (r["ok"], r["success_rate"]), reverse=True)
    return results[:8]


def _build_srtp_probe_summary(
    ports: list[int],
    server_rtp_packets_by_port: dict[int, list[dict[str, object]]],
    stream_key_blocks: list[dict[str, object]],
    probe_packets: int,
    port_roles: dict[str, object] | None = None,
) -> dict[str, object]:
    if pylibsrtp is None:
        return {"available": False, "note": "pylibsrtp not installed"}
    lowlevel_ok, lowlevel_note = _lowlevel_suite5_support()

    # Port->stream mapping can vary across traces. Prefer RTP-shape inference and
    # fall back to configured order when needed.
    stream_to_port: dict[str, int] = {}
    inferred_video = None
    inferred_audio = None
    if isinstance(port_roles, dict):
        vp = port_roles.get("video_candidate_port")
        ap = port_roles.get("audio_candidate_port")
        if isinstance(vp, int):
            inferred_video = vp
        if isinstance(ap, int):
            inferred_audio = ap

    if inferred_video is not None:
        stream_to_port["video1"] = inferred_video
    elif ports:
        stream_to_port["video1"] = ports[0]

    if inferred_audio is not None and inferred_audio != stream_to_port.get("video1"):
        stream_to_port["audio"] = inferred_audio
    elif len(ports) > 1:
        stream_to_port["audio"] = ports[1]

    if len(ports) > 2:
        stream_to_port["video2"] = ports[2]

    per_stream: dict[str, Any] = {}
    for block in stream_key_blocks:
        stream_name = str(block.get("stream"))
        port = stream_to_port.get(stream_name)
        if port is None:
            per_stream[stream_name] = {"note": "no mapped UDP port"}
            continue
        rtp_records = server_rtp_packets_by_port.get(port, [])
        # PT 72 appears as control in observed traces; focus on media packets.
        probe_packets_raw = [r["raw_packet"] for r in rtp_records if int(r["payload_type"]) != 72]
        if not probe_packets_raw:
            per_stream[stream_name] = {"port": port, "note": "no server RTP media packets"}
            continue

        all_candidates: list[dict[str, object]] = []
        stream_key_lens: list[int] = []
        for key_label in ("key1_hex", "key2_hex"):
            key_hex = block.get(key_label)
            if not isinstance(key_hex, str):
                continue
            stream_key_lens.append(len(bytes.fromhex(key_hex)))
            for cand in _candidate_key_materials(key_hex):
                cand["origin"] = f"{stream_name}.{key_label}"
                all_candidates.append(cand)

        if any(klen == AVC_SUITE5_KEY_LEN for klen in stream_key_lens) and not all_candidates:
            suite5_candidates: list[dict[str, object]] = []
            for key_label in ("key1_hex", "key2_hex"):
                key_hex = block.get(key_label)
                if not isinstance(key_hex, str):
                    continue
                key_material = bytes.fromhex(key_hex)
                if len(key_material) != AVC_SUITE5_KEY_LEN:
                    continue
                for profile_name, short_tag in (
                    ("AES_CM_256_HMAC_SHA1_80", False),
                    ("AES_CM_256_HMAC_SHA1_32", True),
                ):
                    suite5_candidates.append(
                        {
                            "origin": f"{stream_name}.{key_label}",
                            "offset": 0,
                            "len": len(key_material),
                            "profile_name": profile_name,
                            "use_short_tag": short_tag,
                            "key_material": key_material,
                        }
                    )
            if lowlevel_ok and suite5_candidates:
                top = _lowlevel_probe_srtp_stream_suite5(
                    packets=probe_packets_raw,
                    candidates=suite5_candidates,
                    probe_packets=probe_packets,
                )
                per_stream[stream_name] = {
                    "port": port,
                    "probe_packets": min(len(probe_packets_raw), max(1, probe_packets)),
                    "candidate_count": len(suite5_candidates),
                    "key_lengths": stream_key_lens,
                    "likely_cipher_suite": "AVC suite-5 (AES_CM_256_HMAC_SHA1_80/32, 46-byte key+salt)",
                    "top_candidates": top,
                    "note": "used low-level pylibsrtp binding for suite-5 probe",
                }
                continue
            per_stream[stream_name] = {
                "port": port,
                "probe_packets": min(len(probe_packets_raw), max(1, probe_packets)),
                "candidate_count": 0,
                "key_lengths": stream_key_lens,
                "likely_cipher_suite": "AVC suite-5 (AES_CM_256_HMAC_SHA1_80/32, 46-byte key+salt)",
                "note": (
                    "46-byte media keys detected; suite-5 probe unavailable "
                    f"({lowlevel_note})"
                ),
            }
            continue

        top = _probe_srtp_stream(
            packets=probe_packets_raw,
            candidates=all_candidates,
            probe_packets=probe_packets,
        )
        per_stream[stream_name] = {
            "port": port,
            "probe_packets": min(len(probe_packets_raw), max(1, probe_packets)),
            "candidate_count": len(all_candidates),
            "top_candidates": top,
        }
    return {
        "available": True,
        "note": (
            "candidate probe only; validate decrypt hits against stream/key direction mapping. "
            f"suite5_lowlevel={'yes' if lowlevel_ok else 'no'}"
        ),
        "stream_port_mapping": stream_to_port,
        "per_stream": per_stream,
    }


def _hist_named(counter: Counter[int]) -> dict[str, int]:
    """Render a NAL-type histogram with human names, sorted by type."""
    return {f"{t} {hevc_depay.nal_name(t)}": int(c) for t, c in sorted(counter.items())}


def _decode_hevc_streams(
    server_rtp_packets_by_port: dict[int, list[dict[str, object]]],
    stream_key_blocks: list[dict[str, object]],
    sample_packets: int = 200,
) -> dict[str, object]:
    """Actually decrypt the server->client SRTP media and depayload HEVC.

    For each port, every 46-byte ``0x1c`` key block (``key2`` first — the proven
    receive direction — then ``key1``) is auth-tested against a leading sample;
    the key that authenticates the most packets wins (HMAC makes this
    self-validating). The whole port is then decrypted, classified per packet,
    and reassembled per SSRC (one SSRC == one tile) using Apple's DONL
    conventions, yielding real NAL-type histograms, param-set / IDR presence,
    and DONL spans."""
    if SRTPReceiver is None or hevc_depay is None:
        return {"available": False, "note": "pycryptodome SRTP backend unavailable"}

    key_candidates: list[dict[str, object]] = []
    for block in stream_key_blocks:
        stream_name = str(block.get("stream"))
        for label in ("key2_hex", "key1_hex"):
            kh = block.get(label)
            if not isinstance(kh, str):
                continue
            try:
                kb = bytes.fromhex(kh)
            except ValueError:
                continue
            if len(kb) != AVC_SUITE5_KEY_LEN:
                continue
            key_candidates.append({"stream": stream_name, "label": label, "blob": kb})
    if not key_candidates:
        return {"available": False, "note": "no 46-byte AVC suite-5 key blocks present"}

    per_port: dict[str, object] = {}
    decoded_any = False
    for port, server_rtp in server_rtp_packets_by_port.items():
        raws = [bytes(r["raw_packet"]) for r in server_rtp
                if isinstance(r.get("raw_packet"), (bytes, bytearray))]
        if not raws:
            per_port[str(port)] = {"note": "no server RTP packets"}
            continue

        sample = raws[: max(1, sample_packets)]
        best: dict[str, object] | None = None
        for cand in key_candidates:
            rx = SRTPReceiver.from_blob(cand["blob"])
            ok = sum(1 for raw in sample if rx.decrypt(raw) is not None)
            if best is None or ok > int(best["ok"]):
                best = {**cand, "ok": ok}
        if best is None or int(best["ok"]) == 0:
            per_port[str(port)] = {
                "note": "no key block authenticated this port's server RTP",
                "sampled": len(sample),
            }
            continue

        decoded_any = True
        is_video = str(best["stream"]).startswith("video")
        pt_hist: Counter[int] = Counter(int(r["payload_type"]) for r in server_rtp
                                        if isinstance(r.get("payload_type"), int))
        rx = SRTPReceiver.from_blob(best["blob"])  # type: ignore[arg-type]
        per_ssrc_payloads: dict[int, list[tuple[int, bytes]]] = defaultdict(list)
        per_ssrc_outer: dict[int, Counter[int]] = defaultdict(Counter)
        per_ssrc_contained: dict[int, Counter[int]] = defaultdict(Counter)
        agg_outer: Counter[int] = Counter()
        agg_contained: Counter[int] = Counter()
        auth = 0
        for raw in raws:
            res = rx.decrypt(raw)
            if res is None:
                continue
            _hdr, pay = res
            auth += 1
            ssrc = int.from_bytes(raw[8:12], "big")
            seq = (raw[2] << 8) | raw[3]
            per_ssrc_payloads[ssrc].append((seq, pay))
            if not is_video:
                continue
            info = hevc_depay.classify_payload(pay)
            if info is None:
                continue
            outer = int(info["outer_type"])
            per_ssrc_outer[ssrc][outer] += 1
            agg_outer[outer] += 1
            for t in info.get("contained", []):
                per_ssrc_contained[ssrc][t] += 1
                agg_contained[t] += 1

        if not is_video:
            # Audio (or any non-HEVC) stream: confirm decrypt, but don't impose
            # HEVC NAL semantics on the payload.
            per_port[str(port)] = {
                "key_stream": best["stream"],
                "key_label": best["label"],
                "kind": "non-video",
                "sample_auth": {"ok": int(best["ok"]), "sampled": len(sample)},
                "authenticated_packets": auth,
                "total_server_packets": len(raws),
                "ssrc_count": len(per_ssrc_payloads),
                "rtp_payload_types": {str(k): int(v) for k, v in sorted(pt_hist.items())},
                "note": "decrypt authenticated; HEVC depayload skipped for non-video stream",
            }
            continue

        tiles: dict[str, object] = {}
        for ti, ssrc in enumerate(sorted(per_ssrc_payloads)):
            ordered_pairs = sorted(per_ssrc_payloads[ssrc], key=lambda sp: sp[0])
            ordered = [p for _seq, p in ordered_pairs]
            nalus = [n for n in hevc_depay.reassemble_group(ordered) if n]
            nal_hist: Counter[int] = Counter((n[0] >> 1) & 0x3F for n in nalus)
            donls = [info["donl"] for info in (hevc_depay.classify_payload(p) for p in ordered)
                     if info and "donl" in info]
            unwrapped = hevc_depay.unwrap_dons(donls)
            tiles[f"0x{ssrc:08x}"] = {
                "tile_index": ti,
                "packets": len(per_ssrc_payloads[ssrc]),
                "nalus": len(nalus),
                "outer_type_counts": _hist_named(per_ssrc_outer[ssrc]),
                "contained_type_counts": _hist_named(per_ssrc_contained[ssrc]),
                "reassembled_nal_type_counts": _hist_named(nal_hist),
                "has_param_sets": all(t in nal_hist for t in (32, 33, 34)),
                "has_idr": any(t in nal_hist for t in hevc_depay.IDR_RANGE),
                "donl_span": [min(unwrapped), max(unwrapped)] if unwrapped else None,
            }

        per_port[str(port)] = {
            "key_stream": best["stream"],
            "key_label": best["label"],
            "kind": "video",
            "sample_auth": {"ok": int(best["ok"]), "sampled": len(sample)},
            "authenticated_packets": auth,
            "total_server_packets": len(raws),
            "ssrc_count": len(per_ssrc_payloads),
            "rtp_payload_types": {str(k): int(v) for k, v in sorted(pt_hist.items())},
            "outer_type_counts": _hist_named(agg_outer),
            "contained_type_counts": _hist_named(agg_contained),
            "has_param_sets": any(bool(t.get("has_param_sets")) for t in tiles.values()),
            "has_idr": any(bool(t.get("has_idr")) for t in tiles.values()),
            "tiles": tiles,
        }

    return {
        "available": True,
        "backend": "self-contained AES-256-CM + HMAC-SHA1-80 (pycryptodome)",
        "note": (
            "Real decrypt of server->client SRTP, depayloaded with Apple's DONL "
            "conventions: single-NAL/AP/FU all carry a leading 2-byte DONL, FU "
            "repeats it in every fragment, and AP has no DOND. One SSRC == one "
            "tile; global decode order is by unwrapped DONL."
        ),
        "decoded_any_port": decoded_any,
        "per_port": per_port,
    }


def analyze_udp_media_streams(
    pcap_path: str | Path,
    client_ip: str,
    server_ip: str,
    ports: list[int],
    entropy_sample_bytes: int = 1_000_000,
    stream_key_blocks: list[dict[str, object]] | None = None,
    srtp_probe_packets: int = 120,
    decrypt_media: bool = True,
    run_probe: bool = True,
) -> dict[str, object]:
    port_set = set(ports)
    stream_packets: dict[int, list[dict[str, object]]] = defaultdict(list)
    with PcapReader(str(pcap_path)) as reader:
        for frame_number, pkt in enumerate(reader, start=1):
            if UDP not in pkt:
                continue
            if IP in pkt:
                src_ip = str(pkt[IP].src)
                dst_ip = str(pkt[IP].dst)
            elif IPv6 in pkt:
                src_ip = str(pkt[IPv6].src)
                dst_ip = str(pkt[IPv6].dst)
            else:
                continue
            if not (
                (src_ip == client_ip and dst_ip == server_ip)
                or (src_ip == server_ip and dst_ip == client_ip)
            ):
                continue
            udp = pkt[UDP]
            src_port = int(udp.sport)
            dst_port = int(udp.dport)
            if src_port not in port_set and dst_port not in port_set:
                continue
            stream_port = src_port if src_port in port_set else dst_port
            payload = bytes(udp.payload)
            stream_packets[stream_port].append(
                {
                    "frame_number": frame_number,
                    "timestamp_epoch": float(pkt.time),
                    "src_ip": src_ip,
                    "dst_ip": dst_ip,
                    "src_port": src_port,
                    "dst_port": dst_port,
                    "payload": payload,
                }
            )

    by_port: dict[str, object] = {}
    server_rtp_packets_by_port: dict[int, list[dict[str, object]]] = {}
    for port in ports:
        packets = stream_packets.get(port, [])
        if not packets:
            by_port[str(port)] = {
                "packet_count": 0,
                "byte_count": 0,
                "rtp_packet_count": 0,
                "note": "no packets",
            }
            continue

        packet_lengths = [len(p["payload"]) for p in packets]
        from_server = [p for p in packets if p["src_ip"] == server_ip]
        from_client = [p for p in packets if p["src_ip"] == client_ip]
        rtp_parsed = []
        for p in packets:
            parsed = _parse_rtp(p["payload"])
            if parsed is None:
                continue
            parsed["src_ip"] = p["src_ip"]
            parsed["timestamp_epoch"] = p["timestamp_epoch"]
            parsed["raw_packet"] = p["payload"]
            rtp_parsed.append(parsed)

        server_rtp = [r for r in rtp_parsed if r["src_ip"] == server_ip]
        server_rtp_packets_by_port[port] = server_rtp
        payload_types = Counter(int(r["payload_type"]) for r in server_rtp)
        marker_count = sum(1 for r in server_rtp if r["marker"])

        sampled = bytearray()
        for r in server_rtp:
            if len(sampled) >= entropy_sample_bytes:
                break
            sampled.extend(r["payload"])
        sampled_bytes = bytes(sampled[:entropy_sample_bytes])

        seq_by_ssrc: dict[int, list[int]] = defaultdict(list)
        for r in server_rtp:
            seq_by_ssrc[int(r["ssrc"])].append(int(r["sequence"]))
        seq_stats = {
            f"0x{ssrc:08x}": _seq_gap_stats(values)
            for ssrc, values in seq_by_ssrc.items()
        }
        hevc_pt100 = _analyze_hevc_pt100(server_rtp)

        by_port[str(port)] = {
            "packet_count": len(packets),
            "byte_count": sum(packet_lengths),
            "duration_seconds": packets[-1]["timestamp_epoch"] - packets[0]["timestamp_epoch"],
            "first_ts": packets[0]["timestamp_epoch"],
            "last_ts": packets[-1]["timestamp_epoch"],
            "direction_counts": {
                "server_to_client": len(from_server),
                "client_to_server": len(from_client),
            },
            "udp_payload_lengths": _summarize_lengths(packet_lengths),
            "rtp_packet_count": len(rtp_parsed),
            "rtp_from_server_count": len(server_rtp),
            "rtp_payload_type_counts": {str(k): v for k, v in payload_types.items()},
            "rtp_marker_count": marker_count,
            "rtp_server_payload_entropy": _payload_entropy(sampled_bytes),
            "rtp_server_payload_sample_bytes": len(sampled_bytes),
            "rtp_seq_stats_by_ssrc": seq_stats,
            "hevc_pt100_analysis": hevc_pt100,
        }

    port_roles = _infer_udp_port_roles(ports=ports, by_port=by_port)
    out: dict[str, object] = {
        "pcap": str(pcap_path),
        "client_ip": client_ip,
        "server_ip": server_ip,
        "ports": ports,
        "port_role_inference": port_roles,
        "streams": by_port,
    }
    if stream_key_blocks:
        if decrypt_media:
            out["hevc_decode"] = _decode_hevc_streams(
                server_rtp_packets_by_port=server_rtp_packets_by_port,
                stream_key_blocks=stream_key_blocks,
            )
        if run_probe:
            out["srtp_probe"] = _build_srtp_probe_summary(
                ports=ports,
                server_rtp_packets_by_port=server_rtp_packets_by_port,
                stream_key_blocks=stream_key_blocks,
                probe_packets=max(1, srtp_probe_packets),
                port_roles=port_roles,
            )
    return out
