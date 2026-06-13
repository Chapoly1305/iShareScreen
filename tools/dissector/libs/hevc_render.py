"""Render the dynamic (HEVC) media of a Screen Sharing session to MP4.

Decrypts the server->client SRTP, depayloads with Apple's DONL conventions,
orders NALUs across the 4 tile streams by their (roll-over-unwrapped) DONL —
Apple's cross-stream decode order — feeds a single HEVC decoder, composites the
per-tile pictures, and encodes an H.264 MP4 at a fixed frame rate.

**Memory**: Compressed NALUs are collected for DONL sorting (up to ~70 MB for
the largest sessions). Decoded tile-frames are never accumulated — only the
most recent frame per tile is kept (one ndarray per tile, ~2 MB total).
The decode runs twice (analysis pass to determine t0/t1/tile shapes, then
the encode pass), which doubles CPU time but keeps memory bounded.

Requires PyAV + numpy (optional dependency); raises RuntimeError if unavailable.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Optional

from scapy.all import IP, UDP, PcapReader  # type: ignore[import-untyped]

from .srtp import SRTPReceiver

try:
    import av  # type: ignore[import-untyped]
    import numpy as np  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional
    av = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]

START = b"\x00\x00\x00\x01"
_PARAM_TYPES = (32, 33, 34)  # VPS, SPS, PPS


def select_video_stream(
    pcap_path: str | Path,
    server_ip: str,
    ports: list[int],
    key_blocks: list[dict[str, object]],
    sample: int = 200,
) -> Optional[dict[str, object]]:
    """Find the (port, key) that carries decryptable HEVC video.

    Samples the leading server packets on each port, auth-tests every 46-byte
    key block (key2 first), and returns the mapping whose winning key block is a
    ``video`` stream. Returns ``None`` if nothing authenticates as video."""
    cands: list[dict[str, object]] = []
    for block in key_blocks:
        name = str(block.get("stream"))
        for label in ("key2_hex", "key1_hex"):
            kh = block.get(label)
            if not isinstance(kh, str):
                continue
            try:
                kb = bytes.fromhex(kh)
            except ValueError:
                continue
            if len(kb) == 46:
                cands.append({"stream": name, "label": label, "blob": kb})
    if not cands:
        return None

    port_set = set(ports)
    samples: dict[int, list[bytes]] = {p: [] for p in ports}
    need = {p: sample for p in ports}
    for p in PcapReader(str(pcap_path)):
        if IP not in p or UDP not in p or p[IP].src != server_ip:
            continue
        sp = int(p[UDP].sport)
        if sp not in port_set or need[sp] <= 0:
            continue
        samples[sp].append(bytes(p[UDP].payload))
        need[sp] -= 1
        if all(v <= 0 for v in need.values()):
            break

    best: Optional[dict[str, object]] = None
    for port, raws in samples.items():
        if not raws:
            continue
        for cand in cands:
            rx = SRTPReceiver.from_blob(cand["blob"])  # type: ignore[arg-type]
            ok = sum(1 for raw in raws if rx.decrypt(raw) is not None)
            if ok > 0 and str(cand["stream"]).startswith("video"):
                if best is None or ok > int(best["ok"]):
                    best = {"port": port, "stream": cand["stream"],
                            "label": cand["label"], "blob": cand["blob"], "ok": ok}
    return best


def _collect_items(
    pcap_path: str | Path,
    server_ip: str,
    video_port: int,
    key_blob: bytes,
) -> tuple[list[dict], dict[int, bytes], dict[int, int]]:
    """Decrypt + depayload the entire capture → DONL-orderable items.

    Returns ``(items, param_sets, ssrc→tile_index)``. Each item is a small
    dict ``{tile, nalu, arr, don}`` — NALU bytes are compressed (a few KB
    each), so the list stays well under 100 MB even for the busiest sessions."""
    dec = SRTPReceiver.from_blob(key_blob)
    ssrc_order: dict[int, int] = {}
    fu: dict[int, list] = {}
    items: list[dict] = []
    ps: dict[int, bytes] = {}

    def add(nalu: bytes, don: int, tile: int, arr: float) -> None:
        t = (nalu[0] >> 1) & 0x3F
        if t in _PARAM_TYPES:
            ps.setdefault(t, nalu)
        items.append({"tile": tile, "nalu": nalu, "arr": arr, "don": don})

    for p in PcapReader(str(pcap_path)):
        if IP not in p or UDP not in p or p[IP].src != server_ip or int(p[UDP].sport) != video_port:
            continue
        raw = bytes(p[UDP].payload)
        res = dec.decrypt(raw)
        if res is None:
            continue
        _hdr, pay = res
        if len(pay) < 2:
            continue
        arr = float(p.time)
        ssrc = int.from_bytes(raw[8:12], "big")
        tile = ssrc_order.setdefault(ssrc, len(ssrc_order))
        nt = (pay[0] >> 1) & 0x3F
        if nt == 48:  # AP: hdr(2)+DONL(2)+[size(2)+data]..
            don = (pay[2] << 8) | pay[3]
            pos, n, k = 4, len(pay), 0
            while pos + 2 <= n:
                sz = (pay[pos] << 8) | pay[pos + 1]
                pos += 2
                if sz == 0 or pos + sz > n:
                    break
                add(bytes(pay[pos:pos + sz]), (don + k) & 0xFFFF, tile, arr)
                pos += sz
                k += 1
        elif nt == 49:  # FU: hdr(2)+fuhdr(1)+DONL(2)+data
            if len(pay) < 6:
                continue
            fh = pay[2]
            if fh & 0x80:
                fu[ssrc] = [bytearray([(pay[0] & 0x81) | ((fh & 0x3F) << 1), pay[1]]) + pay[5:],
                            (pay[3] << 8) | pay[4]]
            elif ssrc in fu:
                fu[ssrc][0] += pay[5:]
                if fh & 0x40:
                    b, don = fu.pop(ssrc)
                    add(bytes(b), don, tile, arr)
        else:  # single NAL: hdr(2)+DONL(2)+data
            if len(pay) >= 4:
                add(bytes(pay[:2]) + bytes(pay[4:]), (pay[2] << 8) | pay[3], tile, arr)

    if not items:
        raise RuntimeError("no decryptable video NALUs (wrong key/port/server_ip?)")

    # unwrap 16-bit DON → monotonic, then sort (global decode order)
    prev: Optional[int] = None
    base = 0
    for it in items:
        d = it["don"]
        if prev is not None and d < prev - 32768:
            base += 65536
        prev = d
        it["don_full"] = base + d
    items.sort(key=lambda it: it["don_full"])
    return items, ps, ssrc_order


def _new_decoder(ps: dict[int, bytes]) -> "av.CodecContext":
    """Create a fresh HEVC decoder with the given param sets as extradata."""
    ctx = av.CodecContext.create("hevc", "r")
    ctx.extradata = b"".join(START + ps[t] for t in _PARAM_TYPES if t in ps)
    ctx.thread_type = "NONE"
    ctx.thread_count = 1
    from av.codec.context import Flags, Flags2
    ctx.flags |= Flags.low_delay | Flags.output_corrupt
    ctx.flags2 |= Flags2.show_all
    ctx.open()
    return ctx


def render_session_mp4(
    pcap_path: str | Path,
    server_ip: str,
    video_port: int,
    key_blob: bytes,
    out_path: str | Path,
    *,
    fps: int = 30,
    out_w: int = 1920,
    out_h: int = 1080,
) -> dict[str, object]:
    """Decode the video stream and write ``out_path`` as H.264 MP4.

    **Memory-safe**: never accumulates decoded RGB frames. Runs the HEVC
    decoder twice (analysis pass + encode pass); CPU cost is modest — the
    bottleneck is the decoder itself, not the compositing."""
    if av is None or np is None:
        raise RuntimeError("PyAV/numpy not available; install av to enable MP4 export")

    # ── Phase 1: decrypt + depayload + DONL sort ────────────────────────
    items, ps, ssrc_order = _collect_items(pcap_path, server_ip, video_port, key_blob)
    ntiles = len(ssrc_order)
    type_hist = Counter((it["nalu"][0] >> 1) & 0x3F for it in items)

    # ── Phase 1b: analysis decode (determine t0, t1, tile shapes) ────────
    ctx1 = _new_decoder(ps)
    pts_meta: dict[int, tuple[int, float]] = {}
    tile_shape: dict[int, tuple[int, int]] = {}  # tile→(h,w)
    t0: Optional[float] = None
    t1: Optional[float] = None
    decoded_count = 0

    for i, it in enumerate(items):
        pk = av.Packet(START + it["nalu"])
        pk.pts = i
        pk.dts = i
        arr = it["arr"]
        tile = it["tile"]
        pts_meta[i] = (tile, arr)
        try:
            for fr in ctx1.decode(pk):
                m = pts_meta.get(fr.pts)
                if m is None:
                    continue
                decoded_count += 1
                fa = m[1]
                if t0 is None:
                    t0 = fa
                t1 = fa
                ti = m[0]
                if ti not in tile_shape:
                    h, w = fr.height, fr.width
                    tile_shape[ti] = (h, w)
        except Exception:
            pass
    for fr in ctx1.decode(None):
        m = pts_meta.get(fr.pts)
        if m is not None:
            decoded_count += 1
            fa = m[1]
            if t0 is None:
                t0 = fa
            t1 = fa
            ti = m[0]
            if ti not in tile_shape:
                tile_shape[ti] = (fr.height, fr.width)
    ctx1 = None  # release decoder

    if t0 is None or t1 is None:
        raise RuntimeError("decoder produced no frames")
    dur = max(t1 - t0, 0.5)
    nframes = int(dur * fps) + 1

    # ── Phase 2: re-decode + encode on-the-fly ──────────────────────────
    th = max((s[0] for s in tile_shape.values()), default=1080)
    tw = max((s[1] for s in tile_shape.values()), default=1920)
    canvas = np.zeros((th * ntiles, tw, 3), np.uint8)
    latest: dict[int, np.ndarray] = {}  # tile→RGB (one frame per tile only)

    ctx2 = _new_decoder(ps)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out = av.open(str(out_path), "w")
    vs = out.add_stream("h264", rate=fps)
    vs.width, vs.height, vs.pix_fmt = out_w, out_h, "yuv420p"

    k_out = 0  # next output frame index
    for i, it in enumerate(items):
        pk = av.Packet(START + it["nalu"])
        pk.pts = i
        pk.dts = i
        try:
            for fr in ctx2.decode(pk):
                m = pts_meta.get(fr.pts)
                if m is None:
                    continue
                ti, fa = m
                latest[ti] = fr.to_ndarray(format="rgb24")
                # encode any output frames whose deadline has passed
                while k_out < nframes:
                    t = t0 + k_out / fps
                    if t > fa:
                        break
                    _encode_composite(canvas, latest, tile_shape, vs, out, out_w, out_h)
                    k_out += 1
        except Exception:
            pass
    # flush decoder
    for fr in ctx2.decode(None):
        m = pts_meta.get(fr.pts)
        if m is not None:
            ti, _fa = m
            latest[ti] = fr.to_ndarray(format="rgb24")
    # encode any remaining output frames
    while k_out < nframes:
        _encode_composite(canvas, latest, tile_shape, vs, out, out_w, out_h)
        k_out += 1
    ctx2 = None

    # flush encoder tail
    for pkt in vs.encode():
        out.mux(pkt)
    out.close()

    return {
        "out_path": str(out_path),
        "nalus": len(items),
        "tiles": ntiles,
        "don_span": [items[0]["don_full"], items[-1]["don_full"]],
        "decoded_tile_frames": decoded_count,
        "duration_seconds": round(dur, 3),
        "frames": nframes,
        "fps": fps,
        "resolution": [out_w, out_h],
        "nal_type_histogram": {str(t): int(c) for t, c in sorted(type_hist.items())},
        "size_bytes": Path(out_path).stat().st_size,
    }


def _encode_composite(
    canvas: np.ndarray,
    latest: dict[int, np.ndarray],
    tile_shape: dict[int, tuple[int, int]],
    vs: "av.stream.Stream",
    out: "av.container.OutputContainer",
    out_w: int,
    out_h: int,
) -> None:
    """Composite `latest[tile]` into `canvas` and mux one frame."""
    for ti in sorted(latest):
        rgb = latest[ti]
        off = sum(tile_shape.get(t, (0, 0))[0] for t in sorted(tile_shape) if t < ti)
        if off + rgb.shape[0] <= canvas.shape[0]:
            canvas[off:off + rgb.shape[0], :rgb.shape[1]] = rgb
    fr = av.VideoFrame.from_ndarray(np.ascontiguousarray(canvas), format="rgb24")
    for pkt in vs.encode(fr.reformat(width=out_w, height=out_h, format="yuv420p")):
        out.mux(pkt)


__all__ = ["render_session_mp4", "select_video_stream"]
