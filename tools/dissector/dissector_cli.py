#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any

from libs.capture_reader import iter_conversation_segments, read_conversation_segments, read_tcp_segments
from libs.cbc_records import (
    decode_cbc_records,
    find_all_rekey_info,
    find_client_cbc_start,
    find_client_cbc_start_by_preface,
    find_rekey_info,
    iter_cbc_records,
)
from libs.cbc_records import decrypt_ecb_block, decrypt_cbc_block
from libs.framebuffer import Framebuffer
from libs.output_jsonl import write_jsonl
from libs.rect_decoders import (
    ZlibStreamDecoder,
    decode_raw_pixels,
    display_layout_backing_size,
    framebuffer_update_expected_len,
    parse_framebuffer_rects,
)
from libs.message_parser import parse_client_message, rectangle_encoding_name
from libs.stream_selector import summarize_conversations
from libs.tcp_reassembly import reassemble_direction
from libs.timeline import PresentationFrame
from libs.types import DecodedRecord, ReassembledStream, RekeyInfo
from libs.udp_media import analyze_udp_media_streams
from libs.video_writer import ExactTimingReplayWriter, write_frame_ledger_with_end


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Standalone Screen Sharing dissector")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_streams = sub.add_parser("list-streams")
    list_streams.add_argument("--pcap", required=True)

    decode = sub.add_parser("decode")
    decode.add_argument("--pcap", required=True)
    decode.add_argument("--client-ip", required=True)
    decode.add_argument("--server-ip", required=True)
    decode.add_argument("--server-port", type=int, default=5900)
    decode.add_argument("--client-port", type=int)
    decode.add_argument("--initial-key-hex",
        help="post-auth transport key (hex). Optional: defaults to the "
             "sibling '<pcap-stem>.key' file written by `iss --record`.")
    decode.add_argument("--client-cbc-start-offset", type=int)
    decode.add_argument("--server-cbc-start-offset", type=int)
    decode.add_argument("--output")

    export_client = sub.add_parser("export-client-requests")
    export_client.add_argument("--pcap", required=True)
    export_client.add_argument("--client-ip", required=True)
    export_client.add_argument("--server-ip", required=True)
    export_client.add_argument("--server-port", type=int, default=5900)
    export_client.add_argument("--client-port", type=int)
    export_client.add_argument("--initial-key-hex",
        help="post-auth transport key (hex). Optional: defaults to the "
             "sibling '<pcap-stem>.key' file written by `iss --record`.")
    export_client.add_argument("--client-cbc-start-offset", type=int)
    export_client.add_argument("--server-cbc-start-offset", type=int)
    export_client.add_argument("--format", choices=["json", "markdown"], default="json")
    export_client.add_argument("--output")

    scan = sub.add_parser("scan-session")
    scan.add_argument("--pcap", required=True)
    scan.add_argument("--client-ip", required=True)
    scan.add_argument("--server-ip", required=True)
    scan.add_argument("--server-port", type=int, default=5900)
    scan.add_argument("--client-port", type=int)
    scan.add_argument("--initial-key-hex",
        help="post-auth transport key (hex). Optional: defaults to the "
             "sibling '<pcap-stem>.key' file written by `iss --record`.")
    scan.add_argument("--client-cbc-start-offset", type=int)
    scan.add_argument("--server-cbc-start-offset", type=int)
    scan.add_argument("--tail-seconds", default="3,5")
    scan.add_argument("--out-json")

    stats = sub.add_parser("stats")
    stats.add_argument("--records", required=True)

    render = sub.add_parser("render-replay")
    render.add_argument("--pcap", required=True)
    render.add_argument("--client-ip", required=True)
    render.add_argument("--server-ip", required=True)
    render.add_argument("--server-port", type=int, default=5900)
    render.add_argument("--client-port", type=int)
    render.add_argument("--initial-key-hex",
        help="post-auth transport key (hex). Optional: defaults to the "
             "sibling '<pcap-stem>.key' file written by `iss --record`.")
    render.add_argument("--output", required=True)
    render.add_argument("--client-cbc-start-offset", type=int)
    render.add_argument("--server-cbc-start-offset", type=int)
    render.add_argument("--start-at-seconds", type=float, default=0.0)
    render.add_argument("--max-seconds", type=float)
    render.add_argument("--emit-frame-ledger")

    analyze_udp = sub.add_parser("analyze-udp-media")
    analyze_udp.add_argument("--pcap", required=True)
    analyze_udp.add_argument("--client-ip", required=True)
    analyze_udp.add_argument("--server-ip", required=True)
    analyze_udp.add_argument("--server-port", type=int, default=5900)
    analyze_udp.add_argument("--client-port", type=int)
    analyze_udp.add_argument("--initial-key-hex",
        help="post-auth transport key (hex). Optional: defaults to the "
             "sibling '<pcap-stem>.key' file written by `iss --record`.")
    analyze_udp.add_argument("--client-cbc-start-offset", type=int)
    analyze_udp.add_argument("--server-cbc-start-offset", type=int)
    analyze_udp.add_argument("--media-ports", help="comma-separated UDP media ports (default: infer from stage1 0x3f2)")
    analyze_udp.add_argument("--port-count", type=int, default=2)
    analyze_udp.add_argument("--entropy-sample-bytes", type=int, default=1_000_000)
    analyze_udp.add_argument("--srtp-probe-packets", type=int, default=120)
    analyze_udp.add_argument("--disable-srtp-probe", action="store_true")
    analyze_udp.add_argument("--disable-media-decode", action="store_true",
                             help="skip the real SRTP decrypt + HEVC depayload of the dynamic media")
    analyze_udp.add_argument("--output")

    export_mp4 = sub.add_parser("export-media-mp4",
                                help="decrypt + depayload + decode the dynamic media to an MP4")
    export_mp4.add_argument("--pcap", required=True)
    export_mp4.add_argument("--client-ip", required=True)
    export_mp4.add_argument("--server-ip", required=True)
    export_mp4.add_argument("--server-port", type=int, default=5900)
    export_mp4.add_argument("--client-port", type=int)
    export_mp4.add_argument("--initial-key-hex",
        help="post-auth transport key (hex). Optional: defaults to the "
             "sibling '<pcap-stem>.key' file written by `iss --record`.")
    export_mp4.add_argument("--client-cbc-start-offset", type=int)
    export_mp4.add_argument("--server-cbc-start-offset", type=int)
    export_mp4.add_argument("--media-ports", help="comma-separated UDP media ports (default: infer from stage1 0x3f2)")
    export_mp4.add_argument("--port-count", type=int, default=2)
    export_mp4.add_argument("--video-port", type=int, help="force the video UDP port (default: auto-select)")
    export_mp4.add_argument("--fps", type=int, default=30)
    export_mp4.add_argument("--width", type=int, default=1920)
    export_mp4.add_argument("--height", type=int, default=1080)
    export_mp4.add_argument("--output", required=True)
    return parser


def cmd_list_streams(args: argparse.Namespace) -> int:
    segments = read_tcp_segments(args.pcap)
    for idx, summary in enumerate(summarize_conversations(segments), start=1):
        server = f"{summary.server_endpoint[0]}:{summary.server_endpoint[1]}" if summary.server_endpoint else "unknown"
        print(
            f"{idx:>3}  {summary.key.a_ip}:{summary.key.a_port} <-> {summary.key.b_ip}:{summary.key.b_port}  "
            f"packets={summary.packet_count} bytes={summary.payload_bytes} server={server}"
        )
    return 0


def resolve_session(args: argparse.Namespace) -> tuple[str, ReassembledStream, ReassembledStream, RekeyInfo, int, int]:
    convo = read_conversation_segments(
        args.pcap,
        args.client_ip,
        args.server_ip,
        args.server_port,
        client_port=getattr(args, "client_port", None),
        stop_after_seconds=None,
    )
    if not convo:
        raise SystemExit("no matching TCP conversation found")

    client_segments = [s for s in convo if s.src_ip == args.client_ip]
    server_segments = [s for s in convo if s.src_ip == args.server_ip and s.src_port == args.server_port]
    client_stream = reassemble_direction(client_segments, "client")
    server_stream = reassemble_direction(server_segments, "server")
    stream_id = f"{args.client_ip}->{args.server_ip}:{args.server_port}"

    rekey = find_rekey_info(server_stream, args.initial_key_hex)
    if rekey is None:
        hint = ""
        if getattr(args, "client_port", None) is None:
            hint = "; try --client-port <ephemeral_port> when multiple sessions share client/server IPs"
        raise SystemExit(f"failed to locate first EncodeEncryptionInfo in selected server stream{hint}")

    server_cbc_start = args.server_cbc_start_offset if args.server_cbc_start_offset is not None else rekey.server_offset + 52
    client_cbc_start = args.client_cbc_start_offset
    if client_cbc_start is None:
        client_cbc_start = find_client_cbc_start_by_preface(
            client_stream,
            rekey.timestamp_epoch,
            rekey.next_key_hex,
            rekey.next_iv_hex,
        )
    if client_cbc_start is None:
        client_cbc_start = find_client_cbc_start(client_stream, rekey.timestamp_epoch)
    if client_cbc_start is None:
        raise SystemExit("failed to infer client CBC start offset; supply --client-cbc-start-offset")
    return stream_id, client_stream, server_stream, rekey, client_cbc_start, server_cbc_start


def decode_records(
    args: argparse.Namespace,
    include_client: bool = True,
    lightweight: bool = False,
) -> list[DecodedRecord]:
    read_stop_after_seconds = None
    if lightweight and getattr(args, "max_seconds", None) is not None:
        read_stop_after_seconds = getattr(args, "start_at_seconds", 0.0) + args.max_seconds + 5.0
    stream_id, client_stream, server_stream, rekey, client_cbc_start, server_cbc_start = resolve_session(args)

    # Find all rekey events
    all_rekeys = find_all_rekey_info(server_stream, args.initial_key_hex)
    server_rekey_offsets = [(r.server_offset + 52, r.next_key_hex, r.next_iv_hex) for r in all_rekeys]

    records = []
    server_stop_after_epoch = None
    if getattr(args, "max_seconds", None) is not None:
        server_stop_after_epoch = rekey.timestamp_epoch + getattr(args, "start_at_seconds", 0.0) + args.max_seconds + 10.0
    if include_client:
        records.extend(
            decode_cbc_records(
                stream_id=stream_id,
                direction="client",
                stream=client_stream,
                start_offset=client_cbc_start,
                key_hex=rekey.next_key_hex,
                iv_hex=rekey.next_iv_hex,
                base_seq=rekey.counter - 1 if rekey.counter > 0 else None,
                include_plain_hex=not lightweight,
                include_parsed=not lightweight,
                rekey_offsets=server_rekey_offsets,
            )
        )
    records.extend(
        decode_cbc_records(
            stream_id=stream_id,
            direction="server",
            stream=server_stream,
            start_offset=server_cbc_start,
            key_hex=rekey.next_key_hex,
            iv_hex=rekey.next_iv_hex,
            base_seq=rekey.counter - 1 if rekey.counter > 0 else None,
            stop_after_epoch=server_stop_after_epoch,
            include_plain_hex=not lightweight,
            include_parsed=not lightweight,
            rekey_offsets=server_rekey_offsets,
        )
    )
    if include_client:
        records.sort(key=lambda r: (r.timestamp_epoch or 0.0, r.direction, r.record_index))
    return records


def iter_server_records_for_replay(args: argparse.Namespace):
    read_stop_after_seconds = None
    if getattr(args, "max_seconds", None) is not None:
        read_stop_after_seconds = getattr(args, "start_at_seconds", 0.0) + args.max_seconds + 5.0
    stream_id = f"{args.client_ip}->{args.server_ip}:{args.server_port}"
    if getattr(args, "client_port", None) is not None:
        stream_id += f" (client_port={args.client_port})"

    next_key_hex: str | None = None
    next_iv_hex: str | None = None
    rekey_ts: float | None = None
    server_stop_after_epoch: float | None = None
    cbc_started = False
    record_index = 0

    server_buffer = bytearray()
    server_meta: deque[list[float | int]] = deque()
    server_start = 0
    pending: dict[int, tuple[bytes, float, int]] = {}
    wrap_base = 0
    prev_raw_seq: int | None = None
    prev_ext_seq: int | None = None
    next_seq: int | None = None

    def buffer_front_meta() -> tuple[float | None, int | None]:
        if not server_meta:
            return None, None
        return float(server_meta[0][1]), int(server_meta[0][2])

    def available_len() -> int:
        return len(server_buffer) - server_start

    def consume_buffer(n: int) -> None:
        nonlocal server_start
        if n <= 0:
            return
        server_start += n
        remaining = n
        while remaining > 0 and server_meta:
            chunk_len = int(server_meta[0][0])
            if remaining >= chunk_len:
                remaining -= chunk_len
                server_meta.popleft()
            else:
                server_meta[0][0] = chunk_len - remaining
                remaining = 0
        if server_start > 4 * 1024 * 1024 and server_start > len(server_buffer) // 2:
            del server_buffer[:server_start]
            server_start = 0

    def append_piece(payload: bytes, ts: float, frame_number: int) -> None:
        if not payload:
            return
        server_buffer.extend(payload)
        if server_meta and server_meta[-1][1] == ts and server_meta[-1][2] == frame_number:
            server_meta[-1][0] = int(server_meta[-1][0]) + len(payload)
        else:
            server_meta.append([len(payload), ts, frame_number])

    def timestamp_at_offset(offset: int) -> tuple[float | None, int | None]:
        remaining = offset
        for chunk_len, ts, frame_number in server_meta:
            if remaining < int(chunk_len):
                return float(ts), int(frame_number)
            remaining -= int(chunk_len)
        return buffer_front_meta()

    def flush_contiguous(segment) -> None:
        nonlocal wrap_base, prev_raw_seq, prev_ext_seq, next_seq
        raw_seq = segment.seq
        if prev_raw_seq is not None and raw_seq < prev_raw_seq and prev_raw_seq - raw_seq > 0x80000000:
            wrap_base += 0x100000000
        ext_seq = raw_seq + wrap_base
        while prev_ext_seq is not None and ext_seq + 0x80000000 < prev_ext_seq:
            ext_seq += 0x100000000
        prev_raw_seq = raw_seq
        prev_ext_seq = ext_seq
        if next_seq is None:
            next_seq = ext_seq
        payload = segment.payload
        if ext_seq + len(payload) <= next_seq:
            return
        if ext_seq < next_seq:
            trim = next_seq - ext_seq
            payload = payload[trim:]
            ext_seq = next_seq
        existing = pending.get(ext_seq)
        if existing is None or len(payload) > len(existing[0]):
            pending[ext_seq] = (payload, segment.timestamp_epoch, segment.frame_number)
        while next_seq in pending:
            piece, ts, frame_number = pending.pop(next_seq)
            append_piece(piece, ts, frame_number)
            next_seq += len(piece)

    def maybe_start_cbc() -> None:
        nonlocal next_key_hex, next_iv_hex, rekey_ts, server_stop_after_epoch, cbc_started
        if cbc_started or available_len() < 52:
            return
        limit = available_len() - 51
        for off in range(limit):
            abs_off = server_start + off
            if server_buffer[abs_off] != 0:
                continue
            if int.from_bytes(server_buffer[abs_off + 12 : abs_off + 16], "big") != 0x44F:
                continue
            block1 = bytes(server_buffer[abs_off + 20 : abs_off + 36])
            block2 = bytes(server_buffer[abs_off + 36 : abs_off + 52])
            next_key_hex = decrypt_ecb_block(args.initial_key_hex, block1).hex()
            next_iv_hex = decrypt_ecb_block(args.initial_key_hex, block2).hex()
            rekey_ts, _ = timestamp_at_offset(off)
            if rekey_ts is not None and getattr(args, "max_seconds", None) is not None:
                server_stop_after_epoch = rekey_ts + getattr(args, "start_at_seconds", 0.0) + args.max_seconds + 10.0
            consume_buffer(off + 52)
            cbc_started = True
            return

    for segment in iter_conversation_segments(
        args.pcap,
        args.client_ip,
        args.server_ip,
        args.server_port,
        client_port=getattr(args, "client_port", None),
        stop_after_seconds=read_stop_after_seconds,
    ):
        if segment.src_ip != args.server_ip or segment.src_port != args.server_port:
            continue
        flush_contiguous(segment)
        if not cbc_started:
            maybe_start_cbc()
            if not cbc_started:
                continue

        while available_len() >= 2:
            cipher_len = int.from_bytes(server_buffer[server_start : server_start + 2], "big")
            if cipher_len == 0 or cipher_len % 16 != 0:
                return
            if available_len() < 2 + cipher_len:
                break
            timestamp_epoch, frame_number = buffer_front_meta()
            if server_stop_after_epoch is not None and timestamp_epoch is not None and timestamp_epoch > server_stop_after_epoch:
                return
            ciphertext = bytes(server_buffer[server_start + 2 : server_start + 2 + cipher_len])
            plain = decrypt_cbc_block(next_key_hex, next_iv_hex, ciphertext)
            next_iv_hex = ciphertext[-16:].hex()
            consume_buffer(2 + cipher_len)
            if len(plain) < 2:
                return
            body_len = int.from_bytes(plain[0:2], "big")
            body_end = 2 + body_len
            if body_end > len(plain):
                return
            body = plain[2:body_end]
            record_index += 1
            msg_id = body[0] if body else None
            yield DecodedRecord(
                stream_id=stream_id,
                record_index=record_index,
                direction="server",
                timestamp_epoch=timestamp_epoch,
                frame_number=frame_number,
                tcp_offset=0,
                cipher_len=cipher_len,
                msg_id=msg_id,
                msg_name=f"server_extension_0x{msg_id:02x}" if msg_id not in {0} and msg_id is not None else "FramebufferUpdate",
                record_kind=f"server_extension_0x{msg_id:02x}" if msg_id not in {0} and msg_id is not None else "FramebufferUpdate",
                plain_len=len(plain),
                plain_hex="",
                body_bytes=body,
                filler_hex="",
                trailing_sha1_hex="",
                parsed={},
            )


def cmd_decode(args: argparse.Namespace) -> int:
    records = decode_records(args)
    if args.output:
        write_jsonl(args.output, records)
        print(args.output)
        return 0
    for record in records:
        print(
            json.dumps(
                {
                    "timestamp_epoch": record.timestamp_epoch,
                    "direction": record.direction,
                    "record_index": record.record_index,
                    "msg_id": record.msg_id,
                    "msg_name": record.msg_name,
                    "record_kind": record.record_kind,
                    "frame_number": record.frame_number,
                },
                sort_keys=True,
            )
        )
    return 0


def load_records(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def find_client_cleartext_prelude(client_stream: ReassembledStream) -> list[tuple[int, bytes]]:
    data = client_stream.data
    prelude_len = 66 + 12 + 4
    if len(data) < prelude_len:
        return []
    for off in range(0, len(data) - prelude_len + 1):
        viewer = data[off : off + 66]
        enc = data[off + 66 : off + 78]
        mode = data[off + 78 : off + 82]
        if viewer[:1] != b"\x21" or enc[:1] != b"\x12" or mode[:1] != b"\x0a":
            continue
        viewer_name, viewer_parsed = parse_client_message(viewer)
        enc_name, enc_parsed = parse_client_message(enc)
        mode_name, mode_parsed = parse_client_message(mode)
        if viewer_name != "ViewerInfo" or enc_name != "SetEncryptionMessage" or mode_name != "SetModeMessage":
            continue
        if enc_parsed.get("form") != "command_1_long":
            continue
        if mode_parsed.get("mode") != 1:
            continue
        return [(off, viewer), (off + 66, enc), (off + 78, mode)]
    return []


def cleartext_client_message_len(data: bytes, off: int) -> int | None:
    if off >= len(data):
        return None
    msg_id = data[off]
    if msg_id == 0x12:
        if off + 8 <= len(data) and data[off + 2 : off + 4] == b"\x00\x02":
            return 8
        if off + 12 <= len(data):
            return 12
        return None
    if msg_id == 0x0A:
        return 4 if off + 4 <= len(data) else None
    if msg_id == 0x1D:
        body_len = int.from_bytes(data[off + 2 : off + 4], "big") if off + 4 <= len(data) else None
        if body_len is None:
            return None
        total = 4 + body_len
        return total if off + total <= len(data) else None
    if msg_id == 0x02:
        count = int.from_bytes(data[off + 2 : off + 4], "big") if off + 4 <= len(data) else None
        if count is None:
            return None
        total = 4 + count * 4
        return total if off + total <= len(data) else None
    return None


def find_client_cleartext_preface(client_stream: ReassembledStream) -> list[tuple[int, bytes]]:
    prelude = find_client_cleartext_prelude(client_stream)
    if not prelude:
        return []
    rows = list(prelude)
    data = client_stream.data
    off = prelude[-1][0] + len(prelude[-1][1])
    while off < len(data):
        msg_len = cleartext_client_message_len(data, off)
        if msg_len is None:
            break
        body = data[off : off + msg_len]
        rows.append((off, body))
        if body[:1] == b"\x12" and len(body) == 8:
            break
        off += msg_len
    return rows


def make_request_row(
    request_index: int,
    transport: str,
    tcp_offset: int,
    timestamp_epoch: float | None,
    frame_number: int | None,
    body: bytes,
    warnings: list[str] | None = None,
) -> dict[str, object]:
    msg_id = body[0] if body else None
    msg_name, parsed = parse_client_message(body)
    return {
        "request_index": request_index,
        "transport": transport,
        "timestamp_epoch": timestamp_epoch,
        "frame_number": frame_number,
        "tcp_offset": tcp_offset,
        "msg_id": msg_id,
        "msg_name": msg_name,
        "plain_len": len(body),
        "plain_hex": body.hex(),
        "parsed": parsed,
        "warnings": warnings or [],
    }


def collect_client_request_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    stream_id, client_stream, server_stream, rekey, client_cbc_start, server_cbc_start = resolve_session(args)
    rows: list[dict[str, object]] = []

    cleartext_preface = find_client_cleartext_preface(client_stream)

    for offset, body in cleartext_preface:
        rows.append(
            make_request_row(
                request_index=0,
                transport="cleartext",
                tcp_offset=offset,
                timestamp_epoch=client_stream.timestamp_for_offset(offset),
                frame_number=client_stream.frame_for_offset(offset),
                body=body,
            )
        )

    if client_cbc_start >= 8 and not any(offset == client_cbc_start - 8 for offset, _ in cleartext_preface):
        body = client_stream.data[client_cbc_start - 8 : client_cbc_start]
        if body[:1] == b"\x12":
            rows.append(
                make_request_row(
                    request_index=0,
                    transport="cleartext",
                    tcp_offset=client_cbc_start - 8,
                    timestamp_epoch=client_stream.timestamp_for_offset(client_cbc_start - 8),
                    frame_number=client_stream.frame_for_offset(client_cbc_start - 8),
                    body=body,
                )
            )

    cbc_records = decode_cbc_records(
        stream_id=stream_id,
        direction="client",
        stream=client_stream,
        start_offset=client_cbc_start,
        key_hex=rekey.next_key_hex,
        iv_hex=rekey.next_iv_hex,
        base_seq=rekey.counter - 1 if rekey.counter > 0 else None,
        include_plain_hex=True,
        include_parsed=True,
    )
    for record in cbc_records:
        rows.append(
            {
                "request_index": 0,
                "transport": "cbc",
                "timestamp_epoch": record.timestamp_epoch,
                "frame_number": record.frame_number,
                "tcp_offset": record.tcp_offset,
                "msg_id": record.msg_id,
                "msg_name": record.msg_name,
                "plain_len": len(record.body_bytes),
                "plain_hex": record.body_bytes.hex(),
                "parsed": record.parsed,
                "warnings": list(record.warnings),
            }
        )

    rows.sort(key=lambda row: ((row.get("timestamp_epoch") or 0.0), int(row.get("tcp_offset") or 0)))
    for idx, row in enumerate(rows, start=1):
        row["request_index"] = idx
    return rows


def render_client_requests_markdown(rows: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(
            f"## [{row['request_index']}] {row['msg_name']} "
            f"(msg_id=0x{(row['msg_id'] or 0):02x}, transport={row['transport']}, ts={row['timestamp_epoch']})"
        )
        lines.append(f"- frame_number = {row['frame_number']}")
        lines.append(f"- tcp_offset = {row['tcp_offset']}")
        lines.append(f"- plain_len = {row['plain_len']}")
        parsed = row.get("parsed") or {}
        for key in sorted(parsed):
            lines.append(f"- {key} = {json.dumps(parsed[key], sort_keys=True)}")
        warnings = row.get("warnings") or []
        for warning in warnings:
            lines.append(f"- warning = {warning}")
        lines.append(f"- raw = {row['plain_hex']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def parse_tail_windows(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    return [value for value in values if value > 0.0] or [3.0, 5.0]


def summarize_event_times(times: list[float]) -> dict[str, object]:
    if len(times) < 2:
        return {
            "event_count": len(times),
            "delta_count": 0,
            "rate_hz": 0.0,
        }
    deltas_ms = [(times[i] - times[i - 1]) * 1000.0 for i in range(1, len(times))]
    elapsed = times[-1] - times[0]
    rate_hz = (len(times) / elapsed) if elapsed > 0 else 0.0
    return {
        "event_count": len(times),
        "delta_count": len(deltas_ms),
        "avg_ms": sum(deltas_ms) / len(deltas_ms),
        "median_ms": median(deltas_ms),
        "min_ms": min(deltas_ms),
        "max_ms": max(deltas_ms),
        "rate_hz": rate_hz,
    }


def summarize_tail_event_times(times: list[float], tail_windows: list[float]) -> dict[str, dict[str, object]]:
    if not times:
        return {str(window): summarize_event_times([]) for window in tail_windows}
    end_ts = times[-1]
    out: dict[str, dict[str, object]] = {}
    for window in tail_windows:
        start_ts = end_ts - window
        tail = [ts for ts in times if ts >= start_ts]
        out[str(window)] = summarize_event_times(tail)
    return out


def group_burst_times(times: list[float], threshold_seconds: float = 0.002) -> list[float]:
    if not times:
        return []
    groups = [times[0]]
    last_ts = times[0]
    for ts in times[1:]:
        if ts - last_ts > threshold_seconds:
            groups.append(ts)
        last_ts = ts
    return groups


def build_scan_summary(args: argparse.Namespace) -> dict[str, object]:
    tail_windows = parse_tail_windows(args.tail_seconds)
    records = decode_records(args, include_client=True, lightweight=True)

    client_counts: Counter[str] = Counter()
    client_times_by_name: dict[str, list[float]] = {}
    focus_names = {
        "FramebufferUpdateRequest",
        "AutoFrameBufferUpdate",
        "client_extension_0x10",
        "AppleScaleFactorMessage",
        "AutoPasteboardCommand",
        "SetDisplayMessage",
        "SetEncodings",
    }

    server_fbupdate_times: list[float] = []
    server_pixel_update_times: list[float] = []
    server_burst_times: list[float] = []
    rect_encoding_counts: Counter[str] = Counter()
    full_screen_pixel_updates = 0
    backing_width = 0
    backing_height = 0

    for record in records:
        if record.direction != "client":
            continue
        client_counts[record.msg_name] += 1
        if record.timestamp_epoch is not None:
            client_times_by_name.setdefault(record.msg_name, []).append(record.timestamp_epoch)

    for record in iter_assembled_server_records(r for r in records if r.direction == "server"):
        if record.msg_name != "FramebufferUpdate" or not record.body_bytes or record.timestamp_epoch is None:
            continue

        server_fbupdate_times.append(record.timestamp_epoch)
        rects = parse_framebuffer_rects(record.body_bytes)
        pixel_affecting = False

        for rect in rects:
            if rect.encoding == 0x00000451:
                size = display_layout_backing_size(rect)
                if size is not None:
                    backing_width, backing_height = size
            elif rect.encoding == 0xFFFFFF21 and rect.width > 0 and rect.height > 0:
                backing_width, backing_height = rect.width, rect.height

            rect_encoding_counts[f"0x{rect.encoding:08x}:{rectangle_encoding_name(rect.encoding)}"] += 1

            if rect.encoding in {0x00000000, 0x00000006, 0x00000010}:
                pixel_affecting = True
                if (
                    backing_width > 0
                    and backing_height > 0
                    and rect.x == 0
                    and rect.y == 0
                    and rect.width == backing_width
                    and rect.height == backing_height
                ):
                    full_screen_pixel_updates += 1

        if pixel_affecting:
            server_pixel_update_times.append(record.timestamp_epoch)

    server_burst_times = group_burst_times(server_pixel_update_times)

    client_focus: dict[str, object] = {}
    for name in sorted(set(client_counts) | focus_names):
        times = client_times_by_name.get(name, [])
        if not times and name not in focus_names:
            continue
        client_focus[name] = {
            "count": client_counts.get(name, 0),
            "first_ts": times[0] if times else None,
            "last_ts": times[-1] if times else None,
            "cadence": summarize_event_times(times),
            "tail": summarize_tail_event_times(times, tail_windows),
        }

    return {
        "stream": {
            "pcap": args.pcap,
            "client_ip": args.client_ip,
            "server_ip": args.server_ip,
            "server_port": args.server_port,
            "client_port": getattr(args, "client_port", None),
        },
        "tail_windows_seconds": tail_windows,
        "record_count": len(records),
        "server": {
            "backing_size": {
                "width": backing_width,
                "height": backing_height,
            },
            "logical_framebuffer_updates": {
                "cadence": summarize_event_times(server_fbupdate_times),
                "tail": summarize_tail_event_times(server_fbupdate_times, tail_windows),
            },
            "pixel_updates": {
                "cadence": summarize_event_times(server_pixel_update_times),
                "tail": summarize_tail_event_times(server_pixel_update_times, tail_windows),
            },
            "pixel_bursts": {
                "cadence": summarize_event_times(server_burst_times),
                "tail": summarize_tail_event_times(server_burst_times, tail_windows),
            },
            "full_screen_pixel_updates": full_screen_pixel_updates,
            "rect_encoding_counts": dict(rect_encoding_counts.most_common()),
        },
        "client": {
            "message_counts": dict(client_counts.most_common()),
            "focus_messages": client_focus,
        },
    }


def print_scan_summary(summary: dict[str, object]) -> None:
    server = summary["server"]
    client = summary["client"]
    logical = server["logical_framebuffer_updates"]["cadence"]
    pixel = server["pixel_updates"]["cadence"]
    bursts = server["pixel_bursts"]["cadence"]
    tail_windows = summary["tail_windows_seconds"]
    pixel_tail = server["pixel_updates"]["tail"]

    print(f"records={summary['record_count']}")
    print(
        "server logical updates:"
        f" count={logical.get('event_count', 0)}"
        f" rate~={logical.get('rate_hz', 0.0):.2f}/s"
        f" median={logical.get('median_ms', 0.0):.3f}ms"
    )
    print(
        "server pixel updates:"
        f" count={pixel.get('event_count', 0)}"
        f" rate~={pixel.get('rate_hz', 0.0):.2f}/s"
        f" median={pixel.get('median_ms', 0.0):.3f}ms"
    )
    print(
        "server pixel bursts:"
        f" count={bursts.get('event_count', 0)}"
        f" rate~={bursts.get('rate_hz', 0.0):.2f}/s"
        f" median={bursts.get('median_ms', 0.0):.3f}ms"
    )
    for window in tail_windows:
        tail = pixel_tail.get(str(window), {})
        print(
            f"tail {window:g}s pixel updates:"
            f" count={tail.get('event_count', 0)}"
            f" rate~={tail.get('rate_hz', 0.0):.2f}/s"
            f" median={tail.get('median_ms', 0.0):.3f}ms"
        )
    print(f"full-screen pixel updates={server['full_screen_pixel_updates']}")
    print("client focus:")
    for name, details in client["focus_messages"].items():
        if details["count"] == 0:
            continue
        cadence = details["cadence"]
        print(
            f"  {name}: count={details['count']}"
            f" first={details['first_ts']}"
            f" last={details['last_ts']}"
            f" rate~={cadence.get('rate_hz', 0.0):.2f}/s"
        )


def cmd_stats(args: argparse.Namespace) -> int:
    rows = load_records(args.records)
    counts = Counter(str(row.get("msg_name")) for row in rows)
    by_dir = Counter((str(row.get("direction")), str(row.get("msg_name"))) for row in rows)
    print(f"records={len(rows)}")
    print("message_counts:")
    for name, count in counts.most_common():
        print(f"  {name}: {count}")
    print("direction_counts:")
    for (direction, name), count in sorted(by_dir.items()):
        print(f"  {direction} {name}: {count}")
    return 0


def cmd_export_client_requests(args: argparse.Namespace) -> int:
    rows = collect_client_request_rows(args)
    if args.format == "markdown":
        rendered = render_client_requests_markdown(rows)
    else:
        rendered = json.dumps(rows, indent=2)
    if args.output:
        Path(args.output).write_text(rendered)
        print(args.output)
        return 0
    print(rendered)
    return 0


def cmd_scan_session(args: argparse.Namespace) -> int:
    summary = build_scan_summary(args)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(args.out_json)
        return 0
    print_scan_summary(summary)
    return 0


def parse_ports_csv(raw: str) -> list[int]:
    ports: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        ports.append(int(part))
    return sorted(set(ports))


def infer_media_ports_from_records(records: list[DecodedRecord], default_count: int) -> list[int]:
    stage1_ports: list[int] = []
    stream_counts: list[int] = []
    for record in records:
        if record.direction != "server" or record.msg_name != "FramebufferUpdate":
            continue
        first_rect = (record.parsed or {}).get("first_rect")
        if not isinstance(first_rect, dict):
            continue
        if int(first_rect.get("encoding") or -1) != 0x3F2:
            continue
        branch = first_rect.get("media_init_branch")
        if branch != "stage1":
            continue
        port = first_rect.get("media_init_u16_10")
        count = first_rect.get("media_init_u32_12")
        if isinstance(port, int) and 0 < port <= 65535:
            stage1_ports.append(port)
        if isinstance(count, int) and 0 < count < 16:
            stream_counts.append(count)
    if not stage1_ports:
        return []
    base_port = stage1_ports[-1]
    inferred_count = max(default_count, 2)
    if stream_counts:
        # stage1 count appears to describe media streams beyond base; keep at least audio+video.
        inferred_count = max(inferred_count, stream_counts[-1] + 1)
    return [base_port + i for i in range(inferred_count)]


def latest_client_media_options(records: list[DecodedRecord]) -> dict[str, object] | None:
    latest: DecodedRecord | None = None
    for record in records:
        if record.direction != "client" or record.msg_id != 28:
            continue
        parsed = record.parsed or {}
        if parsed.get("mode_family") != "adaptive_media":
            continue
        if latest is None:
            latest = record
            continue
        if (record.timestamp_epoch or 0.0, record.record_index) >= (
            latest.timestamp_epoch or 0.0,
            latest.record_index,
        ):
            latest = record
    if latest is None:
        return None
    parsed = latest.parsed or {}
    return {
        "record_index": latest.record_index,
        "timestamp_epoch": latest.timestamp_epoch,
        "plain_len": len(latest.body_bytes),
        "message_version_be16": parsed.get("message_version_be16"),
        "message_flags_be32": parsed.get("message_flags_be32"),
        "message_flag_names": parsed.get("message_flag_names"),
        "audio_offer_len_be16": parsed.get("audio_offer_len_be16"),
        "video1_offer_len_be16": parsed.get("video1_offer_len_be16"),
        "video2_offer_len_be16": parsed.get("video2_offer_len_be16"),
        "session_id_hex": parsed.get("session_id_hex"),
        "call_id": parsed.get("call_id"),
        "negotiator_mode": parsed.get("negotiator_mode"),
        "stream_key_blocks": parsed.get("stream_key_blocks"),
    }


def cmd_analyze_udp_media(args: argparse.Namespace) -> int:
    records = decode_records(args, include_client=True, lightweight=False)
    media_ports: list[int]
    if args.media_ports:
        media_ports = parse_ports_csv(args.media_ports)
    else:
        media_ports = infer_media_ports_from_records(records, default_count=max(1, int(args.port_count)))
    if not media_ports:
        raise SystemExit("unable to infer media ports; pass --media-ports")

    latest_options = latest_client_media_options(records)
    stream_key_blocks = None
    if isinstance(latest_options, dict):
        blocks = latest_options.get("stream_key_blocks")
        if isinstance(blocks, list):
            stream_key_blocks = blocks

    udp_summary = analyze_udp_media_streams(
        pcap_path=args.pcap,
        client_ip=args.client_ip,
        server_ip=args.server_ip,
        ports=media_ports,
        entropy_sample_bytes=max(1, int(args.entropy_sample_bytes)),
        stream_key_blocks=stream_key_blocks,
        srtp_probe_packets=max(1, int(args.srtp_probe_packets)),
        decrypt_media=not args.disable_media_decode,
        run_probe=not args.disable_srtp_probe,
    )
    out: dict[str, Any] = {
        "stream": {
            "pcap": args.pcap,
            "client_ip": args.client_ip,
            "server_ip": args.server_ip,
            "server_port": args.server_port,
            "client_port": getattr(args, "client_port", None),
            "media_ports": media_ports,
        },
        "tcp_adaptive": {
            "latest_client_media_stream_options": latest_options,
        },
        "udp_media": udp_summary,
    }
    rendered = json.dumps(out, indent=2)
    if args.output:
        Path(args.output).write_text(rendered)
        print(args.output)
        return 0
    print(rendered)
    return 0


def cmd_export_media_mp4(args: argparse.Namespace) -> int:
    from libs.hevc_render import render_session_mp4, select_video_stream

    records = decode_records(args, include_client=True, lightweight=False)
    if args.media_ports:
        media_ports = parse_ports_csv(args.media_ports)
    else:
        media_ports = infer_media_ports_from_records(records, default_count=max(1, int(args.port_count)))
    if not media_ports:
        raise SystemExit("unable to infer media ports; pass --media-ports")

    latest_options = latest_client_media_options(records)
    blocks = latest_options.get("stream_key_blocks") if isinstance(latest_options, dict) else None
    if not isinstance(blocks, list) or not blocks:
        raise SystemExit("no media key blocks recovered from the TCP session (check --initial-key-hex)")

    if args.video_port is not None:
        # caller forced a port; pick the video key block that authenticates it
        sel = select_video_stream(args.pcap, args.server_ip, [args.video_port], blocks)
    else:
        sel = select_video_stream(args.pcap, args.server_ip, media_ports, blocks)
    if sel is None:
        raise SystemExit("could not find a decryptable video stream on the media ports")

    print(f"video on UDP {sel['port']} via {sel['stream']}.{sel['label']} "
          f"(sample auth {sel['ok']}); decoding -> {args.output}")
    stats = render_session_mp4(
        pcap_path=args.pcap,
        server_ip=args.server_ip,
        video_port=int(sel["port"]),
        key_blob=sel["blob"],
        out_path=args.output,
        fps=max(1, int(args.fps)),
        out_w=max(2, int(args.width)),
        out_h=max(2, int(args.height)),
    )
    print(json.dumps(stats, indent=2))
    print(args.output)
    return 0


def iter_assembled_server_records(records):
    pending_record: DecodedRecord | None = None
    records_iter = iter(records)

    while True:
        if pending_record is not None:
            record = pending_record
            pending_record = None
        else:
            try:
                record = next(records_iter)
            except StopIteration:
                return
        if record.msg_name != "FramebufferUpdate" or not record.body_bytes:
            yield record
            continue
        combined = bytearray(record.body_bytes)
        source_indices = [record.record_index]
        warnings = list(record.warnings)
        expected = framebuffer_update_expected_len(record.body_bytes)
        last_timestamp = record.timestamp_epoch
        last_frame_number = record.frame_number

        while expected is None or len(combined) < expected:
            try:
                nxt = next(records_iter)
            except StopIteration:
                if expected is not None:
                    warnings.append(
                        f"incomplete server message assembly: expected {expected} body bytes, got {len(combined)}"
                    )
                else:
                    warnings.append("could not determine full FramebufferUpdate length before end of stream")
                break
            if nxt.direction != "server":
                pending_record = nxt
                break
            if nxt.body_bytes:
                combined.extend(nxt.body_bytes)
                source_indices.append(nxt.record_index)
                if nxt.timestamp_epoch is not None:
                    last_timestamp = nxt.timestamp_epoch
                if nxt.frame_number is not None:
                    last_frame_number = nxt.frame_number
                if expected is None:
                    expected = framebuffer_update_expected_len(bytes(combined))
        assembled_len = expected if expected is not None and len(combined) >= expected else len(combined)
        merged = DecodedRecord(
            stream_id=record.stream_id,
            record_index=record.record_index,
            direction=record.direction,
            timestamp_epoch=record.timestamp_epoch,
            frame_number=last_frame_number,
            tcp_offset=record.tcp_offset,
            cipher_len=record.cipher_len,
            msg_id=record.msg_id,
            msg_name=record.msg_name,
            record_kind=record.record_kind,
            plain_len=record.plain_len,
            plain_hex=combined[:assembled_len].hex(),
            body_bytes=bytes(combined[:assembled_len]),
            filler_hex=record.filler_hex,
            trailing_sha1_hex=record.trailing_sha1_hex,
            parsed={
                **record.parsed,
                "assembled_from_record_indices": source_indices,
                "assembled_record_count": len(source_indices),
                "assembled_last_timestamp_epoch": last_timestamp,
                "assembled_complete": expected is not None and len(combined) >= expected,
                "assembled_expected_body_len": expected,
            },
            warnings=warnings,
        )
        yield merged


def build_presentation_frames(
    records,
    start_at_seconds: float,
    max_seconds: float | None,
    writer: ExactTimingReplayWriter | None = None,
) -> tuple[list[PresentationFrame], float]:
    fb = Framebuffer()
    zdecoder = ZlibStreamDecoder()
    frames: list[PresentationFrame] = []
    first_ts: float | None = None
    last_ts: float | None = None

    for record in iter_assembled_server_records(records):
        if record.direction != "server" or record.msg_name != "FramebufferUpdate":
            continue
        if first_ts is None:
            first_ts = record.timestamp_epoch or 0.0
        ts = record.timestamp_epoch or 0.0
        last_ts = ts
        rel = ts - first_ts
        if rel < start_at_seconds:
            # still need to apply state so later frames are correct
            for rect in parse_framebuffer_rects(record.body_bytes):
                if rect.encoding == 0x00000451:
                    size = display_layout_backing_size(rect)
                    if size is not None:
                        fb.ensure_size(*size)
                elif rect.encoding == 0xFFFFFF21:
                    if rect.width > 0 and rect.height > 0:
                        fb.ensure_size(rect.width, rect.height)
                elif rect.encoding == 0x00000000:
                    fb.apply_raw_rect(rect.x, rect.y, rect.width, rect.height, decode_raw_pixels(rect))
                elif rect.encoding == 0x00000006:
                    try:
                        pixels = zdecoder.decode_rect(rect)
                    except Exception:
                        continue
                    fb.apply_raw_rect(rect.x, rect.y, rect.width, rect.height, pixels)
            continue
        if max_seconds is not None and rel > start_at_seconds + max_seconds:
            break

        changed = False
        for rect in parse_framebuffer_rects(record.body_bytes):
            if rect.encoding == 0x00000451:
                size = display_layout_backing_size(rect)
                if size is not None:
                    fb.ensure_size(*size)
            elif rect.encoding == 0xFFFFFF21:
                if rect.width > 0 and rect.height > 0:
                    fb.ensure_size(rect.width, rect.height)
            elif rect.encoding == 0x00000000:
                fb.apply_raw_rect(rect.x, rect.y, rect.width, rect.height, decode_raw_pixels(rect))
                changed = True
            elif rect.encoding == 0x00000006:
                try:
                    pixels = zdecoder.decode_rect(rect)
                except Exception:
                    continue
                fb.apply_raw_rect(rect.x, rect.y, rect.width, rect.height, pixels)
                changed = True
        if changed and fb.width > 0 and fb.height > 0:
            source_record_indices = record.parsed.get("assembled_from_record_indices")
            if not isinstance(source_record_indices, list):
                source_record_indices = [record.record_index]
            frames.append(
                PresentationFrame(
                    frame_index=len(frames),
                    pts_epoch=ts,
                    width=fb.width,
                    height=fb.height,
                    source_record_indices=source_record_indices,
                )
            )
            if writer is not None:
                writer.add_frame(
                    frame_index=frames[-1].frame_index,
                    pts_epoch=ts,
                    width=fb.width,
                    height=fb.height,
                    bgra=fb.snapshot_bgra(),
                )
    if first_ts is None:
        raise SystemExit("no server FramebufferUpdate records available for replay")
    end_pts_epoch = last_ts if last_ts is not None else first_ts
    return frames, end_pts_epoch


def cmd_render_replay(args: argparse.Namespace) -> int:
    writer = ExactTimingReplayWriter(args.output)
    try:
        frames, end_pts_epoch = build_presentation_frames(
            iter_server_records_for_replay(args),
            start_at_seconds=args.start_at_seconds,
            max_seconds=args.max_seconds,
            writer=writer,
        )
        if not frames:
            raise SystemExit("no presentation frames could be reconstructed from the selected trace window")
        if args.emit_frame_ledger:
            write_frame_ledger_with_end(args.emit_frame_ledger, frames, end_pts_epoch)
        writer.finish(end_pts_epoch=end_pts_epoch)
        print(args.output)
        return 0
    finally:
        writer.close()


def resolve_initial_key(args: argparse.Namespace) -> None:
    """Fill args.initial_key_hex from the sibling '<pcap-stem>.key' file when
    not given explicitly. `iss --record session.pcap` writes session.key next
    to it, so the common case needs no --initial-key-hex at all."""
    if getattr(args, "initial_key_hex", None):
        return
    sibling = Path(args.pcap).with_suffix(".key")
    if sibling.exists():
        args.initial_key_hex = sibling.read_text(encoding="ascii").strip()
        print(f"using transport key from {sibling}")
        return
    raise SystemExit(
        f"no transport key: pass --initial-key-hex, or place the key next to "
        f"the pcap at {sibling} (iss --record writes it there automatically)"
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    # Commands that decrypt need the post-auth key; auto-load the sibling
    # .key file when --initial-key-hex isn't supplied.
    if args.cmd in {"decode", "export-client-requests", "scan-session",
                    "render-replay", "analyze-udp-media", "export-media-mp4"}:
        resolve_initial_key(args)
    if args.cmd == "list-streams":
        return cmd_list_streams(args)
    if args.cmd == "decode":
        return cmd_decode(args)
    if args.cmd == "export-client-requests":
        return cmd_export_client_requests(args)
    if args.cmd == "scan-session":
        return cmd_scan_session(args)
    if args.cmd == "stats":
        return cmd_stats(args)
    if args.cmd == "render-replay":
        return cmd_render_replay(args)
    if args.cmd == "analyze-udp-media":
        return cmd_analyze_udp_media(args)
    if args.cmd == "export-media-mp4":
        return cmd_export_media_mp4(args)
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
