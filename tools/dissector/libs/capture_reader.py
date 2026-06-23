from __future__ import annotations

import socket
from pathlib import Path

from scapy.all import IP, IPv6, TCP, PcapReader  # type: ignore[import-untyped]
from scapy.utils import RawPcapNgReader, RawPcapReader  # type: ignore[import-untyped]

from .types import PacketSegment


def _packet_segment_from_pkt(frame_number: int, pkt) -> PacketSegment | None:
    if TCP not in pkt:
        return None
    tcp = pkt[TCP]
    payload = bytes(tcp.payload)
    if not payload:
        return None
    if IP in pkt:
        src_ip = str(pkt[IP].src)
        dst_ip = str(pkt[IP].dst)
    elif IPv6 in pkt:
        src_ip = str(pkt[IPv6].src)
        dst_ip = str(pkt[IPv6].dst)
    else:
        return None
    return PacketSegment(
        frame_number=frame_number,
        timestamp_epoch=float(pkt.time),
        src_ip=src_ip,
        src_port=int(tcp.sport),
        dst_ip=dst_ip,
        dst_port=int(tcp.dport),
        seq=int(tcp.seq),
        payload=payload,
    )


def read_tcp_segments(pcap_path: str | Path) -> list[PacketSegment]:
    path = Path(pcap_path)
    segments: list[PacketSegment] = []
    with PcapReader(str(path)) as reader:
        for frame_number, pkt in enumerate(reader, start=1):
            segment = _packet_segment_from_pkt(frame_number, pkt)
            if segment is not None:
                segments.append(segment)
    return segments


def read_conversation_segments(
    pcap_path: str | Path,
    client_ip: str,
    server_ip: str,
    server_port: int,
    client_port: int | None = None,
    stop_after_seconds: float | None = None,
) -> list[PacketSegment]:
    return list(
        iter_conversation_segments(
            pcap_path=pcap_path,
            client_ip=client_ip,
            server_ip=server_ip,
            server_port=server_port,
            client_port=client_port,
            stop_after_seconds=stop_after_seconds,
        )
    )


def iter_conversation_segments(
    pcap_path: str | Path,
    client_ip: str,
    server_ip: str,
    server_port: int,
    client_port: int | None = None,
    stop_after_seconds: float | None = None,
):
    path = Path(pcap_path)
    selected_client_port: int | None = client_port
    first_match_ts: float | None = None

    def timestamp_from_meta(meta) -> float:
        if hasattr(meta, "tshigh") and hasattr(meta, "tslow"):
            raw = (int(meta.tshigh) << 32) | int(meta.tslow)
            resol = int(getattr(meta, "tsresol", 1000000)) or 1000000
            return raw / resol
        sec = int(getattr(meta, "sec", 0))
        usec = int(getattr(meta, "usec", 0))
        return sec + (usec / 1_000_000.0)

    def parse_segment(frame_number: int, packet: bytes, ts: float) -> PacketSegment | None:
        nonlocal selected_client_port
        if len(packet) < 14:
            return None
        off = 14
        ether_type = int.from_bytes(packet[12:14], "big")
        while ether_type in {0x8100, 0x88A8}:
            if len(packet) < off + 4:
                return None
            ether_type = int.from_bytes(packet[off + 2 : off + 4], "big")
            off += 4

        if ether_type == 0x0800:
            if len(packet) < off + 20:
                return None
            ihl = (packet[off] & 0x0F) * 4
            if ihl < 20 or len(packet) < off + ihl:
                return None
            if packet[off + 9] != 6:
                return None
            src_ip = socket.inet_ntoa(packet[off + 12 : off + 16])
            dst_ip = socket.inet_ntoa(packet[off + 16 : off + 20])
            ip_end = off + ihl
        elif ether_type == 0x86DD:
            if len(packet) < off + 40:
                return None
            if packet[off + 6] != 6:
                return None
            src_ip = socket.inet_ntop(socket.AF_INET6, packet[off + 8 : off + 24])
            dst_ip = socket.inet_ntop(socket.AF_INET6, packet[off + 24 : off + 40])
            ip_end = off + 40
        else:
            return None

        if len(packet) < ip_end + 20:
            return None
        src_port = int.from_bytes(packet[ip_end : ip_end + 2], "big")
        dst_port = int.from_bytes(packet[ip_end + 2 : ip_end + 4], "big")
        is_client_to_server = src_ip == client_ip and dst_ip == server_ip and dst_port == server_port
        is_server_to_client = src_ip == server_ip and src_port == server_port and dst_ip == client_ip
        if not (is_client_to_server or is_server_to_client):
            return None
        client_port = src_port if is_client_to_server else dst_port
        if selected_client_port is None:
            selected_client_port = client_port
        if client_port != selected_client_port:
            return None

        tcp_header_len = ((packet[ip_end + 12] >> 4) & 0x0F) * 4
        if tcp_header_len < 20 or len(packet) < ip_end + tcp_header_len:
            return None
        payload = packet[ip_end + tcp_header_len :]
        if not payload:
            return None
        seq = int.from_bytes(packet[ip_end + 4 : ip_end + 8], "big")
        return PacketSegment(
            frame_number=frame_number,
            timestamp_epoch=ts,
            src_ip=src_ip,
            src_port=src_port,
            dst_ip=dst_ip,
            dst_port=dst_port,
            seq=seq,
            payload=payload,
        )

    reader_cls = RawPcapNgReader if path.suffix.lower().endswith("pcapng") else RawPcapReader
    with reader_cls(str(path)) as reader:
        for frame_number, (packet, meta) in enumerate(reader, start=1):
            ts = timestamp_from_meta(meta)
            if first_match_ts is not None and stop_after_seconds is not None and ts > first_match_ts + stop_after_seconds:
                break
            segment = parse_segment(frame_number, packet, ts)
            if segment is not None:
                if first_match_ts is None:
                    first_match_ts = segment.timestamp_epoch
                yield segment
