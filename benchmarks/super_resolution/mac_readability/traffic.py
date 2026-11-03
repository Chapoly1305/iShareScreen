"""Export only byte/timing statistics from a private in-app capture.

PCAP and authentication material remain on the Mac. Its Ethernet/IP/TCP
framing is synthetic; UDP payload counts are exact socket observations,
not a physical network capture or proof of a constrained WAN.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", type=Path, required=True)
    args = p.parse_args()
    status = json.loads((args.capture / "STATUS.json").read_text())
    start = status["connected_unix"]
    duration = status["elapsed_seconds"]
    end = start + duration
    rows = []
    totals = defaultdict(lambda: dict(payload_bytes=0, chunks=0))
    with (args.capture / "session.pcap").open("rb") as stream:
        header = stream.read(24)
        if header[:4] != b"\xd4\xc3\xb2\xa1":
            raise ValueError("Unexpected PCAP byte order")
        while chunk := stream.read(16):
            sec, usec, n, _ = struct.unpack("<IIII", chunk)
            packet = stream.read(n)
            if len(packet) != n:
                raise ValueError("Truncated PCAP")
            timestamp = sec + usec / 1e6
            if not start <= timestamp <= end:
                continue
            inbound = packet[6:12] == bytes.fromhex("020000000002")
            proto = packet[23]
            ip_length = (packet[14] & 15) * 4
            l4 = 14 + ip_length
            sport, dport = struct.unpack(">HH", packet[l4 : l4 + 4])
            if proto == 17:
                size = len(packet) - l4 - 8
                kind = "video_udp" if 5901 in (sport, dport) else "control_udp"
            elif proto == 6:
                size = len(packet) - l4 - (packet[l4 + 12] >> 4) * 4
                kind = "tcp_application"
            else:
                continue
            direction = "incoming" if inbound else "outgoing"
            key = direction + "_" + kind
            totals[key]["payload_bytes"] += size
            totals[key]["chunks"] += 1
            rows.append(
                dict(t=timestamp - start, kind=kind, direction=direction, bytes=size)
            )
    for value in totals.values():
        value["average_kbps"] = value["payload_bytes"] * 8 / duration / 1000
    windows = {}
    for span in [0.1, 1.0, 10.0]:
        # Fixed-bin peaks, not a continuously sliding-window supremum.
        bins = []
        for i in range(int(duration / span)):
            selected = [r for r in rows if i * span <= r["t"] < (i + 1) * span]
            down = sum(r["bytes"] for r in selected if r["direction"] == "incoming")
            udp_count = sum(
                r["kind"].endswith("_udp") and r["direction"] == "incoming"
                for r in selected
            )
            bins.append(
                dict(
                    start_s=i * span,
                    incoming_payload_kbps=down * 8 / span / 1000,
                    udp_ipv4_overhead_estimated_kbps=udp_count * 28 * 8 / span / 1000,
                )
            )
        windows[str(span)] = bins
    result = dict(
        schema="mac_socket_traffic_diagnostic_v1",
        scope=__doc__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status_sha256=hashlib.sha256(
            (args.capture / "STATUS.json").read_bytes()
        ).hexdigest(),
        duration_s=duration,
        totals=dict(totals),
        fixed_windows=windows,
        exclusions=[
            "Initial connection before capture start",
            "TCP retransmissions and ACK/IP overhead unknown",
            "Ethernet, Wi-Fi, VPN and physical wire overhead unknown",
        ],
        raw_samples=rows,
    )
    (args.capture / "TRAFFIC.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(duration_s=duration, totals=dict(totals)), indent=2))


if __name__ == "__main__":
    main()
