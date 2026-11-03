"""Exercise rate accounting, ordering and stream semantics of the lab link."""

import socket
import threading
import time

import pytest

from benchmarks.super_resolution.mac_readability.link import (
    Direction,
    Packet,
    SocketLink,
)


def test_shared_rate_budget_and_order_with_propagation():
    direction = Direction("test", 600000, 1000, 10)
    seen = []
    packets = []
    try:
        for i in range(24):
            packet = Packet(
                "video" if i % 2 else "control", 1000, 1028, lambda i=i: seen.append(i)
            )
            assert direction.submit(packet)
            packets.append(packet)
        assert packets[-1].done.wait(3)
        assert seen == list(range(24))
        rows = [r for r in direction.records if r["status"] == "delivered"]
        assert len(rows) == 24
        duration = rows[-1]["time"] - rows[0]["service_start"]
        assert sum(r["charged_bytes"] for r in rows) * 8 / duration <= 600000
        for previous, current in zip(rows, rows[1:]):
            assert current["service_start"] >= previous["service_end"]
            assert current["time"] > previous["time"]
        assert all(r["time"] - r["service_end"] >= 0.009 for r in rows)
    finally:
        direction.close()
    assert direction.bytes == 0


def test_bounded_udp_tail_drop_and_shutdown():
    direction = Direction("test", 600000, 20, 0)
    try:
        first = Packet("video", 1200, 1228, lambda: None)
        second = Packet(
            "video", 1200, 1228, lambda: pytest.fail("Dropped packet delivered")
        )
        assert direction.submit(first)
        assert not direction.submit(second)
        assert any(r["status"] == "tail_drop" for r in direction.records)
        assert direction.bytes <= direction.capacity
    finally:
        direction.close()
    assert first.done.is_set() and second.done.is_set() and direction.bytes == 0


def tcp_pair():
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    client = socket.create_connection(listener.getsockname())
    server, _ = listener.accept()
    listener.close()
    client.settimeout(3)
    server.settimeout(3)
    return client, server


def test_tcp_peek_stream_integrity_and_eof():
    client, server = tcp_pair()
    link = SocketLink(600000, 250, 5)
    wrapped = link.wrap_tcp(client)
    payload = bytes(range(256)) * 20

    def sending():
        server.sendall(payload)
        server.shutdown(socket.SHUT_WR)

    worker = threading.Thread(target=sending)
    worker.start()
    try:
        first = wrapped.recv(4, socket.MSG_PEEK)
        assert first == payload[:4] and wrapped.recv(4, socket.MSG_PEEK) == first
        received = bytearray()
        while data := wrapped.recv(137):
            received.extend(data)
        assert bytes(received) == payload
        records = link.snapshot()["directions"]["incoming"]["records"]
        assert sum(
            r["payload_bytes"] for r in records if r["status"] == "delivered"
        ) == len(payload)
        assert wrapped.recv(1) == b""
    finally:
        wrapped.close()
        server.close()
        link.close()
        worker.join(1)


def test_tcp_outgoing_order_under_shared_budget():
    client, server = tcp_pair()
    link = SocketLink(600000, 250, 3)
    wrapped = link.wrap_tcp(client)
    payload = b"abcXYZ012345" * 500
    received = bytearray()

    def reading():
        while len(received) < len(payload):
            received.extend(server.recv(901))

    worker = threading.Thread(target=reading)
    worker.start()
    try:
        wrapped.sendall(payload)
        worker.join(3)
        assert not worker.is_alive()
        assert bytes(received) == payload
        record = link.snapshot()["directions"]["outgoing"]
        assert record["delivered_payload_bytes"] == len(payload)
        assert record["delivered_charged_bytes"] == len(payload) + 40 * 5
    finally:
        wrapped.close()
        server.close()
        link.close()


def test_udp_timeout_truncation_and_bidirectional_delivery():
    raw = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    raw.bind(("127.0.0.1", 0))
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(2)
    link = SocketLink(600000, 250, 2)
    wrapped = link.wrap_udp(raw)
    wrapped.settimeout(0.03)
    try:
        with pytest.raises(socket.timeout):
            wrapped.recvfrom(100)
        server.sendto(b"hello-world", wrapped.getsockname())
        data, address = wrapped.recvfrom(5)
        assert data == b"hello" and address == server.getsockname()
        with pytest.raises(socket.timeout):
            wrapped.recvfrom(100)
        assert wrapped.sendto(b"reply", server.getsockname()) == 5
        assert server.recvfrom(100)[0] == b"reply"
        statistics = link.snapshot()
        assert (
            statistics["directions"]["incoming"]["delivered_charged_bytes"] == 11 + 28
        )
        assert statistics["directions"]["outgoing"]["delivered_charged_bytes"] == 5 + 28
    finally:
        wrapped.close()
        server.close()
        link.close()


def test_close_releases_reliable_sender_waiting_for_capacity():
    direction = Direction("test", 10000, 100, 0)
    waiting = threading.Event()
    errors = []
    first = Packet("video", 1200, 1228, lambda: None)
    assert direction.submit(first)

    def sending():
        waiting.set()
        try:
            direction.submit(Packet("tcp", 1200, 1240, lambda: None), reliable=True)
        except OSError as error:
            errors.append(error)

    worker = threading.Thread(target=sending)
    worker.start()
    assert waiting.wait(1)
    direction.close()
    worker.join(1)
    assert not worker.is_alive() and len(errors) == 1 and direction.bytes == 0
