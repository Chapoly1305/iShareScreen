"""Bounded, process-local socket-link emulator for the Mac capture experiment.

Each direction has one FIFO serializer shared by video UDP, control UDP and
TCP application bytes. Charge UDP payload+28 bytes, TCP segments+40 bytes
(MSS=1200). No physical TCP ACK, retransmission, Wi-Fi or VPN claim. TCP uses
backpressure; UDP tail-drops when the modeled queue is full. It is a link
emulation at the application socket boundary, not kernel packet shaping.
Only sizes, timing and stream names are logged, never payload or credentials.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import socket
import threading
import time


@dataclass
class Packet:
    stream: str
    payload_bytes: int
    charged_bytes: int
    deliver: object
    arrival: float = field(default_factory=time.monotonic)
    done: threading.Event = field(default_factory=threading.Event)
    error: BaseException | None = None
    enqueued: float = 0.0


class Direction:
    def __init__(self, name, rate_bps, queue_ms, delay_ms):
        if rate_bps <= 0 or queue_ms <= 0 or delay_ms < 0:
            raise ValueError("Invalid link parameters")
        self.name = name
        self.rate_bps = rate_bps
        self.capacity = max(1500, int(rate_bps * queue_ms / 8000))
        self.delay = delay_ms / 1000
        self.cv = threading.Condition()
        self.queue = deque()
        self.propagating = deque()
        self.bytes = 0
        self.closed = False
        self.records = []
        self.stop = threading.Event()
        self.worker = threading.Thread(
            target=self._run, name="link-" + name, daemon=True
        )
        self.worker.start()
        self.delivery_worker = threading.Thread(
            target=self._deliver, name="link-deliver-" + name, daemon=True
        )
        self.delivery_worker.start()

    def _record(self, p, status, **more):
        self.records.append(
            dict(
                stream=p.stream,
                payload_bytes=p.payload_bytes,
                charged_bytes=p.charged_bytes,
                arrival=p.arrival,
                status=status,
                time=time.monotonic(),
                **more,
            )
        )

    def submit(self, p, *, reliable=False, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        with self.cv:
            if p.charged_bytes > self.capacity:
                self._record(p, "oversize_drop")
                p.done.set()
                if reliable:
                    raise ValueError("Reliable segment exceeds queue capacity")
                return False
            while self.bytes + p.charged_bytes > self.capacity and not self.closed:
                if not reliable:
                    self._record(p, "tail_drop")
                    p.done.set()
                    return False
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise socket.timeout("Link queue timeout")
                self.cv.wait(remaining)
            if self.closed:
                raise OSError("Link is closed")
            p.enqueued = time.monotonic()
            self.queue.append(p)
            self.bytes += p.charged_bytes
            self.cv.notify_all()
            return True

    def _run(self):
        # Absolute service deadlines avoid adding one OS timer oversleep to
        # every small packet. Delivery has a separate one-packet burst budget.
        service_until = 0.0
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.queue or self.closed)
                if self.closed and not self.queue:
                    return
                p = self.queue.popleft()
            started = max(p.enqueued, service_until)
            service_until = started + p.charged_bytes * 8 / self.rate_bps
            if self.stop.wait(max(0, service_until - time.monotonic())):
                with self.cv:
                    self._record(p, "shutdown_drop")
                    self.bytes -= p.charged_bytes
                    self.cv.notify_all()
                p.error = OSError("Link stopped")
                p.done.set()
            else:
                service_end = service_until
                with self.cv:
                    if self.closed:
                        self.bytes -= p.charged_bytes
                        self._record(p, "shutdown_drop")
                        p.error = OSError("Link stopped")
                        p.done.set()
                    else:
                        self.propagating.append((p, started, service_end))
                    self.cv.notify_all()

    def _deliver(self):
        credit = 0.0
        credit_at = time.monotonic()
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.propagating or self.closed)
                if self.closed and not self.propagating:
                    return
                p, started, service_end = self.propagating.popleft()
            # FIFO preserves TCP byte order. A token budget bounds catch-up
            # after scheduling delays to one MTU (or one oversized datagram).
            due = service_end + self.delay
            try:
                if self.stop.wait(max(0, due - time.monotonic())):
                    raise OSError("Link stopped")
                burst_bytes = max(1500, p.charged_bytes)
                while True:
                    now = time.monotonic()
                    credit = min(
                        burst_bytes, credit + (now - credit_at) * self.rate_bps / 8
                    )
                    credit_at = now
                    if credit >= p.charged_bytes:
                        break
                    if self.stop.wait((p.charged_bytes - credit) * 8 / self.rate_bps):
                        raise OSError("Link stopped")
                credit -= p.charged_bytes
                p.deliver()
                with self.cv:
                    self._record(
                        p,
                        "delivered",
                        service_start=started,
                        service_end=service_end,
                        burst_budget_bytes=burst_bytes,
                    )
            except BaseException as error:
                p.error = error
                with self.cv:
                    self._record(
                        p,
                        "shutdown_drop" if self.stop.is_set() else "delivery_error",
                        error_type=type(error).__name__,
                    )
            finally:
                with self.cv:
                    self.bytes -= p.charged_bytes
                    self.cv.notify_all()
                p.done.set()

    def close(self):
        with self.cv:
            self.closed = True
            self.stop.set()
            while self.queue:
                p = self.queue.popleft()
                self.bytes -= p.charged_bytes
                self._record(p, "shutdown_drop")
                p.error = OSError("Link stopped")
                p.done.set()
            while self.propagating:
                p, _, _ = self.propagating.popleft()
                self.bytes -= p.charged_bytes
                self._record(p, "shutdown_drop")
                p.error = OSError("Link stopped")
                p.done.set()
            self.cv.notify_all()
        self.worker.join(timeout=2)
        self.delivery_worker.join(timeout=2)


class ShapedSocket:
    MSS = 1200

    def __init__(self, raw, link, kind, stream):
        self.raw = raw
        self.link = link
        self.kind = kind
        self.stream = stream
        self.timeout = raw.gettimeout()
        raw.settimeout(0.1)
        self.cv = threading.Condition()
        self.ready = deque()
        self.ready_bytes = 0
        self.closed = False
        self.eof = False
        self.ready_limit = 1024 * 1024
        self.receive_overflows = 0
        self.send_lock = threading.Lock()
        self.reader = threading.Thread(
            target=self._read, name="link-read-" + stream, daemon=True
        )
        self.reader.start()

    def __getattr__(self, name):
        return getattr(self.raw, name)

    def settimeout(self, value):
        if value is not None and value < 0:
            raise ValueError("Negative timeout")
        self.timeout = value

    def gettimeout(self):
        return self.timeout

    def setblocking(self, flag):
        self.timeout = None if flag else 0

    def _receive(self, data, address):
        with self.cv:
            if self.closed:
                raise OSError("Socket closed")
            if self.kind == "udp" and self.ready_bytes + len(data) > self.ready_limit:
                self.receive_overflows += 1
                return
            while (
                self.kind == "tcp"
                and self.ready_bytes + len(data) > self.ready_limit
                and not self.closed
            ):
                self.cv.wait(0.1)
            if self.closed:
                raise OSError("Socket closed")
            self.ready.append((data, address))
            self.ready_bytes += len(data)
            self.cv.notify_all()

    def _finish(self):
        with self.cv:
            self.eof = True
            self.cv.notify_all()

    def _read(self):
        try:
            while not self.closed:
                try:
                    if self.kind == "udp":
                        data, address = self.raw.recvfrom(65536)
                    else:
                        data, address = self.raw.recv(self.MSS), None
                except socket.timeout:
                    continue
                if self.kind == "tcp" and not data:
                    self.link.incoming.submit(
                        Packet(self.stream, 0, 0, self._finish), reliable=True
                    )
                    return
                packet = Packet(
                    self.stream,
                    len(data),
                    len(data) + (28 if self.kind == "udp" else 40),
                    lambda data=data, address=address: self._receive(data, address),
                )
                self.link.incoming.submit(packet, reliable=self.kind == "tcp")
        except OSError:
            self._finish()

    def _take(self, size, flags=0):
        if flags not in (0, socket.MSG_PEEK):
            raise ValueError("Unsupported receive flags")
        if size < 0:
            raise ValueError("Negative read size")
        if size == 0 and self.kind == "tcp":
            return b"", None
        deadline = None if self.timeout is None else time.monotonic() + self.timeout
        with self.cv:
            while not self.ready and not self.eof and not self.closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    if self.timeout == 0:
                        raise BlockingIOError()
                    raise socket.timeout("Shaped receive timeout")
                self.cv.wait(remaining)
            if not self.ready:
                if self.closed:
                    raise OSError("Socket closed")
                return b"", None
            data, address = self.ready[0]
            if flags != socket.MSG_PEEK:
                if self.kind == "udp" or len(data) <= size:
                    self.ready.popleft()
                    self.ready_bytes -= len(data)
                else:
                    self.ready[0] = (data[size:], address)
                    self.ready_bytes -= size
                self.cv.notify_all()
            return data[:size], address

    def recv(self, size, flags=0):
        return self._take(size, flags)[0]

    def recvfrom(self, size, flags=0):
        return self._take(size, flags)

    def sendto(self, data, *args):
        data = bytes(data)
        packet = Packet(
            self.stream, len(data), len(data) + 28, lambda: self.raw.sendto(data, *args)
        )
        self.link.outgoing.submit(packet)
        return len(data)  # UDP enqueue/drop follows normal datagram send semantics.

    def sendall(self, data, flags=0):
        if flags:
            raise ValueError("Unsupported send flags")
        with self.send_lock:
            view = memoryview(data)
            deadline = None if self.timeout is None else time.monotonic() + self.timeout
            for start in range(0, len(data), self.MSS):
                chunk = bytes(view[start : start + self.MSS])
                remaining = (
                    None if deadline is None else max(0, deadline - time.monotonic())
                )
                packet = Packet(
                    self.stream,
                    len(chunk),
                    len(chunk) + 40,
                    lambda chunk=chunk: self.raw.sendall(chunk),
                )
                self.link.outgoing.submit(packet, reliable=True, timeout=remaining)
                if not packet.done.wait(remaining):
                    raise socket.timeout("Shaped send timeout")
                if packet.error:
                    raise packet.error

    def send(self, data, flags=0):
        self.sendall(data, flags)
        return len(data)

    def close(self):
        with self.cv:
            if self.closed:
                return
            self.closed = True
            self.cv.notify_all()
        self.raw.close()
        self.reader.join(timeout=0.5)


class SocketLink:
    def __init__(self, rate_bps=600000, queue_ms=250, delay_ms=10):
        self.config = dict(
            rate_bps_per_direction=rate_bps,
            queue_ms=queue_ms,
            one_way_delay_ms=delay_ms,
            udp_ipv4_header_bytes=28,
            tcp_ipv4_header_bytes=40,
            tcp_segment_bytes=ShapedSocket.MSS,
            service_clock="Absolute FIFO serialization; actual delivery token bucket allows at most one 1500-byte MTU or one larger datagram",
            queue_policy="Shared FIFO per direction; capacity covers queued/in-service/in-propagation bytes; UDP tail-drop, TCP application backpressure",
            scope=__doc__,
            physical_network_claim=False,
        )
        self.incoming = Direction("incoming", rate_bps, queue_ms, delay_ms)
        self.outgoing = Direction("outgoing", rate_bps, queue_ms, delay_ms)
        self.sockets = []

    def wrap_tcp(self, sock):
        result = ShapedSocket(sock, self, "tcp", f"tcp-{len(self.sockets)}")
        self.sockets.append(result)
        return result

    def wrap_udp(self, sock):
        result = ShapedSocket(sock, self, "udp", f"udp-{sock.getsockname()[1]}")
        self.sockets.append(result)
        return result

    def snapshot(self):
        result = dict(config=self.config, directions={})
        for direction in [self.incoming, self.outgoing]:
            with direction.cv:
                records = list(direction.records)
                result["directions"][direction.name] = dict(
                    queued_bytes=direction.bytes,
                    records=records,
                    delivered_payload_bytes=sum(
                        r["payload_bytes"]
                        for r in records
                        if r["status"] == "delivered"
                    ),
                    delivered_charged_bytes=sum(
                        r["charged_bytes"]
                        for r in records
                        if r["status"] == "delivered"
                    ),
                    dropped_packets=sum(r["status"].endswith("drop") for r in records),
                )
        result["socket_receive_overflows"] = {
            s.stream: s.receive_overflows for s in self.sockets
        }
        return result

    def close(self):
        for sock in self.sockets:
            sock.close()
        self.incoming.close()
        self.outgoing.close()
