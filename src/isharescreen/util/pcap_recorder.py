"""In-app packet recorder — taps the live sockets and writes a `.pcap`.

The Session opens one TCP control channel (enc1103 stream cipher) and two
UDP media sockets (SRTP video / audio + RTCP). This module taps all three
at the *byte boundary* — exactly what crosses the wire, still encrypted —
and writes a classic libpcap file (`LINKTYPE_ETHERNET`) with synthetic but
fully-valid Ethernet / IPv4 / TCP|UDP framing around each chunk.

Why a real-looking capture instead of a raw byte dump: the workspace
Python dissector (and Wireshark) reassemble the TCP control stream from
per-direction TCP sequence numbers and select a conversation by
src/dst IP:port. So we stamp each direction's real client/server IP:port
and hand each direction a monotonic sequence counter. The resulting file
is byte-for-byte what `tcpdump -i any host <mac>` would have captured, and
the dissector — which derives the transport keys from the *cleartext*
handshake it sees in the same stream — decodes it with no extra inputs.

Design:
  * `PcapRecorder` owns the output file, a write lock, and per-flow seq
    state. Recording is on for the lifetime of the Session — one file per
    session, opened on connect and closed on teardown.
  * `wrap_tcp(sock)` / `wrap_udp(sock)` return a thin transparent proxy
    (`_SocketTap`) that delegates every attribute to the real socket and
    only intercepts the data-transfer calls (`send`/`sendall`/`recv`/
    `sendto`/`recvfrom`) to copy the bytes into the recorder. Recording
    failures never disturb the real I/O — they're swallowed and logged.

Capturing at the socket boundary (rather than threading a hook through
every `sendto`/`recv` call site) means we record precisely the wire bytes,
in wire order, with zero protocol knowledge — and the non-recording path
pays nothing because the taps are only installed when recording is armed.
"""
from __future__ import annotations

import logging
import socket
import struct
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

log = logging.getLogger("iss.pcap")


# ── libpcap / framing constants ──────────────────────────────────────

# Classic pcap, microsecond timestamps, host byte order. We write the
# little-endian magic explicitly so the file is portable regardless of the
# writing host's endianness.
_PCAP_MAGIC_USEC = 0xA1B2C3D4
_PCAP_VERSION = (2, 4)
_LINKTYPE_ETHERNET = 1
_SNAPLEN = 0x40000  # 256 KiB — never truncate a chunk

_ETHERTYPE_IPV4 = 0x0800
_IPPROTO_TCP = 6
_IPPROTO_UDP = 17

# Deterministic locally-administered MACs (the 0x02 bit). The dissector
# ignores them; Wireshark just shows two endpoints.
_MAC_CLIENT = bytes.fromhex("020000000001")
_MAC_SERVER = bytes.fromhex("020000000002")

# Keep every synthesised frame inside the 16-bit IPv4 total-length field.
# 65535 − 20 (IP) − 20 (TCP) = 65495; round down for headroom. A single
# recv() can return more than this off a fast TCP stream, so TCP payloads
# are split across multiple in-order segments (the reassembler stitches
# them back via the seq counter).
_MAX_TCP_PAYLOAD = 65000


def _ipv4_checksum(header: bytes) -> int:
    total = 0
    for i in range(0, len(header), 2):
        total += (header[i] << 8) | header[i + 1]
    total = (total & 0xFFFF) + (total >> 16)
    total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def _build_ipv4(proto: int, src_ip: bytes, dst_ip: bytes,
                l4_len: int, ident: int) -> bytes:
    total_len = 20 + l4_len
    # ver/ihl, dscp, total_len, id, flags(DF)+frag, ttl, proto, csum=0,
    # src, dst. Compute the checksum over the zero-checksum header.
    hdr = struct.pack(
        ">BBHHHBBH4s4s",
        0x45, 0x00, total_len & 0xFFFF, ident & 0xFFFF,
        0x4000, 64, proto, 0, src_ip, dst_ip,
    )
    csum = _ipv4_checksum(hdr)
    return hdr[:10] + struct.pack(">H", csum) + hdr[12:]


def _build_tcp(src_port: int, dst_port: int, seq: int, payload: bytes) -> bytes:
    # PSH|ACK, fixed window, zero ack/checksum/urg. The dissector and
    # Wireshark don't verify the L4 checksum; leaving it 0 keeps this
    # dependency-free. data-offset = 5 words (20 bytes, no options).
    hdr = struct.pack(
        ">HHIIBBHHH",
        src_port & 0xFFFF, dst_port & 0xFFFF, seq & 0xFFFFFFFF, 0,
        0x50, 0x18, 0xFFFF, 0, 0,
    )
    return hdr + payload


def _build_udp(src_port: int, dst_port: int, payload: bytes) -> bytes:
    length = 8 + len(payload)
    hdr = struct.pack(">HHHH", src_port & 0xFFFF, dst_port & 0xFFFF,
                      length & 0xFFFF, 0)
    return hdr + payload


def _eth_frame(client_to_server: bool, ip_packet: bytes) -> bytes:
    if client_to_server:
        dst, src = _MAC_SERVER, _MAC_CLIENT
    else:
        dst, src = _MAC_CLIENT, _MAC_SERVER
    return dst + src + struct.pack(">H", _ETHERTYPE_IPV4) + ip_packet


def local_ip_towards(server_ip: str) -> str:
    """The source IPv4 the kernel would use to reach `server_ip`, without
    sending a packet. `connect()` on a UDP socket only sets the default
    peer; `getsockname()` then reports the chosen local address. Falls back
    to loopback if the host has no route (the capture is still consistent —
    both endpoints just share 127.0.0.1)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((server_ip, 9))  # discard port; no datagram is sent
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


# ── pcap file ─────────────────────────────────────────────────────────

class _PcapFile:
    """The open libpcap file: global header on construction, one record per
    `write()`, flush+close on `close()`. Framing and sequence state live in
    `PcapRecorder`; this only serialises (ts, frame) tuples to disk, and only
    ever from PcapRecorder's single writer thread — so it needs no lock."""

    def __init__(self, path: str):
        self.path = path
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # 1 MiB buffer so the writer thread's flush() syscalls stay rare.
        self._fh = open(path, "wb", buffering=1 << 20)
        self._fh.write(struct.pack(
            "<IHHiIII",
            _PCAP_MAGIC_USEC, _PCAP_VERSION[0], _PCAP_VERSION[1],
            0, 0, _SNAPLEN, _LINKTYPE_ETHERNET,
        ))
        self.packets = 0
        self.bytes = 0

    def write(self, ts: float, frame: bytes) -> None:
        sec = int(ts)
        usec = int((ts - sec) * 1_000_000) % 1_000_000
        n = len(frame)
        self._fh.write(struct.pack("<IIII", sec, usec, n, n))
        self._fh.write(frame)
        self.packets += 1
        self.bytes += n

    def close(self) -> None:
        try:
            self._fh.flush()
            self._fh.close()
        except OSError:
            pass


# ── recorder ─────────────────────────────────────────────────────────

class PcapRecorder:
    """Thread-safe capture sink for one Session. Construct with the resolved
    client/server IPv4 addresses, tap sockets via `wrap_tcp` / `wrap_udp`,
    and `close()` when the session ends. The whole session is written to one
    file; there is no mid-session start/stop — recording is simply on for
    the lifetime of a Session created with `record_pcap` set.

    The recording threads (the two UDP drain loops, the TCP rx loop, and the
    tx tick) only frame each packet and append it to an in-memory queue under
    a brief lock — they never touch the file. A single background writer
    thread drains the queue to disk, so the latency-critical UDP drain loops
    (whose job is to empty the kernel socket buffer fast enough to avoid
    kernel drops) never block on a write()/flush() syscall."""

    def __init__(self, path: str, client_ip: str, server_ip: str):
        self._base_path = path
        self._client_ip = client_ip
        self._server_ip = server_ip
        self._client_ip_b = socket.inet_aton(client_ip)
        self._server_ip_b = socket.inet_aton(server_ip)
        # _cond's lock guards the framing state (_tcp_seq, _ident) and the
        # _pending queue. Framing is fast (struct.pack, no syscall); the file
        # write happens on the writer thread, outside this lock.
        self._cond = threading.Condition(threading.Lock())
        # (src_ip, src_port, dst_ip, dst_port) -> next seq for that direction.
        # Each direction starts at 1 and advances by payload length, giving
        # the dissector a gap-free stream to reassemble.
        self._tcp_seq: dict[tuple, int] = {}
        self._ident = 0
        self._pending: deque = deque()
        self._stop = False
        self._file: Optional[_PcapFile] = None
        self._writer: Optional[threading.Thread] = None
        # The ephemeral TCP client port of the most recently wrapped TCP
        # socket — the live control channel. Used to print a ready-to-run
        # dissector command at teardown.
        self.last_tcp_client_port: Optional[int] = None
        try:
            self._file = _PcapFile(path)
            log.info("pcap recording -> %s", path)
        except OSError as e:
            log.error("could not open pcap file %s: %s", path, e)
            self._file = None
        if self._file is not None:
            self._writer = threading.Thread(
                target=self._writer_loop, name="iss-pcap-writer", daemon=True,
            )
            self._writer.start()

    @property
    def path(self) -> Optional[str]:
        return self._file.path if self._file is not None else None

    # ── framing helpers (called holding _cond) ───────────────────────

    def _next_ident(self) -> int:
        self._ident = (self._ident + 1) & 0xFFFF
        return self._ident

    def _next_seq(self, key: tuple, advance: int) -> int:
        seq = self._tcp_seq.get(key, 1)
        self._tcp_seq[key] = (seq + advance) & 0xFFFFFFFF
        return seq

    # ── writer thread ────────────────────────────────────────────────

    def _writer_loop(self) -> None:
        while True:
            with self._cond:
                while not self._pending and not self._stop:
                    self._cond.wait()
                if self._stop and not self._pending:
                    return
                batch = self._pending
                self._pending = deque()
                f = self._file
            # Write the batch outside the lock so the recording threads
            # can keep enqueuing while the (possibly flushing) write runs.
            if f is not None:
                for ts, frame in batch:
                    f.write(ts, frame)

    # ── recording entry points (called from the socket taps) ─────────

    def record_tcp(self, outbound: bool, client_port: int, server_port: int,
                   data: bytes) -> None:
        if not data:
            return
        ts = time.time()
        if outbound:
            c2s, src_ip, src_port, dst_ip, dst_port = (
                True, self._client_ip_b, client_port,
                self._server_ip_b, server_port)
        else:
            c2s, src_ip, src_port, dst_ip, dst_port = (
                False, self._server_ip_b, server_port,
                self._client_ip_b, client_port)
        key = (src_ip, src_port, dst_ip, dst_port)
        view = memoryview(data)
        total = len(data)
        with self._cond:
            if self._file is None:
                return
            off = 0
            # Split payloads larger than one IPv4 frame into in-order TCP
            # segments; the per-direction seq counter keeps them contiguous.
            while True:
                chunk = bytes(view[off:off + _MAX_TCP_PAYLOAD])
                seq = self._next_seq(key, len(chunk))
                l4 = _build_tcp(src_port, dst_port, seq, chunk)
                ip_hdr = _build_ipv4(_IPPROTO_TCP, src_ip, dst_ip, len(l4),
                                     self._next_ident())
                self._pending.append((ts, _eth_frame(c2s, ip_hdr + l4)))
                off += len(chunk)
                if off >= total:
                    break
            self._cond.notify()

    def record_udp(self, outbound: bool, local_port: int, remote_port: int,
                   data: bytes) -> None:
        if not data:
            return
        ts = time.time()
        if outbound:
            c2s, src_ip, src_port, dst_ip, dst_port = (
                True, self._client_ip_b, local_port,
                self._server_ip_b, remote_port)
        else:
            c2s, src_ip, src_port, dst_ip, dst_port = (
                False, self._server_ip_b, remote_port,
                self._client_ip_b, local_port)
        l4 = _build_udp(src_port, dst_port, data)
        with self._cond:
            if self._file is None:
                return
            ip_hdr = _build_ipv4(_IPPROTO_UDP, src_ip, dst_ip, len(l4),
                                 self._next_ident())
            self._pending.append((ts, _eth_frame(c2s, ip_hdr + l4)))
            self._cond.notify()

    def close(self) -> tuple[int, int]:
        """Stop the writer, flush every queued packet to disk, close the file.
        Idempotent. Returns the final (packets, bytes) written."""
        with self._cond:
            if self._file is None:
                return (0, 0)
            self._stop = True
            self._cond.notify()
        if self._writer is not None:
            self._writer.join(timeout=2.0)
        with self._cond:
            f = self._file
            if f is None:
                return (0, 0)
            # Drain any straggler the writer didn't reach (e.g. a packet
            # enqueued after stop, or a join() that timed out).
            while self._pending:
                ts, frame = self._pending.popleft()
                f.write(ts, frame)
            stats = (f.packets, f.bytes)
            f.close()
            self._file = None
            return stats

    # ── socket taps ──────────────────────────────────────────────────

    def wrap_tcp(self, sock: socket.socket) -> "_TcpTap":
        try:
            client_port = sock.getsockname()[1]
            server_port = sock.getpeername()[1]
        except (OSError, IndexError, TypeError):
            # Not connected yet / non-AF_INET / odd state — fall back to
            # port 0 (Wireshark just shows port 0; capture still works).
            client_port, server_port = 0, 0
        self.last_tcp_client_port = client_port
        log.debug("tap TCP %s:%d -> %s:%d", self._client_ip, client_port,
                  self._server_ip, server_port)
        return _TcpTap(sock, self, client_port, server_port)

    def wrap_udp(self, sock: socket.socket) -> "_UdpTap":
        try:
            local_port = sock.getsockname()[1]
        except (OSError, IndexError, TypeError):
            local_port = 0
        log.debug("tap UDP local:%d", local_port)
        return _UdpTap(sock, self, local_port)

    # ── reporting ────────────────────────────────────────────────────

    def dissector_hint(self) -> str:
        """Dissector command lines for the captured session. `decode` needs
        the post-auth transport key, which the capture alone doesn't carry —
        it's shown as a placeholder. `list-streams` is runnable as-is, and
        the file opens in Wireshark with no key for the cleartext handshake."""
        port = self.last_tcp_client_port
        port_arg = f" --client-port {port}" if port else ""
        return (
            "dissector_cli.py list-streams --pcap %s\n"
            "dissector_cli.py decode --pcap %s "
            "--client-ip %s --server-ip %s --server-port 5900%s "
            "--initial-key-hex <TRANSPORT_KEY_HEX> --output records.jsonl"
            % (self._base_path, self._base_path,
               self._client_ip, self._server_ip, port_arg)
        )


# ── transparent socket proxies ───────────────────────────────────────

class _SocketTap:
    """Delegates every attribute to the wrapped socket; subclasses override
    only the data-transfer methods. Keeps the wrapped object quacking like a
    real `socket.socket` for `setsockopt`/`settimeout`/`fileno`/`close`/
    `getsockname`/`getpeername`/etc. via `__getattr__`."""

    __slots__ = ("_sock", "_rec")

    def __init__(self, sock: socket.socket, rec: PcapRecorder):
        object.__setattr__(self, "_sock", sock)
        object.__setattr__(self, "_rec", rec)

    def __getattr__(self, name):  # only reached for un-overridden attrs
        return getattr(self._sock, name)

    def __setattr__(self, name, value):
        setattr(self._sock, name, value)


class _TcpTap(_SocketTap):
    __slots__ = ("_cport", "_sport")

    def __init__(self, sock, rec, client_port, server_port):
        super().__init__(sock, rec)
        object.__setattr__(self, "_cport", client_port)
        object.__setattr__(self, "_sport", server_port)

    def _rec_out(self, data) -> None:
        try:
            self._rec.record_tcp(True, self._cport, self._sport, bytes(data))
        except Exception:  # never let recording break real I/O
            log.debug("tcp record (out) failed", exc_info=True)

    def _rec_in(self, data) -> None:
        try:
            self._rec.record_tcp(False, self._cport, self._sport, bytes(data))
        except Exception:
            log.debug("tcp record (in) failed", exc_info=True)

    def sendall(self, data, *args):
        result = self._sock.sendall(data, *args)
        self._rec_out(data)
        return result

    def send(self, data, *args):
        n = self._sock.send(data, *args)
        self._rec_out(memoryview(data)[:n])
        return n

    def recv(self, bufsize, *args):
        data = self._sock.recv(bufsize, *args)
        if data:
            self._rec_in(data)
        return data


class _UdpTap(_SocketTap):
    __slots__ = ("_lport",)

    def __init__(self, sock, rec, local_port):
        super().__init__(sock, rec)
        object.__setattr__(self, "_lport", local_port)

    def sendto(self, data, *args):
        # sendto(data, addr) or sendto(data, flags, addr)
        addr = args[-1] if args else None
        n = self._sock.sendto(data, *args)
        try:
            rport = addr[1] if isinstance(addr, tuple) and len(addr) >= 2 else 0
            self._rec.record_udp(True, self._lport, rport,
                                 bytes(memoryview(data)[:n]))
        except Exception:
            log.debug("udp record (out) failed", exc_info=True)
        return n

    def recvfrom(self, bufsize, *args):
        data, addr = self._sock.recvfrom(bufsize, *args)
        try:
            if data:
                rport = addr[1] if isinstance(addr, tuple) and len(addr) >= 2 else 0
                self._rec.record_udp(False, self._lport, rport, data)
        except Exception:
            log.debug("udp record (in) failed", exc_info=True)
        return data, addr


__all__ = ["PcapRecorder", "local_ip_towards"]
