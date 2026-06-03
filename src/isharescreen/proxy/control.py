"""Control socket: stream stats and (later) accept commands over a local
socket so a separate terminal-side TUI can monitor and tweak a running
iss session without sharing the wgpu viewer's main thread.

Wire protocol is newline-delimited JSON. Server → client message types:

    {"type": "hello",    "version": 1, "session": {...}}        # once per client connect
    {"type": "snapshot", "ts": <epoch>, "data": {...}}          # every profile tick
    {"type": "event",    "ts": <epoch>, "kind": "...", ...}     # connect / disconnect / error

Client → server (phase 2):

    {"action": "fir"}                          # force IDR for all tiles
    {"action": "reconnect"}                    # tear-down + re-handshake
    {"action": "set", "key": "audio", "value": false}
    {"action": "quit"}

Phase 1 (this module): server-side hello + snapshot publish. Inbound
bytes are drained and ignored until commands land.

Transport:
    POSIX  - AF_UNIX at the caller's path (default ~/.iss/control.sock).
             0o600 perms on the socket file; stale leftovers unlinked.
    Win32  - AF_INET on 127.0.0.1, OS-chosen port. The actual port is
             written to `<addr>.port` so clients can discover it without
             a well-known-port collision.

Local-only by design: UDS is filesystem-permission-protected on POSIX,
loopback-only on Windows. If you can read the socket you can already see
the streamed screen, so we add no auth on top.
"""
from __future__ import annotations

import json
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional


log = logging.getLogger("iss.control")

# Type alias for the caller-supplied command dispatcher. Takes the raw
# command dict (already JSON-parsed) and runs whatever Session method
# matches. Errors are caught and logged inside the read loop; the
# callback should NOT re-raise.
CommandHandler = Callable[[dict[str, Any]], None]


def default_socket_path() -> str:
    """Per-user default address. POSIX: a UDS path; Windows: a sidecar
    file path that gets `.port` appended for the actual TCP port file."""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return str(Path(base) / "iss" / "control.sock")
    runtime = os.environ.get("XDG_RUNTIME_DIR")
    base = Path(runtime) if runtime else Path.home() / ".iss"
    if runtime:
        base = base / "iss"
    return str(base / "control.sock")


class ControlServer:
    """Publishes session snapshots and accepts client commands on a local
    socket. One server per Session; any number of clients can subscribe.

    Thread model: a daemon thread accepts new clients; broadcasts go out
    on whatever thread calls `publish_snapshot` / `publish_event` (the
    session's tx tick, by design). A per-server lock guards the client
    list during broadcast and accept. Dead sockets are pruned lazily on
    the next broadcast."""

    def __init__(self, addr: str, on_command: Optional[CommandHandler] = None):
        self._addr = addr
        self._is_posix = sys.platform != "win32"
        self._listen: Optional[socket.socket] = None
        self._clients: list[socket.socket] = []
        self._client_threads: list[threading.Thread] = []
        self._clients_lock = threading.Lock()
        self._accept_thr: Optional[threading.Thread] = None
        self._stop = threading.Event()
        # Cached "hello" payload — sent to each new client on connect so
        # the TUI has the static session header before the first
        # snapshot lands.
        self._hello: dict[str, Any] = {}
        # Per-client command dispatcher; None disables the inbound path
        # (server is publish-only).
        self._on_command = on_command

    @property
    def address(self) -> str:
        """Where clients connect. UDS path on POSIX, `host:port` on
        Windows (port resolved after `start()` binds)."""
        if self._is_posix:
            return self._addr
        if self._listen is None:
            return self._addr
        host, port = self._listen.getsockname()
        return f"{host}:{port}"

    def start(self) -> None:
        """Bind, listen, start the accept loop. Idempotent."""
        if self._listen is not None:
            return
        Path(self._addr).parent.mkdir(parents=True, exist_ok=True)
        if self._is_posix:
            # Stale UDS file from a crashed previous run would block
            # bind with EADDRINUSE; unlink it.
            try:
                os.unlink(self._addr)
            except FileNotFoundError:
                pass
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.bind(self._addr)
            try:
                os.chmod(self._addr, 0o600)
            except OSError:
                pass
        else:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            _, port = sock.getsockname()
            try:
                Path(self._addr + ".port").write_text(str(port))
            except OSError as e:
                log.warning("could not write port sidecar %s: %s",
                            self._addr + ".port", e)
        sock.listen(8)
        # Short timeout so the accept thread can notice _stop.
        sock.settimeout(0.5)
        self._listen = sock
        self._accept_thr = threading.Thread(
            target=self._accept_loop, name="iss-ctrl-accept", daemon=True,
        )
        self._accept_thr.start()
        log.info("control socket listening at %s", self.address)

    def set_hello(self, **fields: Any) -> None:
        """Replace the cached hello payload. Called by the session once
        the static header (host, canvas, decoder, advertised geometry,
        etc.) is known after a successful handshake."""
        self._hello = dict(fields)

    def publish_snapshot(self, data: dict[str, Any]) -> None:
        """Broadcast a periodic snapshot to all subscribed clients."""
        self._broadcast({"type": "snapshot", "ts": time.time(), "data": data})

    def publish_event(self, kind: str, **fields: Any) -> None:
        """Broadcast a one-off event (connect / disconnect / error /
        canvas change / ...)."""
        msg = {"type": "event", "ts": time.time(), "kind": kind}
        msg.update(fields)
        self._broadcast(msg)

    def close(self) -> None:
        """Stop the accept loop, close every client, unbind, and clean up
        the on-disk artifacts (UDS file or port sidecar)."""
        self._stop.set()
        sock, self._listen = self._listen, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass
        with self._clients_lock:
            for c in self._clients:
                try:
                    c.close()
                except OSError:
                    pass
            self._clients.clear()
            # Snapshot the threads so we can join them outside the lock
            # (their loops grab `_clients_lock` on exit to deregister).
            threads = list(self._client_threads)
            self._client_threads.clear()
        # Best-effort cleanup of the discovery file(s).
        try:
            if self._is_posix:
                os.unlink(self._addr)
            else:
                os.unlink(self._addr + ".port")
        except OSError:
            pass
        if self._accept_thr is not None:
            self._accept_thr.join(timeout=1.0)
        # Reap any per-client reader threads. They notice the closed
        # socket on their next recv (or `_stop` on the 0.5 s timeout) and
        # exit; joining ensures we don't return until they've removed
        # themselves cleanly.
        for t in threads:
            if t.is_alive():
                t.join(timeout=1.0)

    # ── internals ──────────────────────────────────────────────────

    def _broadcast(self, msg: dict[str, Any]) -> None:
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        with self._clients_lock:
            dead: list[socket.socket] = []
            for c in self._clients:
                try:
                    c.sendall(line)
                except OSError:
                    dead.append(c)
            for c in dead:
                self._clients.remove(c)
                try:
                    c.close()
                except OSError:
                    pass

    def _send_to(self, c: socket.socket, msg: dict[str, Any]) -> bool:
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            c.sendall(line)
            return True
        except OSError:
            return False

    def _accept_loop(self) -> None:
        sock = self._listen
        if sock is None:
            return
        while not self._stop.is_set():
            try:
                client, _ = sock.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            log.debug("control client connected")
            # Greet immediately with the cached hello so the TUI can
            # render static panels before the first periodic snapshot.
            hello = {"type": "hello", "version": 1, "session": self._hello}
            if not self._send_to(client, hello):
                try:
                    client.close()
                except OSError:
                    pass
                continue
            with self._clients_lock:
                self._clients.append(client)
            # Per-client read thread, only when a dispatcher is wired
            # (publish-only servers skip this and just drain bytes via
            # short reads in `_broadcast` failures).
            if self._on_command is not None:
                t = threading.Thread(
                    target=self._client_read_loop, args=(client,),
                    name="iss-ctrl-client-rx", daemon=True,
                )
                t.start()
                with self._clients_lock:
                    self._client_threads.append(t)

    def _client_read_loop(self, client: socket.socket) -> None:
        """Read newline-delimited JSON commands from one client and
        dispatch via `_on_command`. Stops on EOF / OS error / `_stop`."""
        buf = b""
        client.settimeout(0.5)
        while not self._stop.is_set():
            try:
                chunk = client.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line.strip():
                    continue
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("malformed control command (%s): %r", e, line[:80])
                    continue
                if not isinstance(msg, dict):
                    log.warning("control command not a dict: %r", msg)
                    continue
                try:
                    if self._on_command is not None:
                        self._on_command(msg)
                except Exception:
                    log.exception("command dispatcher raised")
        with self._clients_lock:
            if client in self._clients:
                self._clients.remove(client)
        try:
            client.close()
        except OSError:
            pass


__all__ = ["ControlServer", "default_socket_path"]
