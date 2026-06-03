"""Async subscriber for a Session's ControlServer.

Connects to a local control socket (UDS on POSIX, localhost TCP on
Windows via the `.port` sidecar file), reads newline-delimited JSON
messages forever, and dispatches them to a callback. Designed to be
driven by Textual's asyncio loop.

The first message after connect is always `{"type": "hello", ...}`. After
that come periodic `{"type": "snapshot", ...}` messages and occasional
`{"type": "event", ...}` ones; both are pushed to `on_message` in arrival
order. On read failure (server closed, peer reset) the client raises
`ConnectionResetError`; the caller decides whether to back off + reconnect
or surface a banner.

Phase 1 is read-only — `send_command` is wired for the next phase.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional


log = logging.getLogger("iss.tui.ctrl")

# Maximum wait per attempt for the server to bind its socket after the
# child process is spawned. Negotiation + key derivation can take a few
# seconds, plus startup-burst reconnects up to ~30 s on a wedged host.
_WAIT_FOR_SOCKET_TIMEOUT_S = 45.0
_WAIT_POLL_INTERVAL_S = 0.2


async def wait_for_socket(addr: str, timeout: float = _WAIT_FOR_SOCKET_TIMEOUT_S) -> None:
    """Block until `addr` is a connectable control socket, or `timeout`
    elapses. POSIX checks the UDS path; Windows checks the `.port` sidecar
    and tries to TCP-connect."""
    deadline = asyncio.get_event_loop().time() + timeout
    if sys.platform == "win32":
        port_file = addr + ".port"
        while asyncio.get_event_loop().time() < deadline:
            if Path(port_file).exists():
                try:
                    port = int(Path(port_file).read_text().strip())
                    r, w = await asyncio.open_connection("127.0.0.1", port)
                    w.close()
                    try:
                        await w.wait_closed()
                    except Exception:
                        pass
                    return
                except (OSError, ValueError):
                    pass
            await asyncio.sleep(_WAIT_POLL_INTERVAL_S)
        raise TimeoutError(f"control socket {addr!r} did not come up in {timeout:.0f}s")
    # POSIX UDS
    while asyncio.get_event_loop().time() < deadline:
        if os.path.exists(addr):
            try:
                r, w = await asyncio.open_unix_connection(addr)
                w.close()
                try:
                    await w.wait_closed()
                except Exception:
                    pass
                return
            except OSError:
                pass
        await asyncio.sleep(_WAIT_POLL_INTERVAL_S)
    raise TimeoutError(f"control socket {addr!r} did not come up in {timeout:.0f}s")


class ControlClient:
    """One client connection to a Session's ControlServer. Subscribes for
    the lifetime of the session; the caller owns the reconnect policy."""

    def __init__(
        self,
        addr: str,
        on_message: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        self._addr = addr
        self._on_message = on_message
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._task: Optional[asyncio.Task] = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        """Open the connection (waiting for the socket to come up if
        needed) and spawn the read task. Returns once the read task is
        running; messages start flowing via `on_message`."""
        await wait_for_socket(self._addr)
        if sys.platform == "win32":
            port = int(Path(self._addr + ".port").read_text().strip())
            self._reader, self._writer = await asyncio.open_connection("127.0.0.1", port)
        else:
            self._reader, self._writer = await asyncio.open_unix_connection(self._addr)
        self._task = asyncio.create_task(self._read_loop(), name="iss-tui-ctrl-rx")

    async def send_command(self, action: str, **fields: Any) -> None:
        """Send a JSON command. Phase 2: actual handling on the server."""
        if self._writer is None:
            return
        msg = {"action": action}
        msg.update(fields)
        line = (json.dumps(msg, separators=(",", ":")) + "\n").encode("utf-8")
        self._writer.write(line)
        try:
            await self._writer.drain()
        except OSError as e:
            log.debug("control command send failed: %s", e)

    async def stop(self) -> None:
        """Cancel the read loop and close the connection. Idempotent."""
        self._stop.set()
        if self._task is not None and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except OSError:
                pass

    # ── internals ──────────────────────────────────────────────────

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while not self._stop.is_set():
                line = await self._reader.readline()
                if not line:
                    log.debug("control server closed the connection")
                    break
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError as e:
                    log.warning("malformed control message (%s): %r", e, line[:80])
                    continue
                try:
                    await self._on_message(msg)
                except Exception:
                    log.exception("on_message handler raised")
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("control read loop crashed")


__all__ = ["ControlClient", "wait_for_socket"]
