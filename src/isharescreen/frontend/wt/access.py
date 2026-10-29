"""Local viewer authorization shared by HTTP bootstrap and WebTransport."""
from __future__ import annotations

import secrets
from urllib.parse import parse_qs, urlsplit


class ViewerAccess:
    def __init__(self, port: int):
        self.token = secrets.token_urlsafe(32)
        self.authorities = {f"localhost:{port}", f"127.0.0.1:{port}", f"[::1]:{port}"}
        self.origins = {f"{scheme}://{host}" for host in self.authorities
                        for scheme in ("http", "https")}

    def allows(self, authority: str, origin: str | None, token: str) -> bool:
        return (authority in self.authorities
                and (origin is None or origin in self.origins)
                and secrets.compare_digest(token.encode(), self.token.encode()))

    def allows_connect(self, headers: dict[bytes, bytes]) -> bool:
        path = headers.get(b":path", b"").decode("utf-8", errors="replace")
        token = parse_qs(urlsplit(path).query).get("token", [""])[0]
        return self.allows(
            headers.get(b":authority", b"").decode("utf-8", errors="replace"),
            headers.get(b"origin", b"").decode("utf-8", errors="replace") or None,
            token,
        )
