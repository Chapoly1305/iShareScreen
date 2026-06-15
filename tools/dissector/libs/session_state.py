from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SessionConfig:
    stream_id: str
    initial_key_hex: str
    client_cbc_start_offset: int | None = None
    server_cbc_start_offset: int | None = None


@dataclass
class SessionRuntime:
    cbc_key_hex: str | None = None
    server_iv_hex: str | None = None
    client_iv_hex: str | None = None
    rekey_counter: int = 0
