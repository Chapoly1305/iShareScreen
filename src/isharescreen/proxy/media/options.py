"""Resolve legacy decoder settings once, before session startup."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class DecoderOptions:
    override: str = "auto"
    prefer_hardware: bool = True
    avc_hardware: bool = True

    @classmethod
    def from_environment(cls, override: str | None = None) -> DecoderOptions:
        return cls(
            override=override or os.environ.get("ISS_DECODER", "auto"),
            prefer_hardware=os.environ.get("ISS_FORCE_SW_HEVC", "0") == "0",
            avc_hardware=os.environ.get("ISS_AVC_HWACCEL", "1") != "0",
        )
