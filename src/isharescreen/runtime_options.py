"""CLI catalog for runtime and diagnostic settings.

Legacy ISS_* variables remain a compatibility backend. Resolve explicit flags
before importing the session/frontends: several native modules snapshot settings
at import time. The catalog is the single place that defines their CLI surface.
"""
from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Callable


def number(minimum: float, maximum: float, *, integer: bool = False) -> Callable:
    def parse(raw: str):
        try:
            value = int(raw) if integer else float(raw)
        except ValueError as error:
            raise argparse.ArgumentTypeError("expected an integer" if integer else "expected a number") from error
        if not math.isfinite(value) or not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError(f"expected a value between {minimum:g} and {maximum:g}")
        return value
    return parse


def uint32(raw: str) -> int:
    try:
        value = int(raw, 16 if raw.lower().startswith("0x") else 10)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a decimal or hexadecimal integer") from error
    if not 0 <= value <= 0xFFFFFFFF:
        raise argparse.ArgumentTypeError("expected an unsigned 32-bit integer")
    return value


@dataclass(frozen=True)
class RuntimeOption:
    env: str
    flag: str
    help: str
    kind: object = str
    choices: tuple[str, ...] | None = None
    group: str = "runtime tuning"

    @property
    def dest(self) -> str:
        return "runtime_" + self.env.removeprefix("ISS_").lower()


B = bool
DEBUG = "diagnostics and fault injection"
OPTIONS = (
    RuntimeOption("ISS_RCTL", "rctl", "receiver rate control (default on)", B),
    RuntimeOption("ISS_RCTL_BWE", "rctl-bwe", "fixed receiver bandwidth estimate in kbps (otherwise adaptive)", number(1, 65535, integer=True)),
    RuntimeOption("ISS_CLIP", "clipboard", "bidirectional clipboard synchronization (default on)", B),
    RuntimeOption("ISS_MEDIA_FEEDBACK", "media-feedback", "media feedback routing (default legacy)", choices=("legacy", "rr")),
    RuntimeOption("ISS_UDP_BIND_HOST", "udp-bind-host", "local address for UDP media sockets"),
    RuntimeOption("ISS_VIDEO_BITRATE_BPS", "video-bitrate-bps", "advertised video bandwidth cap in bits/s; 0 restores protocol defaults", number(0, 0xFFFFFFFF, integer=True)),
    RuntimeOption("ISS_TILES_PER_FRAME", "tiles-per-frame", "requested pictures per frame (browser default 1, otherwise 4)", number(1, 65535, integer=True)),
    RuntimeOption("ISS_LTRP", "ltrp", "HEVC long-term reference pictures (default on)", B),
    RuntimeOption("ISS_AVC_HWACCEL", "avc-hwaccel", "H.264 hardware acceleration (default on)", B),
    RuntimeOption("ISS_FORCE_SW_DECODE", "force-sw-decode", "force software H.264 decoding", B),
    RuntimeOption("ISS_FORCE_SW_HEVC", "force-sw-hevc", "force software HEVC decoding", B),
    RuntimeOption("ISS_HEVC420", "hevc420", "override HEVC 4:2:0 hardware capability detection", B),
    RuntimeOption("ISS_HEVC444", "hevc444", "override HEVC 4:4:4 hardware capability detection", B),
    RuntimeOption("ISS_AUDIO_F4", "audio-f4", "experimental audio capability mask (decimal or hex)", uint32),
    RuntimeOption("ISS_LEGACY_CURSOR", "legacy-cursor", "use the legacy cursor negotiation", B),
    RuntimeOption("ISS_LEGACY_CHROMA", "legacy-chroma", "use legacy HEVC chroma handling", B),
    RuntimeOption("ISS_PERTILE_RECOVERY", "pertile-recovery", "HEVC per-tile recovery (default on)", B),
    RuntimeOption("ISS_AVC_SPS_PATCH", "avc-sps-patch", "patch H.264 SPS for local decoding (default on)", B),
    RuntimeOption("ISS_BROWSER_SPS_PATCH", "browser-sps-patch", "patch H.264 SPS for browser decoding (default on)", B),
    RuntimeOption("ISS_VSYNC", "vsync", "desktop display synchronization (macOS default off, other platforms on)", B),
    RuntimeOption("ISS_PRESENT_FPS", "present-fps", "desktop redraw limit in frames/s (default 120)", number(30, 1000)),
    RuntimeOption("ISS_GPU_BACKEND", "gpu-backend", "wgpu backend override (e.g. Vulkan, Metal, D3D12)"),
    RuntimeOption("ISS_GPU_FALLBACK", "gpu-fallback", "request the fallback GPU adapter", B),
    RuntimeOption("ISS_LOCAL_CURSOR", "local-cursor", "use the OS cursor instead of the canvas cursor", B),
    RuntimeOption("ISS_CURSOR_SCALE", "cursor-scale", "cursor size calibration multiplier (default 1)", number(0.01, 100)),
    RuntimeOption("ISS_CTRL_AS_CMD", "ctrl-as-cmd", "map Control to macOS Command", B),
    RuntimeOption("ISS_GRAB_KEYBOARD", "grab-keyboard", "grab keyboard shortcuts in desktop viewer (default on)", B),
    RuntimeOption("ISS_SCALE_FILTER", "scale-filter", "desktop upscaling filter (default sharp)", choices=("sharp", "linear")),
    RuntimeOption("ISS_SHARPEN", "sharpen", "desktop sharpening strength, 0..1 (default 0)", number(0, 1)),
    RuntimeOption("ISS_WT_VIDEO_SINGLE_STREAM", "wt-video-single-stream", "experimental single WebTransport video stream", B),
    RuntimeOption("ISS_WT_SINGLE_STREAM_HIGH_WATER", "wt-single-stream-high-water", "WebTransport video buffering high-water mark in bytes", number(1, 0xFFFFFFFF, integer=True)),
    RuntimeOption("ISS_WT_LOCAL_DECODE", "wt-local-decode", "also decode browser video locally", B),
    RuntimeOption("ISS_SESSION_CHOICE_URL", "session-choice-url", "launcher callback URL for choosing the remote session"),
    RuntimeOption("ISS_NALU_DUMP", "nalu-dump", "write decoded-input NAL units to this path", group=DEBUG),
    RuntimeOption("ISS_FAKE_DISPLAY_RECTS", "fake-display-rects", "override display layout for testing (existing rectangle syntax)", group=DEBUG),
    RuntimeOption("ISS_DROP_PCT", "drop-pct", "simulated inbound packet loss percentage", number(0, 100), group=DEBUG),
    RuntimeOption("ISS_DROP_BURST_EVERY_S", "drop-burst-every-s", "synthetic loss burst interval in seconds (0 disables)", number(0, 86400), group=DEBUG),
    RuntimeOption("ISS_DROP_AFTER_S", "drop-after-s", "delay before simulated loss starts, in seconds (default 8)", number(0, 86400), group=DEBUG),
    *(RuntimeOption("ISS_" + name, name.lower().replace("_", "-"), help, B, group=DEBUG)
      for name, help in (
          ("CHROMA_TRACE", "trace decoded chroma"),
          ("CURSOR_DEBUG", "log cursor diagnostics"),
          ("DPB_TRACE", "trace decoded picture buffer recovery"),
          ("LOG_LIBAV", "forward libav diagnostics (default on)"),
          ("LOG_LTR", "log long-term reference acknowledgments"),
          ("LOG_RFB_IN", "log incoming RFB messages"),
          ("LOG_RFB_OUT", "log outgoing RFB messages"),
          ("VT_DEBUG", "log VideoToolbox diagnostics"),
          ("WT_DEBUG_SETTINGS", "log WebTransport settings"),
          ("WT_PROFILE", "profile WebTransport and video receive processing"),
      )),
)

# Already exposed by the main CLI, with codec/decoder cross-validation.
EXISTING_OPTIONS = {
    "ISS_VIDEO_CODEC": "--codec", "ISS_VIDEO_CHROMA": "--chroma", "ISS_DECODER": "--decoder",
}


def add_arguments(parser: argparse.ArgumentParser) -> None:
    groups = {}
    for option in OPTIONS:
        if option.group not in groups:
            groups[option.group] = parser.add_argument_group(option.group)
        kwargs = dict(dest=option.dest, default=None,
                      help=f"{option.help}; overrides {option.env}")
        if option.kind is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = option.kind
            if option.choices:
                kwargs["choices"] = option.choices
        groups[option.group].add_argument("--" + option.flag, **kwargs)


def apply_arguments(args: argparse.Namespace) -> None:
    """Bridge explicit CLI values to legacy consumers before native imports.

    Omitted flags inherit the environment and existing platform defaults.
    Boolean false must be written as '0', including for default-on features.
    """
    for option in OPTIONS:
        value = getattr(args, option.dest, None)
        if value is None:
            continue
        if option.env == "ISS_VIDEO_BITRATE_BPS" and value == 0:
            os.environ.pop(option.env, None)
        else:
            os.environ[option.env] = ("1" if value else "0") if option.kind is bool else str(value)
