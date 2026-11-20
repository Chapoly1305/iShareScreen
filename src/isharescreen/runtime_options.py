"""Curated user options and a separate developer diagnostic catalog.

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

    @property
    def dest(self) -> str:
        return "runtime_" + self.env.removeprefix("ISS_").lower()


B = bool
_CATALOG = (
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
    RuntimeOption("ISS_NALU_DUMP", "nalu-dump", "write decoded-input NAL units to this path"),
    RuntimeOption("ISS_FAKE_DISPLAY_RECTS", "fake-display-rects", "override display layout for testing (existing rectangle syntax)"),
    RuntimeOption("ISS_DROP_PCT", "drop-pct", "simulated inbound packet loss percentage", number(0, 100)),
    RuntimeOption("ISS_DROP_BURST_EVERY_S", "drop-burst-every-s", "synthetic loss burst interval in seconds (0 disables)", number(0, 86400)),
    RuntimeOption("ISS_DROP_AFTER_S", "drop-after-s", "delay before simulated loss starts, in seconds (default 8)", number(0, 86400)),
    *(RuntimeOption("ISS_" + name, name.lower().replace("_", "-"), help, B)
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

# Keep normal CLI settings tied to user-visible behavior. Protocol experiments,
# probe overrides, and per-subsystem diagnostics belong to the developer entry.
_USER_FLAGS = frozenset({
    "rctl", "rctl-bwe", "clipboard", "udp-bind-host", "video-bitrate-bps",
    "vsync", "present-fps", "cursor-scale", "ctrl-as-cmd", "grab-keyboard",
    "scale-filter", "sharpen",
})
OPTIONS = tuple(o for o in _CATALOG if o.flag in _USER_FLAGS)
LAUNCHER_OPTIONS = tuple(o for o in _CATALOG if o.flag == "session-choice-url")
DEVELOPER_OPTIONS = tuple(o for o in _CATALOG if o not in OPTIONS + LAUNCHER_OPTIONS)
_DEVELOPER_BY_NAME = {o.flag: o for o in DEVELOPER_OPTIONS}


# Already exposed by the main CLI, with codec/decoder cross-validation.
EXISTING_OPTIONS = {
    "ISS_VIDEO_CODEC": "--codec", "ISS_VIDEO_CHROMA": "--chroma", "ISS_DECODER": "--decoder",
}


def parse_developer_option(raw: str) -> tuple[RuntimeOption, object]:
    name, separator, value = raw.partition("=")
    option = _DEVELOPER_BY_NAME.get(name)
    if option is None:
        raise argparse.ArgumentTypeError(
            f"unknown developer option {name!r}; see docs/runtime-options.md"
        )
    if not separator or not value:
        raise argparse.ArgumentTypeError("expected NAME=VALUE")
    if option.kind is bool:
        if value not in ("0", "1", "true", "false", "on", "off"):
            raise argparse.ArgumentTypeError(f"{name}: expected on/off, true/false, or 1/0")
        parsed = value in ("1", "true", "on")
    else:
        try:
            parsed = option.kind(value)
        except argparse.ArgumentTypeError as error:
            raise argparse.ArgumentTypeError(f"{name}: {error}") from error
        if option.choices and parsed not in option.choices:
            raise argparse.ArgumentTypeError(f"{name}: expected one of {', '.join(option.choices)}")
    return option, parsed


def add_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("runtime tuning")
    for option in OPTIONS + LAUNCHER_OPTIONS:
        kwargs = dict(dest=option.dest, default=None,
                      help=option.help if option in OPTIONS else argparse.SUPPRESS)
        if option.kind is bool:
            kwargs["action"] = argparse.BooleanOptionalAction
        else:
            kwargs["type"] = option.kind
            if option.choices:
                kwargs["choices"] = option.choices
        group.add_argument("--" + option.flag, **kwargs)
    group.add_argument(
        "--dev-option", metavar="NAME=VALUE", action="append", default=[],
        type=parse_developer_option,
        help="developer diagnostics/experiments; repeatable (see docs/runtime-options.md)",
    )


def apply_arguments(args: argparse.Namespace) -> None:
    """Bridge validated startup values to legacy consumers before native imports.

    Omitted flags inherit the environment and existing platform defaults.
    Boolean false must be written as '0', including for default-on features.
    """
    values = [(o, getattr(args, o.dest, None)) for o in OPTIONS + LAUNCHER_OPTIONS]
    values.extend(getattr(args, "dev_option", []))
    for option, value in values:
        if value is None:
            continue
        if option.env == "ISS_VIDEO_BITRATE_BPS" and value == 0:
            os.environ.pop(option.env, None)
        else:
            os.environ[option.env] = ("1" if value else "0") if option.kind is bool else str(value)
