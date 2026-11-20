# Runtime command-line options

Normal CLI parameters cover connection, media selection, and visible viewer
behavior. Run `iss --help` for their grouped help. Low-level experiments and
diagnostics use the separate repeatable `--dev-option NAME=VALUE` entry.
Explicit parameters override legacy environment variables. Omitted parameters
preserve the environment or platform default.

```sh
iss --host mac.local -u alex --frontend desktop --no-rctl
iss --host mac.local -u alex --frontend desktop --rctl-bwe 30000
iss --host mac.local -u alex --frontend desktop --present-fps 240 --no-vsync
iss --host mac.local -u alex --video-bitrate-bps 30000000 --no-clipboard
```

RCTL defaults to enabled and adaptive. `--rctl-bwe` pins its feedback estimate
in **kbps** (1–65535); it does not enable RCTL when `--no-rctl` is set.
`--video-bitrate-bps` is the separate media-offer bandwidth cap in **bits/s**;
`0` clears an inherited cap and restores protocol defaults. Neither option
promises a constant actual stream rate.

## User options

Boolean parameters accept both `--name` and `--no-name`.

| Parameter | Legacy variable | Purpose |
| --- | --- | --- |
| `--codec` | `ISS_VIDEO_CODEC` | Validated video selection |
| `--chroma` | `ISS_VIDEO_CHROMA` | Validated video selection |
| `--decoder` | `ISS_DECODER` | Validated video selection |
| `--rctl` / `--no-rctl` | `ISS_RCTL` | receiver rate control (default on) |
| `--rctl-bwe VALUE` | `ISS_RCTL_BWE` | fixed receiver bandwidth estimate in kbps (otherwise adaptive) |
| `--clipboard` / `--no-clipboard` | `ISS_CLIP` | bidirectional clipboard synchronization (default on) |
| `--udp-bind-host VALUE` | `ISS_UDP_BIND_HOST` | local address for UDP media sockets |
| `--video-bitrate-bps VALUE` | `ISS_VIDEO_BITRATE_BPS` | advertised video bandwidth cap in bits/s; 0 restores protocol defaults |
| `--vsync` / `--no-vsync` | `ISS_VSYNC` | desktop display synchronization (macOS default off, other platforms on) |
| `--present-fps VALUE` | `ISS_PRESENT_FPS` | desktop redraw limit in frames/s (default 120) |
| `--cursor-scale VALUE` | `ISS_CURSOR_SCALE` | cursor size calibration multiplier (default 1) |
| `--ctrl-as-cmd` / `--no-ctrl-as-cmd` | `ISS_CTRL_AS_CMD` | map Control to macOS Command |
| `--grab-keyboard` / `--no-grab-keyboard` | `ISS_GRAB_KEYBOARD` | grab keyboard shortcuts in desktop viewer (default on) |
| `--scale-filter VALUE` | `ISS_SCALE_FILTER` | desktop upscaling filter (default sharp) |
| `--sharpen VALUE` | `ISS_SHARPEN` | desktop sharpening strength, 0..1 (default 0) |

## Developer diagnostics and experiments

These settings are not standalone CLI parameters. They are retained for protocol
research, reproducing failures, and comparing implementations. Routine use should
rely on automatic hardware detection, `--decoder`, and `--verbose` instead of
individual probe overrides or subsystem traces.

```sh
iss --host mac.local -u alex --headless --dev-option drop-pct=2 --dev-option drop-after-s=8
iss --host mac.local -u alex --dev-option wt-profile=on
iss --host mac.local -u alex --frontend desktop --dev-option gpu-backend=Vulkan
```

Only the names below are accepted; arbitrary environment names are rejected.
Boolean values are `on`/`off`, `true`/`false`, or `1`/`0`. Numeric ranges are
validated. Repeat `--dev-option` to set multiple values; the last value for a
repeated name wins. Legacy environment variables remain supported for existing
scripts and regression tests.

| Developer name (`NAME=VALUE`) | Legacy variable | Retained purpose |
| --- | --- | --- |
| `media-feedback` | `ISS_MEDIA_FEEDBACK` | media feedback routing (default legacy) |
| `tiles-per-frame` | `ISS_TILES_PER_FRAME` | requested pictures per frame (browser default 1, otherwise 4) |
| `ltrp` | `ISS_LTRP` | HEVC long-term reference pictures (default on) |
| `avc-hwaccel` | `ISS_AVC_HWACCEL` | H.264 hardware acceleration (default on) |
| `force-sw-decode` | `ISS_FORCE_SW_DECODE` | force software H.264 decoding |
| `force-sw-hevc` | `ISS_FORCE_SW_HEVC` | force software HEVC decoding |
| `hevc420` | `ISS_HEVC420` | override HEVC 4:2:0 hardware capability detection |
| `hevc444` | `ISS_HEVC444` | override HEVC 4:4:4 hardware capability detection |
| `audio-f4` | `ISS_AUDIO_F4` | experimental audio capability mask (decimal or hex) |
| `legacy-cursor` | `ISS_LEGACY_CURSOR` | use the legacy cursor negotiation |
| `pertile-recovery` | `ISS_PERTILE_RECOVERY` | HEVC per-tile recovery (default on) |
| `avc-sps-patch` | `ISS_AVC_SPS_PATCH` | patch H.264 SPS for local decoding (default on) |
| `browser-sps-patch` | `ISS_BROWSER_SPS_PATCH` | patch H.264 SPS for browser decoding (default on) |
| `gpu-backend` | `ISS_GPU_BACKEND` | wgpu backend override (e.g. Vulkan, Metal, D3D12) |
| `gpu-fallback` | `ISS_GPU_FALLBACK` | request the fallback GPU adapter |
| `local-cursor` | `ISS_LOCAL_CURSOR` | use the OS cursor instead of the canvas cursor |
| `wt-video-single-stream` | `ISS_WT_VIDEO_SINGLE_STREAM` | experimental single WebTransport video stream |
| `wt-single-stream-high-water` | `ISS_WT_SINGLE_STREAM_HIGH_WATER` | WebTransport video buffering high-water mark in bytes |
| `wt-local-decode` | `ISS_WT_LOCAL_DECODE` | also decode browser video locally |
| `nalu-dump` | `ISS_NALU_DUMP` | write decoded-input NAL units to this path |
| `fake-display-rects` | `ISS_FAKE_DISPLAY_RECTS` | override display layout for testing (existing rectangle syntax) |
| `drop-pct` | `ISS_DROP_PCT` | simulated inbound packet loss percentage |
| `drop-burst-every-s` | `ISS_DROP_BURST_EVERY_S` | synthetic loss burst interval in seconds (0 disables) |
| `drop-after-s` | `ISS_DROP_AFTER_S` | delay before simulated loss starts, in seconds (default 8) |
| `chroma-trace` | `ISS_CHROMA_TRACE` | trace decoded chroma |
| `cursor-debug` | `ISS_CURSOR_DEBUG` | log cursor diagnostics |
| `dpb-trace` | `ISS_DPB_TRACE` | trace decoded picture buffer recovery |
| `log-libav` | `ISS_LOG_LIBAV` | forward libav diagnostics (default on) |
| `log-ltr` | `ISS_LOG_LTR` | log long-term reference acknowledgments |
| `log-rfb-in` | `ISS_LOG_RFB_IN` | log incoming RFB messages |
| `log-rfb-out` | `ISS_LOG_RFB_OUT` | log outgoing RFB messages |
| `vt-debug` | `ISS_VT_DEBUG` | log VideoToolbox diagnostics |
| `wt-debug-settings` | `ISS_WT_DEBUG_SETTINGS` | log WebTransport settings |
| `wt-profile` | `ISS_WT_PROFILE` | profile WebTransport and video receive processing |

## Removed and internal options

- `--legacy-chroma` and `ISS_LEGACY_CHROMA` were removed. The decoder only
  assigned the variable; no active path read it, so it had no effect.
- Historical A/B toggles (`avc-sps-patch`, `browser-sps-patch`,
  `pertile-recovery`, `legacy-cursor`) are developer settings because changing
  them bypasses normal compatibility/recovery behavior.
- Hardware probe overrides and force-software switches are developer settings;
  normal decoder choice uses `--decoder` and `--list-decoders`.
- Transport experiments, synthetic packet loss, fake layouts, NAL dumps, and
  subsystem traces are developer settings rather than user-facing flags.
- `--session-choice-url` is a private launcher argument, omitted from help. The
  GUI uses it for the share/alternate-session dialog. It remains supported for
  the launcher; it is not a viewer preference.

## Compatibility and initialization

The catalog in `runtime_options.py` validates CLI values and translates explicit
settings to the existing environment-backed engine before importing Session or
frontend/native modules. This preserves import-time native configuration and
legacy launch scripts. It is process-wide startup configuration, not an API for
independently configuring simultaneous sessions or changing settings live.
Existing codec/chroma/decoder cross-validation is retained.

The GUI forwards its bandwidth slider and session-choice callback through CLI
parameters. Passwords continue to travel through stdin.

System variables such as `XDG_STATE_HOME`, `LOCALAPPDATA`, and `CONDA_PREFIX`
retain their platform semantics. `ISS_NATIVE_TESTS` is a pytest-only opt-in,
not an application runtime option.
