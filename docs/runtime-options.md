# Runtime command-line options

All application `ISS_*` settings have command-line equivalents. Run `iss --help`
for the complete grouped help. Flags take precedence over legacy environment
variables; an omitted flag preserves the environment or the platform default.
Boolean options accept both `--name` and `--no-name`.

```sh
iss --host mac.local -u alex --frontend desktop --no-rctl
iss --host mac.local -u alex --frontend desktop --rctl-bwe 30000
iss --host mac.local -u alex --frontend desktop --present-fps 240 --no-vsync
iss --host mac.local -u alex --video-bitrate-bps 30000000 --no-clipboard
iss --host mac.local -u alex --headless --drop-pct 2 --drop-after-s 8
```

RCTL defaults to enabled and adaptive. `--rctl-bwe` pins its feedback estimate
in **kbps** (1–65535); it does not enable RCTL when `--no-rctl` is set.
`--video-bitrate-bps` is the separate media-offer bandwidth cap in **bits/s**;
`0` clears an inherited cap and restores protocol defaults. Neither option
promises a constant actual stream rate.

The GUI bandwidth slider and session-choice callback are forwarded as CLI
parameters. Passwords continue to travel through stdin, not command-line flags.
Diagnostic and fault-injection options are intended for troubleshooting.

## Compatibility and initialization

The option catalog in `runtime_options.py` validates CLI values and translates
explicit flags to the existing environment-backed engine settings before
importing Session or frontend/native decoding modules. This retains compatibility
with existing launch scripts and import-time native settings. It is a process-wide
startup configuration, not an API for independently configuring simultaneous
sessions or changing settings during a session. Existing codec/chroma/decoder
cross-validation is retained.

System variables such as `XDG_STATE_HOME`, `LOCALAPPDATA`, and `CONDA_PREFIX`
are platform integration settings and retain their usual environment semantics.
`ISS_NATIVE_TESTS` is a pytest-only opt-in, not an application runtime option.

## Parameter reference

| Parameter | Legacy variable | Purpose |
| --- | --- | --- |
| `--codec` | `ISS_VIDEO_CODEC` | Existing validated video selection |
| `--chroma` | `ISS_VIDEO_CHROMA` | Existing validated video selection |
| `--decoder` | `ISS_DECODER` | Existing validated video selection |
| `--rctl` / `--no-rctl` | `ISS_RCTL` | receiver rate control (default on) |
| `--rctl-bwe VALUE` | `ISS_RCTL_BWE` | fixed receiver bandwidth estimate in kbps (otherwise adaptive) |
| `--clipboard` / `--no-clipboard` | `ISS_CLIP` | bidirectional clipboard synchronization (default on) |
| `--media-feedback VALUE` | `ISS_MEDIA_FEEDBACK` | media feedback routing (default legacy) |
| `--udp-bind-host VALUE` | `ISS_UDP_BIND_HOST` | local address for UDP media sockets |
| `--video-bitrate-bps VALUE` | `ISS_VIDEO_BITRATE_BPS` | advertised video bandwidth cap in bits/s; 0 restores protocol defaults |
| `--tiles-per-frame VALUE` | `ISS_TILES_PER_FRAME` | requested pictures per frame (browser default 1, otherwise 4) |
| `--ltrp` / `--no-ltrp` | `ISS_LTRP` | HEVC long-term reference pictures (default on) |
| `--avc-hwaccel` / `--no-avc-hwaccel` | `ISS_AVC_HWACCEL` | H.264 hardware acceleration (default on) |
| `--force-sw-decode` / `--no-force-sw-decode` | `ISS_FORCE_SW_DECODE` | force software H.264 decoding |
| `--force-sw-hevc` / `--no-force-sw-hevc` | `ISS_FORCE_SW_HEVC` | force software HEVC decoding |
| `--hevc420` / `--no-hevc420` | `ISS_HEVC420` | override HEVC 4:2:0 hardware capability detection |
| `--hevc444` / `--no-hevc444` | `ISS_HEVC444` | override HEVC 4:4:4 hardware capability detection |
| `--audio-f4 VALUE` | `ISS_AUDIO_F4` | experimental audio capability mask (decimal or hex) |
| `--legacy-cursor` / `--no-legacy-cursor` | `ISS_LEGACY_CURSOR` | use the legacy cursor negotiation |
| `--legacy-chroma` / `--no-legacy-chroma` | `ISS_LEGACY_CHROMA` | use legacy HEVC chroma handling |
| `--pertile-recovery` / `--no-pertile-recovery` | `ISS_PERTILE_RECOVERY` | HEVC per-tile recovery (default on) |
| `--avc-sps-patch` / `--no-avc-sps-patch` | `ISS_AVC_SPS_PATCH` | patch H.264 SPS for local decoding (default on) |
| `--browser-sps-patch` / `--no-browser-sps-patch` | `ISS_BROWSER_SPS_PATCH` | patch H.264 SPS for browser decoding (default on) |
| `--vsync` / `--no-vsync` | `ISS_VSYNC` | desktop display synchronization (macOS default off, other platforms on) |
| `--present-fps VALUE` | `ISS_PRESENT_FPS` | desktop redraw limit in frames/s (default 120) |
| `--gpu-backend VALUE` | `ISS_GPU_BACKEND` | wgpu backend override (e.g. Vulkan, Metal, D3D12) |
| `--gpu-fallback` / `--no-gpu-fallback` | `ISS_GPU_FALLBACK` | request the fallback GPU adapter |
| `--local-cursor` / `--no-local-cursor` | `ISS_LOCAL_CURSOR` | use the OS cursor instead of the canvas cursor |
| `--cursor-scale VALUE` | `ISS_CURSOR_SCALE` | cursor size calibration multiplier (default 1) |
| `--ctrl-as-cmd` / `--no-ctrl-as-cmd` | `ISS_CTRL_AS_CMD` | map Control to macOS Command |
| `--grab-keyboard` / `--no-grab-keyboard` | `ISS_GRAB_KEYBOARD` | grab keyboard shortcuts in desktop viewer (default on) |
| `--scale-filter VALUE` | `ISS_SCALE_FILTER` | desktop upscaling filter (default sharp) |
| `--sharpen VALUE` | `ISS_SHARPEN` | desktop sharpening strength, 0..1 (default 0) |
| `--wt-video-single-stream` / `--no-wt-video-single-stream` | `ISS_WT_VIDEO_SINGLE_STREAM` | experimental single WebTransport video stream |
| `--wt-single-stream-high-water VALUE` | `ISS_WT_SINGLE_STREAM_HIGH_WATER` | WebTransport video buffering high-water mark in bytes |
| `--wt-local-decode` / `--no-wt-local-decode` | `ISS_WT_LOCAL_DECODE` | also decode browser video locally |
| `--session-choice-url VALUE` | `ISS_SESSION_CHOICE_URL` | launcher callback URL for choosing the remote session |
| `--nalu-dump VALUE` | `ISS_NALU_DUMP` | write decoded-input NAL units to this path |
| `--fake-display-rects VALUE` | `ISS_FAKE_DISPLAY_RECTS` | override display layout for testing (existing rectangle syntax) |
| `--drop-pct VALUE` | `ISS_DROP_PCT` | simulated inbound packet loss percentage |
| `--drop-burst-every-s VALUE` | `ISS_DROP_BURST_EVERY_S` | synthetic loss burst interval in seconds (0 disables) |
| `--drop-after-s VALUE` | `ISS_DROP_AFTER_S` | delay before simulated loss starts, in seconds (default 8) |
| `--chroma-trace` / `--no-chroma-trace` | `ISS_CHROMA_TRACE` | trace decoded chroma |
| `--cursor-debug` / `--no-cursor-debug` | `ISS_CURSOR_DEBUG` | log cursor diagnostics |
| `--dpb-trace` / `--no-dpb-trace` | `ISS_DPB_TRACE` | trace decoded picture buffer recovery |
| `--log-libav` / `--no-log-libav` | `ISS_LOG_LIBAV` | forward libav diagnostics (default on) |
| `--log-ltr` / `--no-log-ltr` | `ISS_LOG_LTR` | log long-term reference acknowledgments |
| `--log-rfb-in` / `--no-log-rfb-in` | `ISS_LOG_RFB_IN` | log incoming RFB messages |
| `--log-rfb-out` / `--no-log-rfb-out` | `ISS_LOG_RFB_OUT` | log outgoing RFB messages |
| `--vt-debug` / `--no-vt-debug` | `ISS_VT_DEBUG` | log VideoToolbox diagnostics |
| `--wt-debug-settings` / `--no-wt-debug-settings` | `ISS_WT_DEBUG_SETTINGS` | log WebTransport settings |
| `--wt-profile` / `--no-wt-profile` | `ISS_WT_PROFILE` | profile WebTransport and video receive processing |
