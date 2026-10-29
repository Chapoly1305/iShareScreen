# myMain integration review — 2026-10-28

## Baseline and recovery

The original `myMain` tip was `7eca6ba`. It is preserved locally as
`backup/myMain-before-integration-20261028`.

Merge `5e2fcaa` retains both branch histories and resolves the tree to current
`main` (`9ab40d3`). This deliberately replaces older implementations before
selectively restoring the useful fork changes. The initial merge had 26
conflicted files; simply preferring the old branch would lose newer AVC
recovery, the browser connect form, multiple-monitor support, and QSV fixes.

## Upstream PRs incorporated

| Source | Decision | Integration follow-up |
| --- | --- | --- |
| [PR #24](https://github.com/renegadelink/iShareScreen/pull/24) | Take both the upload flush and macOS AVC capability check. | Make its simulated macOS test independent of the local PyAV build's device list, which PR #27 now filters. |
| [PR #25](https://github.com/renegadelink/iShareScreen/pull/25) | Take Mac Option-to-Meta mapping and printable-key handling in both desktop callbacks, plus browser mapping. | Live Mac/browser keyboard validation remains necessary. |
| [PR #27](https://github.com/renegadelink/iShareScreen/pull/27) | Take supported-device filtering, the larger HEVC probe sample, NVDEC support, and hardware reporting fixes. | Add probe decision tests and an actual software decode of the 160×160 4:4:4 sample. |
| [PR #28](https://github.com/renegadelink/iShareScreen/pull/28) | Take Linux compositor identity. | Restrict initialization to Linux; tolerate a missing string-hint API and only request the Wayland hint with native GLFW 3.4+. |

The original authors and commit references are preserved through cherry-picks.

## Fork changes selectively retained

- Serialize HEVC start/restart/close/software-fallback transitions with a
  lifecycle lock, independent of the decode worker's codec lock.
- Atomically claim throttled decoder restarts across video and watchdog
  threads. Keep current upstream's mandatory reset for each new AVC SSRC
  generation; that reset must bypass the HEVC debounce interval.
- Correct both Win32 hook return types to pointer-sized values. Retain and
  reuse the callback across disable/re-enable instead of freeing an address
  Windows may still call. This improves on the old patch, which retained the
  callback on disable but replaced it on the next enable.
- Probe generic libav hardware and QSV independently so both can be offered.
  Retain current upstream's default selection priority; explicit QSV selection
  remains available even when generic hardware works.
- Let a supported decoder override determine the codec when codec is `auto`.
  An explicit codec still wins, and a decoder for the wrong codec is rejected
  instead of being constructed for an incompatible stream.

## Older behavior deliberately superseded

- The Textual UI and its decoder selector: upstream replaced that interface.
- The older AVC decoder and frame-counter/POC-wrap workaround: retain current
  upstream's SPS handling, reference recovery, RTP repair window, and D3D11
  fallback/re-anchor behavior.
- The older QSV restart implementation: upstream already contains the latency
  tuning and frame-lifetime fix, and additionally preserves the DPB on ordinary
  restart requests while rebuilding on real parameter changes.
- A blanket preference for QSV over generic hardware: the preferred device
  depends on the machine; retain upstream defaults and manual selection.
- The old unconditional AVC LTR advertisement and four-tile default: retain
  upstream's codec-aware negotiation and single-picture AVC default.
- FIR-exhaustion-triggered restarts from the old fork: retain the newer
  loss-aware recovery policy rather than reintroducing shared-DPB resets.
- Older GPU allocation/cropping code: retain lazy biplanar setup and current
  per-monitor rendering/crop support.

## Validation

- Full suite: **118 passed**, including new concurrency, decoder selection,
  hardware probe, GLFW compatibility, and Win32 ABI/callback-lifetime tests.
- CLI `--help` exits successfully.
- No merge markers or whitespace errors in the integrated changes.
- Tested with Python 3.12.14, PyAV 19.0.2, GLFW bindings 2.11.0,
  rendercanvas 2.7.2, and wgpu 0.32.0 in an isolated local environment.

Tests exercise recovery logic and mocked hardware interfaces; they do not
establish live GPU or remote-host correctness. Remaining live checks: macOS
AVC resize between software/hardware-capable canvases; Mac Option characters
and shortcuts in desktop/browser; Linux NVIDIA decoding; Wayland/X11 identity
with multiple windows; background-window memory behavior; Windows QSV and
keyboard focus changes during a real session.

No enhancement/training branches or changes from external forks linked in PR
discussions were incorporated in this integration.
