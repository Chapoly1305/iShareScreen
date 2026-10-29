# AMD versus QSV timing investigation

Both live sessions used HEVC 4:2:0 at 200% scaling, a 3840x2160 backing
canvas, four 3840x544 encoded tiles and the same AMD renderer. The user
dragged a remote window. These are sequential manual runs, not an identical
recorded input replay. Raw timing-only evidence is in
`amd-qsv-timing-2026-10-29.json`.

## Findings

Ranges below are the p95 values of three separate five-second windows,
not pooled percentiles. Times are milliseconds per operation.

| Measurement | AMD | QSV |
| --- | ---: | ---: |
| Display loop interval | 25.9–27.3 | 6.68–6.75 |
| Fetch fresh tile on display thread | 6.01–6.77 | 0.014 |
| Explicit GPU download per tile | 4.30–5.13 | Not a separate operation in this path |
| Decode queue wait | 0.82–0.88 | 0.14–0.18 |
| Decode API call | 2.84–3.12 | 1.87–1.93 |
| Texture upload per tile | 1.13–1.34 | 0.93–1.11 |
| Draw API call | 3.13–4.17 | 5.84–5.90 |

AMD's video engine averaged 73.07%, peaking at 75.03%; its rendering engine
averaged 13.11%. This does not demonstrate GPU saturation. QSV's Intel video
engine averaged 14.23%, Intel 3D 43.35%, and AMD rendering 14.5%. Different
engine labels/utilization percentages are not directly comparable throughput.

The main measured hotspot is AMD's synchronous download and conversion in
`HevcDecoder.get_frame()`, called by the display loop. QSV converts in its
decode worker and returns CPU data immediately. AMD's lifecycle-lock wait was
negligible. Queue waits were short; steady-state superseded tile updates were
few (1–5 per tile/window), unlike a growing decode backlog. AMD tile output
cadence was also less regular, so download removal alone is not proven to
solve every skip. Earlier background-copy experiments failed user testing.

The next optimization target is reducing synchronous GPU-to-CPU transfer on
the display thread, with correct frame ownership and bounded buffering.
This investigation does not claim a completed AMD playback fix.

## Limits and reproduction

Instrumentation uses `perf_counter` (Windows QueryPerformanceCounter).
An earlier capture using `monotonic` on this Python had coarse ~16 ms
resolution and is excluded. Startup windows are excluded; one retained AMD
window contains an isolated ~748 ms tile-0 publication gap. It must not be
attributed to GPU download without further evidence.

The temporary profiling instrumentation emitted five-second summaries of
p50/p95/p99/max stage timings and superseded updates. It was removed after
the live fix was validated; the saved JSON retains the historical evidence.

Draw calls and tile uploads are not physical monitor presentation. There is
no measured server-to-monitor end-to-end latency or frame-ID trace here.
Windows system tracing was attempted but WPR returned 0xc5585011 (unable to
enable system profiling policy). An elevated WPR GPU/CPU/DesktopComposition
capture is still needed to distinguish driver/GPU waits and compositor timing.
