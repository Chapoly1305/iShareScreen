# AMD system trace findings

Source: `amd-lag.etl`, captured October 29, 2026, 01:16:05–01:17:18
America/New_York. Duration 73.264 seconds, no lost ETW events/buffers.
AMD viewer PID 24504. Structured results: `amd-system-trace-2026-10-29.json`.

## Displayed frames

PresentMon 2.6.0 processed the saved trace offline. In the steady AMD window
25–45 seconds after trace start:

| Measurement | Result |
| --- | ---: |
| Presented frames | 1183 in 20 seconds (about 59 FPS) |
| Marked dropped by PresentMon | 0 |
| Display change interval, median | 17.15 ms |
| Display change interval, p95 / p99 | 28.56 / 34.28 ms |
| Longest display change interval | 40.04 ms |
| Display intervals exceeding 25 ms | 76 |
| Present-to-render-complete, p95 | 0.489 ms |
| Present-to-display, p95 | 6.723 ms |
| Present API duration, p95 | 0.311 ms |

This is uneven delivery before presentation, not a large queue of submitted
frames being discarded by the compositor. It does not rule out decoded tile
updates being superseded earlier. Displayed client frames are not necessarily
new complete remote frames: a draw can update one tile or only the cursor.

The trace also contains a short QSV session preceding the AMD connection.
Its 0.3–5 second window has display-change p95 5.84 ms (many extra redraws),
3 intervals over 25 ms, and 3 dropped presentations. This is supporting
context, not a matched replay or a claim that the remote stream is 175 FPS.

## GPU packets and the hotspot

DxgKrnl Context and NodeMetadata events map the viewer's decoder context to
node 14, `Video Codec 0`, and its other contexts to node 0, 3D. QueuePacket and
DmaPacket events were paired by context and queue submission sequence over
trace seconds 25–35. Complete matched packets only:

- Decoder: 2352 packets; scheduled execution-span median 4.174 ms, p95
  5.620 ms; queue-wait p95 1.266 ms.
- Decoder DMA intervals overlap. Their union covers **72.45%** of the window;
  summing durations would incorrectly suggest 100% utilization. This agrees
  with the earlier ~73% engine-counter measurement.
- Two contexts submitting from display thread 3864 have queue-wait p95
  0.150/0.173 ms and execution-span p95 0.740/0.532 ms.
- A further 3D context on the decoder's D3D11 device, submitting from thread
  27952, has short execution spans (p95 0.349 ms) but queue-wait p95 3.319 ms.
  Its exact operation is not established solely by these events.

Together with the application timings (fresh-frame retrieval about
630–645 ms per second, short decode queue waits, GPU-download p95 around
4–5 ms per tile), this identifies synchronous frame retrieval/download on
the display thread as the main measured hot path. GPU rendering is small.
This does not establish that moving a copy to any worker will fix it; previous
background-copy experiments failed user testing. A fix must preserve decoder
reference frames, frame ownership, tile freshness and bounded buffering.

## Analysis method and limits

The installed WPA/XPerf parser rejected a CLR rundown event with 0x80070032.
Windows ITraceRelogger made an analysis copy without CLR rundown metadata.
A smaller copy also excluded syscall Enter/Exit and page-fault events; GPU,
display, CPU samples and scheduling events were preserved. The original ETL
was not modified. XPerf exported a 10-second DxgKrnl event window for packet
pairing and a small context/engine metadata trace. The general WPA exporter
was stopped after expensive processing; the results above come from XPerf
event extraction and PresentMon, not a completed WPA table export.

After explicit user approval, the portable executable was downloaded from
the [official PresentMon release](https://github.com/GameTechDev/PresentMon/releases/tag/v2.6.0)
and run locally. No trace was uploaded. PresentMon CSVs and local analysis
scripts are under `.venv-review/amd-trace/`.

These are observations during profiling, which can add overhead. DMA spans
are scheduling/completion intervals, not shader timestamps. No measurement
here gives remote input-to-photon latency. Startup reference-frame recovery
was excluded from the steady display comparison. AMD playback is not yet
fixed by this instrumentation.

## Follow-up fix and live validation

AMD now submits a bounded batch of up to four decoded GPU outputs before
downloading them on the decode producer itself. Partial batches flush after
at most 4 ms of collection wait, so unchanged tiles cannot stall publication.
The display consumer receives CPU-owned TileFrames and never performs AMD
readback. Notifications occur after conversion. The shared HEVC reference
context and encoded-frame order remain intact. Shutdown joins the producer
before releasing its device. This differs from the abandoned extra download
thread, which competed with the producer for GPU access.

The user reconnected at 200% and reported: “good i think its fixed.” In
`gui-9984-6.log`, the 01:39:05 window recorded:

- Display-thread fresh-frame retrieval p95 0.016 ms (previously ~6–7 ms).
- Total display-thread get_frame cost 4.1 ms/s (previously ~630–645 ms/s).
- Display loop p95 10.106 ms, p99 14.277 ms.
- Tile uploads 59.0–59.4/s; zero superseded tiles in that window.

Earlier windows still contained isolated longer loop pauses and transient
decode queue waits. This is evidence for removing the sustained blocking
readback problem, not a guarantee of perfect pacing or zero latency. No
post-fix physical-presentation trace was collected.

Synthetic hardware validation decoded and consumed all 400 test tiles with
no conversion/surface/shutdown errors. The batching benchmark measured about
474 tiles/s (four tiles at 60 FPS require 240); this is synthetic, not a live
screen-content throughput guarantee. Regression tests cover bounded batches,
quiet partial batches, nonblocking consumers, error-flag preservation, and
shutdown waiting for an in-flight download.

Temporary per-frame timing collection and five-second profiling logs were
removed after validation. Ordinary connection and error logging remains.
