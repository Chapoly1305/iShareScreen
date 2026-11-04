# Historical model evidence extracts

These JSON files are compact extracts, not fresh GPU runs. Each has `source`,
`source_sha256`, `extraction`, and `data`. Source hashes identify the original
local files, not the extracted JSON. Historical status/goal fields are retained
as snapshots and must not be interpreted as current project status.

| File | Question supported |
| --- | --- |
| [optimization-history.json](optimization-history.json) | Initial speed/quality progression, including rejected ordinary INT8 |
| [baseline-forward.json](baseline-forward.json) | Same-checkpoint PyTorch AMP baseline |
| [tensorrt-fp16.json](tensorrt-fp16.json) | Historical TensorRT FP16 level-5 baseline |
| [model-first.json](model-first.json), [model-repeat.json](model-repeat.json) | Accepted engine eager/graph timing and repetition |
| [packed-state-summary.json](packed-state-summary.json) | Accepted engine identity, coding, quality, memory and sanitizer limits |
| [rejected-terminal-int8.json](rejected-terminal-int8.json) | Why the faster 32-convolution INT8 candidate was rejected |
| [calibration-selection.json](calibration-selection.json) | Train-only selection policy and source manifest hash |
| [balanced-calibration.json](balanced-calibration.json) | Channel balancing, calibration count, split and validation decisions |
| [balanced-build.json](balanced-build.json) | Names of 30 quantized convolutions and calibration hash binding |
| [legacy-nvdec-stages.json](legacy-nvdec-stages.json) | Earlier 25-frame hardware profile; not a breakdown of the final runtime |

Large arrays (`samples_ms`, per-frame `records`, calibration `maxima`, telemetry,
per-frame hardware events) are omitted. Timing extracts add `sample_count`
computed from the original raw timing arrays. The optimization history retains
its summary `rows`; calibration selection omits individual sample rows.
The extracts permit inspection of reported summaries without the ignored
research archive; recomputing model percentiles and per-frame quality requires
the original hash-bound files. Full final-runtime samples remain in the parent
directory's `latency-first.json` and `latency-repeat.json`.

See the [reviewer Q&A](../../REVIEW.zh-CN.md) for interpretation and limitations.
