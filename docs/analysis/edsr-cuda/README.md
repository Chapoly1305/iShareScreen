# EDSR GPU pipeline: retained result, 2026-11-04

The cleaned, optional runtime preserves the accepted EDSR checkpoint and
mixed-precision TensorRT engine. On the lab RTX 4090, two post-cleanup runs of
1,000 frames each measured **10.352 / 10.370 ms P95** from submission of a
complete HEVC access unit through GPU decode, preprocessing, EDSR, GPU resizing,
GPU consumer handoff and input mapping release. Inputs arrived at a controlled
30 fps. Network, demuxing and actual display are outside this boundary.

This does **not** establish a stable sub-10ms pipeline, Mac performance, or a
completed desktop-renderer integration. It is separate from the canonical
[held-out CodeSCAN results](../../../results/README.md); those FP32 quality
scores and earlier AMP forward timings have not been replaced by these numbers.

## Retained changes

- Image planes remain on the GPU throughout the timed path. NVDEC mappings
  have explicit ownership and input-read completion events.
- Deterministic CUDA driver graphs include preprocessing and avoid the two
  PyTorch RNG-state fill kernels previously seen on each replay.
- The native NVDEC callback launches the prepared graph after mapping. The
  hot submission path avoids repeated Torch device/stream context setup and
  explicitly rejects the wrong CUDA context before native work.
- One inference may run at a time, with two output leases and a bounded graph
  cache. Delayed consumers, stale mappings, cancellation and partial-launch
  failures have explicit checks.

The default is `inline`: reservation remains inside the measured submission
call. Earlier paired results were 10.425 / 10.395 ms versus 10.530 / 10.593 ms
for the preceding callback path. Pre-reserving before simulated arrival did
not show a stable additional benefit, so it is optional and reports its cost
separately. Temporary GPU clock and CPU policy experiments were restored; the
retained measurements use the ordinary machine policies.

## Model and quality

The checkpoint is EDSR ×2, 16 residual blocks / 64 features, epoch 25:
`2c923e9ba9541cb85006684c0072868b2d71354d665cdd85cd2b0baac6faaabb`.
The accepted engine is:
`74f418fb7ad0cd03ae25d6ff701fead1414a6f180e334d13e9281263007dde74`.
It uses 30 INT8 convolutions, FP16 for the remaining final block/projection/tail,
and byte-coded residual corrections. Cleanup did not rebuild or replace it.

Post-cleanup validation passed all 72 frozen frames from 24 Mac validation
clips. Normalized input and final RGBA output are bit-exact against the accepted
mixed-precision engine with independent reference preprocessing/resizing.
This is not bit equivalence to FP32 and is not a new evaluation on the 1,200
held-out CodeSCAN test samples. The original engine quality gates and rejected
precision variants remain in the frozen local research records.

The validation also passed input reuse with deliberately delayed preprocessing,
held-output preservation, bounded slots, cache eviction/rebinding, failure
injection after successful native launch, wrong-context recovery, prepared
ticket cancellation and delayed consumer checks. Earlier whole-engine
TensorRT sanitizer diagnostics remain unresolved; these targeted checks do
not establish a clean whole-engine sanitizer run.

## Reproduction and evidence

Use the [build, validation and timing commands](../../../benchmarks/super_resolution/edsr_cuda/README.md).
The validated Linux stack was PyTorch 2.6.0+cu124, Triton 3.2, TensorRT 11.4.0.106,
PyNvVideoCodec 2.2.4 and driver 595.99.02. The GPU was
`GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a` on `Alienware`.

`SUMMARY.json` binds the cleaned source files and records the measurement
boundary. `evidence/quality.json` contains the 72-frame and lifecycle checks;
`evidence/latency-first.json` and `latency-repeat.json` retain the raw timing
samples. `evidence/RUN.json` records the executed commands and successful exits;
`nvdec.build.json` records compiler inputs and the native-library hash.

The repository test suite passed after cleanup: 672 passed, 57 skipped, with
GPU visibility disabled. The RTX 4090 checks above were run separately.

Large TensorRT plans, datasets, compiled libraries and Nsight captures stay
in the ignored local research directories. Original exploratory sources are
versioned in the [source archive](../../../research/edsr-source-archive-20261104/README.md).
