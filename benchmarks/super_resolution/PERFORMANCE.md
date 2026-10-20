# Super-Resolution Inference Performance

Measured on October 20, 2026, using the same 960×540 desktop recording on
Apple M4 and NVIDIA RTX 4090. All values are adapter latency in milliseconds,
**p50 / p95**, sorted by RTX 4090 p95.

| Model | Scale | Fine-tuned | M4 adapter | RTX 4090 adapter |
| --- | --- | :---: | ---: | ---: |
| QuickSRNet-Small | ×2 | | 19.92 / 21.14 | 2.40 / 2.43 |
| PlainUSR-U | ×2 | | 19.66 / 23.42 | 2.62 / 2.65 |
| RT4KSR-S | ×2 | | 24.24 / 26.25 | 2.62 / 2.66 |
| CASR | ×4 | | 26.72 / 28.13 | 4.42 / 4.48 |
| SPANV2-ESR C2 | ×4 | | 109.87 / 115.38 | 8.34 / 8.39 |
| EfRLFN | ×2 | | 262.28 / 273.48 | 21.09 / 21.12 |
| IMDN | ×2 | ✓ | 385.42 / 409.91 | 28.43 / 28.49 |
| EDSR-baseline | ×2 | ✓ | 509.26 / 541.04 | 35.87 / 35.91 |
| SAFMN | ×2 | ✓ | 445.74 / 453.77 | 45.64 / 45.67 |

**Fine-tuned** indicates that a project fine-tuned variant is available, not
that it was used in this latency measurement. Every row above was measured
with official pretrained weights. The current training runner supports EDSR,
IMDN and SAFMN; the six new candidates are inference integrations only.

Precision: M4 uses PyTorch MPS FP32 with CPU fallback disabled. RTX 4090 uses
CUDA AMP FP16 with FP32 weights, except RT4KSR, which uses FP32 because its
adapter was faster. TF32 is disabled. SPANV2 uses the reference implementation,
not the custom CUDA-fused path.

Native ×4 models output 3840×2160; ×2 models output 1920×1080. Their measured
times are sorted together for convenience, not as equivalent-resolution tasks.
CASR assets have no identified upstream license and remain private evaluation
assets, outside the repository and distribution.

## Measurement protocol

The six-second, 60 FPS clip contains 360 decoded frames. Each device processes
the same 120 uniformly selected frames, including the first and last; video
hashes, sampled indices and sampled RGB hashes match. The clip is a lossless
H.264 storage copy of prepared low-resolution frames, not an encoder benchmark.

Each configuration runs serially at batch size 1 with 20 excluded warm-up calls
and 120 measured adapter calls. PyTorch and OpenCV each use one CPU thread;
cuDNN autotuning is enabled on CUDA. Adapter latency includes RGB preparation,
accelerator transfer, model execution, output validation and uint8 conversion.
It excludes decoding, network transport, canvas assembly and display upload.
It is not sustained live FPS or input-to-display latency.

The four lightweight M4 ×2 candidates were retested with explicit
`ISS_SR_GPU_ONLY=1 PYTORCH_ENABLE_MPS_FALLBACK=0`. Other M4 rows retain the
original MPS measurements, where fallback was not enabled. No TensorRT,
compiled Torch graph or CUDA Graphs are used. AMP differences from FP32 were
checked on the first sampled frame, not through full quality re-evaluation.

Hardware: Apple M4 with 16 GiB unified memory on macOS 15.7.9, and RTX 4090
with 24 GiB VRAM. Both use PyTorch 2.6.0; the NVIDIA build uses CUDA 12.4.
The previous live viewer session was closed during measurement. The NVIDIA
run used process-local libraries matching its loaded 595.91.07 driver, without
changing the system driver.

## Execution verification

All nine full-size Torch model forwards passed an MPS tensor-device audit with
CPU fallback explicitly disabled; no CPU tensor operations were observed inside
those audited forwards. CPU image conversion and application control remain
outside the model-inference restriction.

A separate Instruments recording captured 120 Neural Engine prediction events
for QuickSRNet Core ML. Core ML plans prefer Neural Engine for QuickSRNet/EfRLFN
and GPU for PlainUSR/RT4KSR. Plans are not runtime traces, and Core ML provides
no public setting excluding CPU. Consequently the GPU-only comparison above
uses Torch, and the strict Live evaluation profile disables Core ML.

## Reproduction

After installing the pinned official assets, run from the checkout root:

```sh
ISS_SR_GPU_ONLY=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python -m benchmarks.super_resolution.video_latency \
  --video /path/to/960x540-recording.mkv --output /path/to/fresh-results \
  --device cuda --samples 120 --warmup 20 \
  --models quicksrnet plainusr rt4ksr casr spanv2 efrlfn imdn edsr safmn
```

Use `--device mps` on Apple Silicon. CUDA defaults to FP32 and AMP measurements;
MPS uses FP32. [Setup and profiling](README.md#performance-profiling) describes
asset installation and execution audits; [Live setup](../../docs/video-enhancement.md)
describes the fixed-resolution GPU-only profile.

Local raw observations and execution evidence remain under
`artifacts/cross-device-video-latency/`, ignored by Git. They retain per-call
samples, input identities, source/weight hashes, device/precision metadata and
the Instruments recording. This document is the maintained result table.
