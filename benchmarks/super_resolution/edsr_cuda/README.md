# Optional CUDA EDSR runtime checks

This is the maintained entry point for the fixed-format RTX 4090 experiment.
The runtime uses no dated research directories. It is an optional component,
not a registered desktop renderer backend. See the
[measurement report](../../../docs/analysis/edsr-cuda/README.md) for results,
engine identity and limitations.

## Requirements

Use the project environment (`source dev-environment.sh`). The validated stack
was PyTorch 2.6.0+cu124, Triton 3.2, TensorRT 11.4.0.106, PyNvVideoCodec 2.2.4
and NVIDIA driver 595.99.02 on Linux / RTX 4090. TensorRT and PyNvVideoCodec
are optional and are not installed by the ordinary viewer. The runtime needs
Torch, Triton and TensorRT; these validation commands additionally use
PyNvVideoCodec for demuxing and OpenCV for independent reference resizing.

Building the native shim needs a C++17 compiler, CUDA headers, `libcuda.so.1`
and `libnvcuvid.so.1`. Two unmodified NVIDIA SDK headers and their license
notices are included under `src/isharescreen/enhancement/native/`.

Provide the accepted engine and frozen validation directory yourself. Model
weights, TensorRT plans, decoded images and Nsight captures are not in Git.
The benchmark requires an explicitly selected, otherwise idle single-GPU host.
Follow `AGENTS.md`; do not run it on occupied H100 devices.

## Build and validate

From the project root, choose a fresh output directory:

```bash
python -m benchmarks.super_resolution.edsr_cuda.build_native \
  --cuda-include /usr/local/cuda-13.4/include \
  --output benchmarks/super_resolution/artifacts/edsr-cuda/nvdec.so

python -m benchmarks.super_resolution.edsr_cuda.validate \
  --engine /path/to/edsr.engine \
  --engine-sha256 74f418fb7ad0cd03ae25d6ff701fead1414a6f180e334d13e9281263007dde74 \
  --validation /path/to/validation-baselines-2026-10-29 \
  --native-library benchmarks/super_resolution/artifacts/edsr-cuda/nvdec.so \
  --expected-gpu-uuid GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a \
  --output benchmarks/super_resolution/artifacts/edsr-cuda/quality.json
```

The validator checks 72 frozen Mac frames against an independent preprocessing
and resize reference using the accepted engine. It also checks mapped-input
reuse, held outputs, cache eviction, errors after native launch, cancellation,
wrong contexts, EOS and delayed consumers. Reference pixel downloads happen
only in correctness checks, outside timing.

Then measure the preferred `inline` submission path:

```bash
python -m benchmarks.super_resolution.edsr_cuda.benchmark \
  --engine /path/to/edsr.engine \
  --engine-sha256 74f418fb7ad0cd03ae25d6ff701fead1414a6f180e334d13e9281263007dde74 \
  --validation /path/to/validation-baselines-2026-10-29 \
  --native-library benchmarks/super_resolution/artifacts/edsr-cuda/nvdec.so \
  --expected-gpu-uuid GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a \
  --mode inline --fps 30 --samples 1000 \
  --output benchmarks/super_resolution/artifacts/edsr-cuda/latency.json
```

The current standalone timing input is the first frozen clip, repeated as
needed, with 50 warmup frames per decoder pass. Each result records actual
arrival spacing, raw samples and source hashes. `--mode prearmed` additionally
reports preparation cost and adds it back into a separate total; it did not
show a stable advantage over inline submission. `--capture` brackets the
measured frames with CUDA profiler start/stop for external Nsight collection.

## API and ownership

`NativeCudaSurfaceEDSR` is in `isharescreen.enhancement.cuda_native`;
`NvdecDecoder` is in `isharescreen.enhancement.cuda_decoder`.

Construct the runtime with a verified engine SHA256 and
`input_format="nv12_bt709_full"`. Create the decoder on `runtime.stream`,
passing `library_path`, or set `ISS_NVDEC_LIBRARY`. The decoder requires that
same current CUDA context. Feed contiguous, complete HEVC access units from
a progressive, no-B-frame 1280×720, 8-bit full-range BT.709 4:2:0 stream.
Network fragments, format changes, HDR and other color ranges are unsupported.

`runtime.decode_submit(decoder, packet)` returns `(mapped_frame, lease)`, or
`None` for no output. Borrow with `lease.wait_on(consumer_stream)`, enqueue
consumer reads, then `lease.release()`; discard unused output with
`lease.discard()`. Release the decoder mapping only after an event proves all
input reads have completed. Keeping a Python frame object alone does not
protect a decoder pool from reuse.

There is at most one inference in flight and two consumer-owned output slots.
`SurfaceBusy` is backpressure: keep at most the latest pending frame, rather
than accumulating a latency-producing queue. A prepared ticket reserves one
slot and must be submitted once or cancelled. Close all leases/tickets and
release the input mapping before closing the runtime and decoder.

## Historical investigation

Original profiling and engine-builder sources are retained byte for byte in
the [source archive](../../../research/edsr-source-archive-20261104/README.md).
They include failed candidates and use historical assumptions. They are not
dependencies of these commands.
