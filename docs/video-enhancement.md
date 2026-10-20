# Video enhancement

The native desktop viewer supports optional ×2 upscaling after decoding. The
default is **Original (native scaling)**. After connecting, open Diagnostics →
Video enhancement → Display model to select a baseline, an official pretrained
model, an installed fine-tuned model, or return to Original. Browser playback is
unchanged.

## Installation

For a source checkout, use Python 3.10 or newer:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[enhancement]'
.venv/bin/python -m benchmarks.super_resolution.setup_models
```

On NVIDIA hosts, install the appropriate PyTorch CUDA build for the host before
installing the enhancement dependencies. The pinned benchmark environment is
separate; see [the evaluation tools](../benchmarks/super_resolution/README.md).

Setup downloads pinned official source checkouts and pretrained checkpoints. It
checks source revisions, rejects tracked vendor edits, and verifies checkpoint
SHA256 hashes. Inference requires exact tensor keys/shapes. There is no automatic
download during playback and no random-weight fallback. Missing models remain
unavailable in the dropdown. EDSR-baseline, IMDN and SAFMN have pinned ×2 weights;
ECBSR remains unavailable until an eligible original checkpoint is supplied.

Model data defaults to `~/.cache/isharescreen/super_resolution/`, independently of
the installation directory. Set `ISS_SR_HOME` before both setup and playback to
use another location. Setup also accepts `--home`, but playback must use the
matching `ISS_SR_HOME`. Sources are stored under `vendor/` and weights under
`checkpoints/`; neither is bundled in the viewer wheel. Refer to each provider's
terms before redistributing model assets.

`ISS_SR_DEVICE=auto` chooses CUDA, then MPS, then CPU. Set it to `cuda`, `mps` or
`cpu` to choose explicitly. CPU inference may be too slow for interactive use.

### QuickSRNet on Core ML

On macOS 13 or newer, the live dropdown also supports **QuickSRNet-Small -
Pretrained ×2 (Core ML FP16, 540p)**. This is the official pretrained Small x2
model, not a fine-tuned variant. It uses a validated fixed-resolution conversion:
960×540 RGB input, 1920×1080 output, FP32 tensor I/O and FP16 internal computation.
Core ML may use the CPU, GPU and Neural Engine; actual operation placement is
chosen by the framework. This option is independent of PyTorch inference.

Install the optional runtime and the converted package retained by the pretrained
latency probe (including the adjacent provider `LICENSE.md`):

```sh
.venv/bin/python -m pip install -e '.[enhancement-coreml]'
.venv/bin/python -m benchmarks.super_resolution.install_coreml \
  --package /path/to/quicksrnet-small-x2-latency-v1/model.mlpackage
```

The installer verifies the exact validated package's graph, weights and manifest,
then copies it atomically into the model asset home. It does not overwrite a
different installation. Arbitrary conversions are rejected; a new conversion
must be validated and pinned before use. Model assets stay outside the wheel.

Core ML is available only outside GPU-only mode: it cannot prohibit CPU
execution. The standard **Super-resolution evaluation** profile now disables
Core ML choices; use **QuickSRNet-Small (PyTorch)** there instead. A non-strict
Core ML session still requires a fixed 960×540 canvas. Leave
`ISS_SR_DEVICE` unset or set to `auto`. Other input sizes produce an explicit error
and restore Original; the adapter never resizes the source silently. Use the same
dropdown to switch back to Bicubic or the existing pretrained/fine-tuned models.
The timing and weight-identity rows reflect the selected backend.
For Core ML, the model stage is the blocking prediction API call, including
framework-managed transfers; the output stage is CPU RGB conversion.
Output conversion reuses planar scratch buffers while preserving FP32 scaling,
round-to-even and clipping. Each returned RGB frame owns its pixels, so the next
prediction cannot overwrite a frame waiting for display. Enhanced texture upload
uses vectorized RGB-to-RGBA packing when OpenCV is available; ordinary playback
retains its dependency-free path. Restart the viewer process to pick up code
changes; changing the display-model selection does not reload Python modules.

### Lightweight model candidates

Each candidate has a dedicated directory under
`src/isharescreen/enhancement/candidates/`, containing `model.json` (official
revision, download URLs, SHA256 hashes and license) and `loader.py` (architecture
and deployment transformation). Shared backend code handles timing, input/output
validation and live-worker integration. The existing training adapter and its
source-bound checkpoints are unchanged.

| Model | Official scale | Live backends | Validated Core ML precision |
| --- | --- | --- | --- |
| PlainUSR-U | ×2 | PyTorch, Core ML | FP32 |
| QuickSRNet-Small | ×2 | PyTorch, Core ML | FP16 |
| RT4KSR-S | ×2 | PyTorch, Core ML | FP32 |
| EfRLFN | ×2 | PyTorch, Core ML | FP16 |
| CASR | ×4 | Offline PyTorch only | — |
| SPANV2-ESR C2 | ×4 | Offline PyTorch only | — |

Install the pinned source files and weights explicitly:

```sh
.venv/bin/python -m benchmarks.super_resolution.setup_candidates
```

Assets are installed under `ISS_SR_HOME/candidates/<model>/`, not in the viewer
wheel or repository. CASR is excluded from the default download because its
upstream repository has no identified license. Private evaluation requires
`--models casr --allow-unlicensed`; do not redistribute its assets without
appropriate permission. SPANV2 uses the official reference implementation, not
the custom CUDA-fused implementation.

For PlainUSR-U, RT4KSR-S and EfRLFN, create a local Core ML bundle using three or
more distinct RGB 960×540 images:

```sh
.venv/bin/python -m benchmarks.super_resolution.convert_candidate \
  --model plainusr --precision fp32 \
  --images /path/to/code.png /path/to/terminal.png /path/to/desktop.png
```

Repeat with `--model rt4ksr --precision fp32` and
`--model efrlfn --precision fp16`. Conversion requires both optional enhancement
extras. QuickSRNet uses the separate pinned-package installer above. Every
conversion must agree with the official CPU FP32 model within one RGB level on
all supplied images before installation. These images are conversion checks,
not a guarantee for every possible input. PlainUSR-U and RT4KSR-S FP16
conversions failed this check on the current Mac, so only their validated FP32
versions are installed. Existing bundles are never silently overwritten.

Restart the launcher, enable **Super-resolution evaluation**, and select an
installed PyTorch candidate in the same Display model dropdown. This profile
requires GPU-only execution and excludes Core ML. Outside the strict profile,
leave `ISS_SR_DEVICE` unset or `auto` when using Core ML: explicit `mps` applies
only to PyTorch. Core ML versions require fixed 540p input; PyTorch candidates
use FP32 on MPS. Native ×4 weights are not relabeled or resized to pretend they
are ×2 models, and remain unavailable in Live.

## Live comparison

Start `.venv/bin/iss` and enable **Super-resolution evaluation** in the connection
form. Enter the host Mac's address and account credentials. The profile locks:

- Native desktop frontend and a single combined display window.
- 960×540 encoded input, 100% host UI scale, and dynamic resolution disabled.
- HEVC with the client's best available HEVC decoder and audio off.
- PyTorch FP32 on MPS/CUDA only; CPU fallback and Core ML are disabled.
- A private virtual display (Curtain), which blanks the host's physical screen.

The launcher enforces these settings even if disabled form controls are omitted
or modified. Reconnect retains the profile. The diagnostics banner checks the
initial negotiated canvas and warns if the host did not provide 960×540. Verify
that canvas before comparing. Resizing the viewer scales the displayed image;
it does not request a new host resolution in this mode. For visual comparisons,
keep the viewer window size unchanged across models; ×2 reconstruction produces
1920×1080 pixels, but the window may scale those pixels to fit its actual surface.

In Diagnostics → Video enhancement → Display model, switch between Original
and installed pretrained/fine-tuned GPU models. CPU interpolation baselines
and Core ML choices remain disabled in this profile. The fine-tuned label includes epoch, fold and current/best
checkpoint role. Diagnostics also shows the processing device, precision and
weight hash; hover over that identity for the installed weight path. Switching
models reuses the connection and input stream. Model-load/weight errors restore
the original rendering path, never silently substitute pretrained weights.
The stage timing row separates canvas assembly, input preparation/transfer,
model execution, output conversion/transfer, and display-upload enqueue time.
Display enqueue time is a CPU submission measurement, not GPU completion or
input-to-display latency. Restart an already-running viewer after code updates
to use new diagnostics.

This profile requests a low-resolution virtual desktop directly; it is not a
downsampled copy of a simultaneously captured 1080p reference. Live interaction
and visual checks therefore complement the paired offline evaluation rather
than produce directly comparable PSNR/SSIM/LPIPS scores. Host encoder settings
remain negotiated; this is not a lock on bitrate, CRF or source FPS.

### Install fine-tuned weights

Use a completed epoch campaign and its matching `epoch_report` audit:

```sh
.venv/bin/python -m benchmarks.super_resolution.export_live \
  --runs /path/to/completed-epochs \
  --audit /path/to/report/audit.json \
  --fold 0 --role current
```

The default exports all three architectures from fold 0 at the completed epoch
ceiling. This is an explicit comparison fold, not an ensemble or an all-data
deployment model. Use `--fold N` to install other folds, `--role best` for each
model's selected checkpoint, or `--models imdn` for one architecture.
Current and best variants remain separate and display their actual epochs.

Bundles live under `ISS_SR_HOME/fine-tuned/` (or the default model asset home).
Each contains `manifest.json` and inference-only `weights.pt`, with no optimizer,
RNG state, recordings or dataset. Export verifies audited checkpoint identity,
official source/initializer identity, finite tensors and strict architecture
compatibility. Existing training checkpoints and official weights are not
modified. Copy complete bundles to the same asset-home layout on another viewer
after installing its official assets. Restart the viewer to discover new bundles.

## Runtime boundaries

- `src/isharescreen/enhancement/models.py` owns the shared inference adapter.
- `variants.py` discovers live choices and loads hash-verified fine-tuned states
  through that same adapter without changing training source bindings.
- `candidates/<model>/` owns each lightweight model's metadata and loader.
- `coreml.py` validates fixed-540p conversions and provides the optional Core ML
  adapter through the same live worker contract.
- `live.py` assembles decoded tiles and runs inference outside the control/render
  thread, with at most one pending canvas and generation-bound results.
- The native renderer uploads RGB reconstructions while keeping cursor composition
  separate. Switching models, resizing or returning to Original invalidates stale
  results. A model/upload failure restores the original rendering path.
- Offline tools import the same adapter; they do not duplicate model definitions.

Enhancement selection is opt-in and does not change transport negotiation or
source stream resolution; the separate evaluation connection profile pins the
initial request. Original mode needs neither PyTorch nor OpenCV. Offline inference
timings are not input-to-display latency or sustained viewer FPS.

### GPU-only execution

The evaluation launcher sets `ISS_SR_GPU_ONLY=1` and
`PYTORCH_ENABLE_MPS_FALLBACK=0` in the viewer child **before importing PyTorch**.
`execution.py` rejects CPU devices, unavailable GPUs, enabled MPS fallback,
CPU baselines and Core ML choices, and verifies model parameters/buffers are on
the selected GPU. Unsupported MPS operations fail instead of silently running
on CPU. Failure restores Original, not CPU inference.

For other launcher/CLI sessions, enable the same policy explicitly:

```sh
ISS_SR_GPU_ONLY=1 PYTORCH_ENABLE_MPS_FALLBACK=0 .venv/bin/iss
```

The policy concerns model inference. Decoding, image preparation, RGB output
conversion and application control can still use CPU. Restart an existing GUI
launcher after updating code; an already-imported launcher does not acquire new
child-environment settings merely by reconnecting.

## Maintenance and verification

Keep runtime behavior in `src/`, reproducible evaluation tools in `benchmarks/`,
and user-facing documentation here. Do not add datasets, checkpoints, machine
paths or campaign-specific schedulers to the runtime package.

```sh
.venv/bin/python -m pip install -e '.[test]'
.venv/bin/python -m pytest
ISS_TEST_GPU=1 .venv/bin/python -m pytest tests/test_enhancement_gpu.py
# Include installed Core ML candidates in real Core ML -> Metal readback:
ISS_TEST_GPU=1 ISS_TEST_COREML=1 .venv/bin/python -m pytest tests/test_enhancement_gpu.py
ISS_SR_CANDIDATE_SMOKE=1 .venv/bin/python -m pytest tests/test_enhancement_candidates.py
.venv/bin/python -m pip wheel --no-deps .
```

Tests cover optional-dependency bypass, bounded work queues, stale-result rejection,
control dispatch and geometry. GPU readback tests are opt-in. Benchmark tests skip
when their optional dependencies are absent; CUDA parity tests need a CUDA device.
Recorded-video checks do not establish live Screen Sharing or physical-device
performance.
