# Desktop video enhancement

Optional source-checkout tools for training and evaluating the viewer's shared
[enhancement adapter](../../docs/video-enhancement.md). These tools and their
dependencies are not included in the viewer wheel.

- [Training](TRAINING.md): frozen data, five content folds, epoch training and recovery.
- [Data curation](CURATION.md): paired extraction, similarity filtering and review.
- [Results](RESULTS.md): the completed 1,500-epoch comparison and local viewing material.
- [Inference performance](PERFORMANCE.md): M4 and RTX 4090 adapter latency.

## Environment

Run from the repository root with Python 3.10/3.11 and FFmpeg/ffprobe
(libx265 and libx264rgb):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m benchmarks.super_resolution.setup_models
.venv/bin/python -m pip check
```

Install the appropriate PyTorch CUDA build first on NVIDIA hosts.
`requirements-ml.txt` pins the ML environment; the macOS ARM64 and Linux CUDA
lock files retain the measured platform environments. The CUDA lock needs
`--extra-index-url https://download.pytorch.org/whl/cu124`.
FiftyOne curation uses a separate optional environment.

Cache the official LPIPS backbone before scoring or training; scoring never
silently downloads missing weights:

```sh
.venv/bin/python -c 'from torch.hub import load_state_dict_from_url; from torchvision.models import AlexNet_Weights; load_state_dict_from_url(AlexNet_Weights.IMAGENET1K_V1.url, check_hash=True, weights_only=True)'
```

## Prepare and evaluate recordings

Use fresh output directories. Preparation accepts 1920x1080 MOV recordings,
preserves complete frame counts and timestamps, and creates lossless decoded
RGB references plus 960x540 inter-frame HEVC 4:4:4 CRF 28 inputs. Reduction uses
Lanczos with full-range BT.709. References preserve decoded recording pixels;
they do not undo compression in the original recordings.

```sh
python -m benchmarks.super_resolution.prepare_video \
  --dataset /path/to/recordings --output /path/to/pairs

python -m benchmarks.super_resolution.video_benchmark \
  --manifest /path/to/pairs/manifest.json --output /path/to/reconstructions \
  --models bicubic edsr imdn safmn --device cuda --precision fp32

python -m benchmarks.super_resolution.video_report \
  --manifest /path/to/pairs/manifest.json --run /path/to/reconstructions \
  --output /path/to/VIDEO_RESULTS.md

python -m benchmarks.super_resolution.video_perceptual \
  --manifest /path/to/pairs/manifest.json --run /path/to/reconstructions \
  --output /path/to/perceptual --device cuda
```

Every reconstructed frame is saved losslessly and re-decoded to check its RGB
sequence hash. Pixel metrics sample every 60 frames plus the final frame.
LPIPS verifies the same frame identities and producer pixel scores before scoring.

Metrics are RGB PSNR (higher), SSIM (higher) and LPIPS (lower). SSIM uses
scikit-image's 7x7 uniform window and sample covariance. LPIPS uses AlexNet v0.1,
FP32, native 1080p RGB in [-1,1], with no resizing or tiling.
Means average frames within recordings, then recordings equally. Retain paired
per-frame scores and compare against bicubic. No OCR metric is used.

## Comparison media

`render_trained prepare` extracts aligned scenes from the paired recordings.
`render_trained infer` reconstructs them using each recording's held-out fold
and exact CURRENT checkpoint at the requested epoch. Both steps check input,
checkpoint and fold identities. Use `--help` for scene-spec and path arguments.

```sh
python -m benchmarks.super_resolution.compose_trained \
  --prepared /path/to/excerpts/manifest.json --predictions /path/to/predictions \
  --pretrained /path/to/completed-pretrained-run \
  --output /path/to/comparison --epoch 1500
```

The eight panels align Original / pretrained EDSR, IMDN, SAFMN above
Bicubic / fine-tuned EDSR, IMDN, SAFMN. Omitting `--pretrained` produces the
six-panel input/bicubic/fine-tuned comparison. Pretrained outputs must be verified
lossless reconstructions from the same source manifest; they are aligned by
source-frame index, not approximate video seeks.

The compositor writes a viewing video, lossless detail PNGs and an HTML gallery.
Image panels have separators; descriptive titles appear only in HTML.
`compose_trained --output /path/to/comparison --refresh-gallery` restyles an
existing gallery without model inference or video encoding.

`render_candidate_media compose-matrix` creates a unified comparison from native
×2 candidate reconstructions. Model columns stay fixed: Original, the unenhanced
540p input and the four pretrained models on top; Bicubic, the same input and
their fine-tuned counterparts below. Input panels use nearest-neighbor display
scaling. Original-size 960×540 PNGs and lossless input videos are retained as artifacts,
without a separate input-image section in the gallery.
Videos and detail PNGs retain per-panel model/variant headers and separators.
The HTML page provides buttons for each overview and detail chapter.

For screenshot-only scene extensions, use `review_recordings` to inspect source
recordings and save a selected-frame plan, then `candidate_stills render` to
reconstruct those frames with each capture's held-out fold. `candidate_stills
compose` appends enlarged, labeled detail crops to an existing
gallery. It preserves the existing images and presents scene navigation without
rendering or embedding a new video. Original-size 540p inputs remain available.
Viewing videos are lossy and are never metric inputs.

For live comparisons, `export_live` installs inference-only bundles from audited
completed trajectories without modifying training checkpoints or official
weights. See [live comparison setup](../../docs/video-enhancement.md#live-comparison)
for the fixed 540p connection profile and pretrained/fine-tuned switching.

## Performance profiling

Profile warmed inference at the intended input dimensions before training a new
deployment candidate. The live dashboard exposes assembly, preparation, input,
model, output and display-upload enqueue times after an enhanced frame completes.
They are processing stages, not input-to-display latency. The reference adapter
uses explicit device barriers to make its per-stage measurements meaningful.

```sh
python -m benchmarks.super_resolution.profile_live \
  --images /path/to/540p-frame.png --models edsr imdn safmn \
  --device mps --leaf-hotspots --output /path/to/fresh-profile.json
```

Stop competing GPU inference while measuring. The report separates model load,
warm-up, raw stage samples, median and p95 latency. Optional leaf-module timings
insert barriers and omit functional operations; use them to locate hotspots,
not as additive production-latency estimates. A 30-FPS deployment needs roughly
33 ms for the entire post-decode enhancement path, not just the model kernel.
Reject an optimized backend if paired output quality changes unacceptably, even
if its timing improves. Lower parameter count alone does not establish latency.

### Lightweight candidate comparison

Per-model runtime directories hold pinned asset metadata and deployment loaders;
`setup_candidates` downloads and verifies assets without executing them.
`convert_candidate` installs a Core ML conversion only after agreement checks
against CPU FP32 on at least three distinct full-size inputs. See the
[candidate installation guide](../../docs/video-enhancement.md#lightweight-model-candidates)
for supported scales, precisions and licensing boundaries.

```sh
python -m benchmarks.super_resolution.setup_candidates
python -m benchmarks.super_resolution.benchmark_candidates \
  --prepared /path/to/prepared --output /path/to/fresh-comparison \
  --device mps --warmup 5 --iterations 40 --lpips
```

The prepared directory contains `code/`, `terminal/` and `desktop/`, each with
`low.png` (960×540 RGB) and `reference.png` (1920×1080 RGB). The probe saves
reconstructed PNGs, input/reference hashes, source bindings, weight/backend
identity, raw timing samples, p50/p95 and paired PSNR/SSIM/LPIPS. Bicubic is
included as a baseline. Torch resident-forward timing is separate from complete
adapter timing. Core ML prediction includes framework-managed I/O; it is not a
resident GPU-only measurement. Native ×4 outputs are timed separately without
paired quality scores because the input protocol has no 2160p ground truth.
The JSON records unsupported and failed configurations explicitly; suite
completion does not mean every backend passed. Use a quiet GPU for comparison.

For a video-based latency comparison, use the same decoded recording frames
for every model and precision:

```sh
python -m benchmarks.super_resolution.video_latency \
  --video /path/to/960x540-recording.mkv --output /path/to/fresh-latency \
  --device cuda --samples 120 --warmup 20
# On Apple Silicon, include the installed fixed-resolution Core ML models:
python -m benchmarks.super_resolution.video_latency \
  --video /path/to/960x540-recording.mkv --output /path/to/fresh-mac-latency \
  --device mps --backends torch coreml --samples 120 --warmup 20
```

The probe samples frames uniformly, separately measures decode/RGB conversion,
then runs serial batch-one resident forwards with CUDA events (CUDA) or
synchronized host wall time (MPS), and the actual
complete adapters with synchronized stage timing. FP32 disables TF32; the second
CUDA mode uses FP16 autocast with FP32 weights, not pure FP16. MPS uses FP32;
Core ML uses each validated conversion's precision, with prediction API timing
instead of resident-forward timing. cuDNN autotuning is
enabled and warm-up calls are excluded. Output checks remain in the adapter
measurement but outside resident event intervals. AMP differences on the first
sample are recorded against CUDA FP32; this is not full quality validation.
The JSON includes raw samples, frame identities, source/weight bindings and GPU
identity. It does not claim TensorRT, live FPS or end-to-end display latency.

For CPU-prohibited model inference, start with
`ISS_SR_GPU_ONLY=1 PYTORCH_ENABLE_MPS_FALLBACK=0` and specify only learned Torch
models (omit `bicubic` and `--backends coreml`). The Live evaluation profile sets
these flags automatically. Core ML permits CPU in every public compute-unit
configuration, so it is not eligible for this strict mode. `inspect_execution`
audits model-forward tensor devices and records Core ML anticipated placement:

```sh
ISS_SR_GPU_ONLY=1 PYTORCH_ENABLE_MPS_FALLBACK=0 \
  python -m benchmarks.super_resolution.inspect_execution \
  --image /path/to/540p-frame.png --output /path/to/fresh-execution-audit.json
```

Core ML compute plans are not runtime hardware traces. The retained Instruments
recording in the local performance artifacts separately demonstrates Neural
Engine prediction activity for QuickSRNet. CPU preprocessing and output
conversion are outside the GPU-only model-forward requirement.

## Code layout and checks

`train_epochs.py` is the only training runner; `epochs.py` handles sampling,
`train.py` contains shared data/update/validation functions, and
`training_state.py` handles atomic checkpoints. `epoch_report.py` audits and
aggregates scores; `epoch_plots.py` renders curves. `gallery.py` owns image/HTML
layout separately from video composition. `setup_models.py` and
`models.lock.json` pin official model assets.

```sh
.venv/bin/python -m pytest
.venv/bin/python -m pip wheel --no-deps . --wheel-dir /path/to/build-output
```

Datasets, checkpoints and generated media remain local and ignored by Git.
Offline timing is not live viewer latency or sustained FPS; live integration
must be checked separately. Continuous comparison videos help inspect flicker,
which frame-average image metrics do not measure.
