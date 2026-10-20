# Desktop video enhancement

Optional source-checkout tools for training and evaluating the viewer's shared
[enhancement adapter](../../docs/video-enhancement.md). These tools and their
dependencies are not included in the viewer wheel.

- [Training](TRAINING.md): frozen data, five content folds, epoch training and recovery.
- [Data curation](CURATION.md): paired extraction, similarity filtering and review.
- [Results](RESULTS.md): the completed 1,500-epoch comparison and local viewing material.

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
Viewing videos are lossy and are never metric inputs.

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
