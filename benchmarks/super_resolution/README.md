# Desktop-video super-resolution evaluation

These optional source-checkout tools compare paired reference recordings with
low-resolution HEVC streams reconstructed by the production enhancement adapter.
They are not included in the viewer wheel. See [runtime setup and architecture](../../docs/video-enhancement.md).

## Environment

From the repository root, using Python 3.10/3.11:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m benchmarks.super_resolution.setup_models
.venv/bin/python -m pip check
```

`requirements-ml.txt` pins the evaluation environment. Install the appropriate
PyTorch CUDA build first on NVIDIA hosts. FFmpeg/ffprobe with libx264rgb and libx265
is required for video preparation and viewing copies. Select a free GPU explicitly;
these tools do not schedule remote jobs or manage other users' processes.

## Complete video workflow

All commands run from the repository root. Use fresh output directories; failed
or partial runs must not be silently mixed with later runs.

```sh
.venv/bin/python -m benchmarks.super_resolution.prepare_video \
  --dataset /path/to/recordings --output benchmarks/super_resolution/artifacts/pairs

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m benchmarks.super_resolution.video_benchmark \
  --manifest benchmarks/super_resolution/artifacts/pairs/manifest.json \
  --output benchmarks/super_resolution/artifacts/reconstructions --precision fp32

.venv/bin/python -m benchmarks.super_resolution.video_report \
  --manifest benchmarks/super_resolution/artifacts/pairs/manifest.json \
  --run benchmarks/super_resolution/artifacts/reconstructions \
  --output benchmarks/super_resolution/artifacts/VIDEO_RESULTS.md \
  --comparisons benchmarks/super_resolution/artifacts/comparisons
```

Preparation accepts 1920×1080 MOV recordings and preserves complete frame counts
and frame rates. It exports a lossless RGB8 decoded reference and 960×540 HEVC
4:4:4 CRF 28 input (Lanczos reduction, full-range BT.709, inter-frame prediction).
Other source geometries are rejected. `--captures` selects recording stems;
`--review` optionally supplies a dataset-specific JSON file with `excluded`
(stem-to-reason map), `content` (stem-to-label map) and `session_group` fields.
No recordings or dataset-specific labels are bundled.

Every frame is reconstructed and saved losslessly. Output videos are decoded
again to verify the complete RGB sequence hash. PSNR and SSIM use paired RGB
samples every 60 frames plus the final frame. `--reuse-identical` can reuse exact
consecutive LR inputs for offline quality evaluation; counts remain explicit and
must not be reported as interactive inference throughput.

### LPIPS

The metric is LPIPS/AlexNet v0.1, FP32, native 1080p RGB normalized to [-1,1], with
no metric resizing or tiling. Prepare the official backbone cache explicitly:

```sh
.venv/bin/python -c 'from torch.hub import load_state_dict_from_url; from torchvision.models import AlexNet_Weights; load_state_dict_from_url(AlexNet_Weights.IMAGENET1K_V1.url, check_hash=True, weights_only=True)'
.venv/bin/python -m benchmarks.super_resolution.video_perceptual \
  --run benchmarks/super_resolution/artifacts/reconstructions \
  --manifest benchmarks/super_resolution/artifacts/pairs/manifest.json \
  --output benchmarks/super_resolution/artifacts/perceptual --device cuda
```

Backbone/calibration hashes and loaded network-state identity are recorded and
checked; scoring refuses missing cached weights instead of downloading them.
The companion re-decodes complete videos, validates frame/timestamp alignment,
and checks sampled PSNR/SSIM against producer values before calculating LPIPS.
CUDA hosts may add `--pixel-metrics cuda` for FP64 PSNR/SSIM verification and
`--reference-cache-mib 8192` for bounded decoded-reference reuse. Producer values
are retained only after numerical agreement is verified.

To overlap evaluation with a running producer on the same host, add
`--follow-pid PID --threads 2`. Only published verified outputs are consumed;
source failure preserves partial scores and stops without automatic retry. The
default deadline is six hours. Final completion requires the full source audit
and exact paired coverage. `--resume-from` accepts a previous companion only
after metric provenance and sampled RGB hashes match. Use a new output directory.
Shared-resource timings are not isolated model-latency measurements.

### Simultaneous viewing

```sh
.venv/bin/python -m benchmarks.super_resolution.make_multiview \
  --manifest benchmarks/super_resolution/artifacts/pairs/manifest.json \
  --run benchmarks/super_resolution/artifacts/reconstructions \
  --clip RECORDING_STEM --output benchmarks/super_resolution/artifacts/multiview
```

The six panels show original / LR / bicubic and EDSR / IMDN / SAFMN. The viewing
copy repeats the timeline with a common detail crop; it currently requires an
even-length 60 fps clip. Viewing copies are lossy and never used as metric inputs.
`preview --input VIDEO --loop-video` exercises the native renderer and model
dropdown without a remote host; `verify_preview --url URL --require-video` checks
the control path and bounded advancing playback.

## Other tools and interpretation

`prepare`, `benchmark` and `report` provide a separate still-frame protocol;
`compare_precision` compares matched FP32/AMP runs; `verify_native` checks official
preprocessing/quantization; `verify_video_repeat` checks separate repeated runs.
Use `python -m benchmarks.super_resolution.MODULE --help` for arguments.

Report PSNR ↑, SSIM ↑ and LPIPS ↓ with equal-clip means and paired gains over
bicubic. Adjacent frames and clips from one recording session are not independent
test sessions. Decoded references are proxies for the original recordings, not
lossless acquisition. Keep still-image and complete-video protocols separate.
Report synchronized inference timings separately from I/O, renderer latency and
end-to-end client FPS. No OCR evaluation or fine-tuning is performed here.
