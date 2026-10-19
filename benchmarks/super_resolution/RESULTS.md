# Pretrained desktop-video enhancement results

Measured 2026-10-19. This phase uses original pretrained weights, with no
fine-tuning. The [evaluation protocol](EVALUATION.md) and [commands](README.md)
describe reproduction; `models.lock.json` records the official source revisions
and checkpoint hashes. ECBSR is unavailable because an eligible original ×2
checkpoint was not established; no substitute or random-weight result is included.

## Complete recorded videos

24 complete 1920×1080 recordings, 53,763 source frames at 60 fps (896.05 seconds),
were reduced to 960×540 and encoded as HEVC RExt 4:4:4 CRF 28, full-range BT.709,
with Lanczos reduction and inter-frame prediction. Original recordings are lossy;
the lossless RGB8 references preserve their decoded pixels, not lossless acquisition.

Bicubic, EDSR-baseline, IMDN and SAFMN each produced all 24 lossless reconstructions:
96 videos and 215,052 output frames. PSNR/SSIM/LPIPS use the same 931 samples per
method (every 60 frames plus the final frame), totaling 3,724 paired measurements.
Means below weight each clip equally, not each frame or clip duration.

| Method | PSNR dB ↑ | SSIM ↑ | LPIPS ↓ | Paired ΔPSNR vs bicubic |
|---|---:|---:|---:|---:|
| Bicubic | 26.085 | 0.92061 | 0.14789 | — |
| EDSR-baseline ×2 | 28.785 | 0.94785 | 0.05649 | +2.700 |
| IMDN ×2 | 28.648 | 0.94652 | 0.05167 | +2.563 |
| SAFMN ×2 | 28.646 | 0.94681 | 0.05776 | +2.561 |

EDSR has the highest mean PSNR and SSIM in this set. IMDN has the lowest mean
LPIPS, so there is no single winner across all quality measures. All three neural
models improve clip-mean PSNR over bicubic on every included clip. Their minimum /
maximum paired clip gains are EDSR +1.767 / +3.850 dB, IMDN +1.735 / +3.755 dB and
SAFMN +1.900 / +3.707 dB. These are descriptive paired results, not significance
tests or independent-session generalization.

LPIPS is AlexNet v0.1, FP32, native 1080p RGB in [-1,1], without resizing or metric
tiling. The original/calibrated weights and loaded state are hash-bound. PSNR/SSIM
verification on CUDA uses FP64 and must agree with producer values before retaining
those original scores. 3,510 existing LPIPS scores were reused only after matching
source identity and re-decoded sampled RGB hashes; 214 remaining scores were added.
The continuation used three isolated workers with disjoint clips and fresh outputs.

Each method reused 16,276 exact consecutive LR frames and wrote every output frame;
37,487 actual processing calls remained per method. This is an offline quality
optimization, not an estimate of sustained viewer FPS. Shared-GPU full-video
timings are excluded from the latency comparison below.

## Separate still-image quality and timing

The still-image campaign used 288 paired inputs: three timestamps from each of
24 clips, with clean antialiased bicubic reduction and intra HEVC CRF 18/28/38.
Six methods produced 1,728 measurements; a matched three-model AMP campaign added
864. These results are not pooled with the inter-frame video table.

| Model | Clean ΔPSNR vs bicubic | CRF 28 ΔPSNR | CRF 38 ΔPSNR |
|---|---:|---:|---:|
| EDSR-baseline | +4.890 dB | +1.919 dB | +0.254 dB |
| IMDN | +5.110 dB | +1.800 dB | +0.162 dB |
| SAFMN | +4.869 dB | +1.828 dB | +0.219 dB |

Severe compression sharply narrows the reconstruction gains. Native-size crops
still show damaged or distorted small glyphs: better mean scores do not establish
correct text recovery. No OCR evaluation or text-accuracy claim is made.

Warm synchronized 960×540 → 1920×1080 processing on NVIDIA H100 NVL:

| Model | FP32 total p50 / p95 ms | AMP total p50 / p95 ms | AMP mean ΔPSNR vs FP32 |
|---|---:|---:|---:|
| EDSR-baseline | 54.57 / 56.88 | 20.08 / 20.61 | -0.00053 dB |
| IMDN | 34.96 / 36.18 | 18.23 / 18.30 | -0.00143 dB |
| SAFMN | 42.49 / 42.56 | 33.70 / 33.84 | +0.00071 dB |

Processing includes preparation, synchronized transfers, forward pass and output
quantization; it excludes disk I/O, canvas assembly, native rendering and input
latency. The maximum individual AMP PSNR loss was 0.01159 dB. Pure-half IMDN was
rejected for nonfinite spatial reductions; AMP retains FP32 reductions. These
H100 measurements do not establish RTX 4090 performance or live client FPS.

IMDN is a useful pretrained efficiency candidate here: lowest video LPIPS and
lowest measured neural processing p50. EDSR remains the stronger PSNR/SSIM
reference. Neither statement establishes interactive deployability.

## Validation and viewing

- Every reconstructed video was fully re-decoded by its producer and matched its
  pre-encode RGB-sequence hash. The assembled files and inputs were hash-checked.
- A separate audit checked exact clip/model/frame coverage, timestamps, raw-to-clip
  means, all three equal-clip means, paired deltas, metric-state identity, preserved
  cache provenance and child completion hashes. It accepted all 3,724 rows.
- Twelve 3840×1080 original/model H264 viewing copies cover four representative
  recordings and all three models. Their frame counts, frame rates and hashes
  passed checks. Start/middle/end images and native-size text/UI crops were visually
  inspected for all twelve; the six-panel overview/detail demo was played in a
  normal video player. This is qualitative inspection, not a formal flicker metric.
- The post-login native display path was exercised with advancing recorded HEVC:
  installed models reached GPU presentation, rapid reselection worked, Original
  bypass returned, unavailable selections were rejected and pending queues stayed
  bounded. MPS preprocessing/quantization fixtures matched official helpers exactly
  for all three models. This does not validate live transport or real-time MPS SR.
- Final checks: macOS 162 tests passed with GPU readback enabled (five CUDA-only
  cases skipped); Linux/H100 165 passed (two opt-in display-GPU cases skipped).
  Wheel build, ML-optional packaged import/CLI checks and dependency checks passed.

Local outputs, intentionally excluded from Git, are under
`artifacts/video-parallel-collected-v2-20261019/`: `reconstructions/`,
`perceptual/{quality.csv,summary.json,environment.json,REPORT.md,COMPLETED.json}`,
`INDEPENDENT_AUDIT.json`, `comparisons/` and `visual-qa/`. Open
`all-models-overview-and-detail.mp4` for original / LR / bicubic / EDSR / IMDN /
SAFMN simultaneously; it contains the same 21.8-second clip as overview and detail
chapters (43.6 seconds, 30 fps viewing copy). Evaluation outputs remain 60 fps.
The demo's underlying decoded model sequences match the full run's first clip.

Source recordings, checkpoints and large experiment outputs are not distributed
with the repository. Frozen execution snapshots and hashes remain in the local
provenance package; portable tool organization changed without rewriting those
historical records. Platform dependency snapshots accompany these tools.

## Limits and next phase

The 24 clips share a capture session; this is exploratory evaluation, not a held-out
session-disjoint test. Some boundaries include recording controls and provisional
content labels are not task-level ground truth. No statistical significance,
temporal stability guarantee, human preference, bandwidth saving or end-to-end
responsiveness is established. Live-machine testing and OCR are outside this phase.

Future fine-tuning needs session-disjoint train/validation/test groups and a fresh
evaluation boundary. No fine-tuning or trained checkpoint is delivered here.
