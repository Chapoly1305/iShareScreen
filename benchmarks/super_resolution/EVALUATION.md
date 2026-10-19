# Pretrained enhancement evaluation protocol

## Scope

Compare eligible original ×2 pretrained checkpoints before any fine-tuning, using
the same inference adapter as the native desktop viewer. Missing comparable
checkpoints are reported as unavailable, never replaced by random weights or an
unattributed alternative. Pinned source revisions and checkpoint hashes are in
`models.lock.json`.

Primary metrics are RGB PSNR ↑, SSIM ↑ and LPIPS ↓. No OCR labels, OCR scoring,
training, live transport testing or human-subject study is part of this phase.
Recorded-video reconstruction is the principal validation; still-frame experiments
provide complementary quality/latency measurements.

## Data and controls

1. Inventory recordings and inspect content before inclusion. Preserve original
   files; record hashes, codec, geometry, frame rate and capture-session grouping.
   Related clips and adjacent frames are not independent sessions.
2. For the complete-video protocol, accept 1920×1080 sources, export complete RGB8
   decoded references losslessly, and reduce to 960×540 with FFmpeg Lanczos.
   Encode/decode HEVC RExt 4:4:4, CRF 28, full-range BT.709, inter-frame prediction
   and maximum GOP 120. Preserve frame counts, native frame rates and timestamps.
3. Compare bicubic with eligible EDSR-baseline, IMDN and SAFMN checkpoints. Use
   official preprocessing and strict tensor loading. Primary inference is FP32,
   without tiling, self-ensemble, test-time augmentation or compilation.
4. Reconstruct and save every frame losslessly. Re-decode the complete output and
   compare its RGB-sequence hash with the pre-encode result. Reject missing frames,
   timestamp shifts, unexpected geometry and nonfinite outputs.
5. Sample quality every 60 frames plus the final frame, deduplicating that final
   index. All methods must use exactly the same samples. Report equal-clip means
   and paired model-minus-bicubic differences; retain per-sample values.

PSNR/SSIM use RGB uint8 and data range 255. SSIM uses the scikit-image defaults
(7×7 uniform window and sample covariance). LPIPS is AlexNet v0.1, FP32, native
1920×1080 RGB normalized to [-1,1], without resizing or metric tiling. Record the
backbone, calibrated weights and loaded network-state hashes. CUDA FP64 pixel
verification may be used only after agreement with producer values; retain the
original producer PSNR/SSIM after checking.

Exact consecutive-input reuse is an optional offline optimization, with explicit
inference/reuse counts and every output frame still written. Parallel continuation
uses disjoint clips, immutable completed results, source-bound cache reuse and
full coverage audits. Neither optimization measures live throughput.

## Separate still-image and precision experiments

The still-image protocol extracts three timestamps per eligible clip, downsamples
with antialiased Pillow bicubic, and compares clean inputs with intra-frame HEVC
CRF 18/28/38. Include bilinear, bicubic and bicubic-plus-sharpening baselines.
Report full RGB and common two-pixel-border results, LPIPS, paired gains and
capture-weighted means. Do not pool these values with the inter-frame video run.

Time synchronized preprocessing, transfers, forward pass and quantization
separately; report warm total p50/p95, first-use cost, repeat count and peak CUDA
allocation. Exclude disk I/O, canvas assembly and rendering. Matched AMP experiments
must use the same source/weights/inputs and report quality changes, not just speed.
Pure-half IMDN is rejected because its spatial reductions can overflow at 540p;
the AMP path keeps those reductions in FP32. Keep failure evidence visible.

## Acceptance gates

- A working optional ML environment, pinned dependencies, dependency checks and
  platform snapshots; ordinary viewer installation remains ML-optional.
- Official model/checkpoint provenance and pixel/preprocessing checks for every
  eligible model; explicit unavailable status for ineligible candidates.
- Audited still-image and complete-video coverage, raw metrics, descriptive paired
  comparisons, and hardware-specific timing with correct scope labels.
- Post-login selection reaches the actual native renderer, with asynchronous
  inference, bounded queues, stale-generation rejection and Original bypass.
  Check control dispatch, real GPU readback and advancing recorded-video playback.
- Playable original/model comparisons and a simultaneous multi-model demo. Check
  boundaries, representative native-size UI/text crops and continuous playback.
  Do not claim formal temporal stability from image-quality means.

Decoded references are proxies for the original lossy recordings, not lossless
acquisition. This exploratory capture set is not a held-out independent-session
test set. Future training must use session-disjoint splits and a fresh evaluation
boundary; these results do not establish target-receiver FPS, input latency,
network savings, text correctness or human preference.
