# Final results: low-bitrate desktop super-resolution

Canonical project results — 2026-11-03. Open [index.html](index.html) for seven scenes with synchronized before/after sliders. Reference is above the grid; bicubic is first left and EDSR first right. All media are local and can be viewed offline.

## Main findings

With 720p / 600 Kbps / 4:2:0 input, fine-tuned EDSR achieves the highest validation PSNR: **31.504 dB**, up **1.922 dB** over its official pretrained initialization and **4.019 dB** over direct bicubic resizing (27.485 dB). Its RTX 4090 forward p95 is **64.246 ms**.

QuickSRNet is the highest-quality model among the three models with measured forward p95 below 2 ms: **29.689 dB** at **1.780 ms**, compared with PlainUSR at 29.500 dB / 1.428 ms and RT4KSR at 28.975 dB / 1.382 ms. These are quality/compute tradeoffs, not guarantees of live interaction performance. EfRLFN has the largest improvement over its own pretrained baseline (+2.945 dB), but not the highest absolute quality.

## Aggregate validation results

| Method | Best epoch | Pretrained PSNR (dB) | Fine-tuned PSNR (dB) | Gain (dB) | RTX 4090 p50 (ms) | RTX 4090 p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| EDSR | 25 | 29.581 | 31.504 | +1.922 | 64.211 | 64.246 |
| IMDN | 85 | 29.533 | 31.362 | +1.829 | 51.780 | 51.808 |
| SAFMN | 85 | 29.540 | 30.806 | +1.266 | 82.441 | 82.472 |
| EfRLFN | 100 | 26.865 | 29.810 | +2.945 | 37.499 | 37.541 |
| QuickSRNet | 100 | 29.283 | 29.689 | +0.406 | 1.678 | 1.780 |
| PlainUSR | 30 | 29.027 | 29.500 | +0.474 | 1.411 | 1.428 |
| RT4KSR | 95 | 28.481 | 28.975 | +0.494 | 1.362 | 1.382 |

No-SR control: direct 720p → 1080p bicubic resizing gives **27.485 dB**. GPU PyTorch bicubic p95 is **0.304 ms** (actual FP32); CPU OpenCV bicubic p95 is **2.048 ms** on the same host. The gallery shows the CPU OpenCV output and explicitly labels its CPU timing. The alternative 720p → 1440p → 1080p bicubic path scores 27.126 dB; its measured GPU resize-to-1440p p95 is 0.477 ms, excluding the final resize.

## Protocol and interpretation

- All seven models were trained for 100 epochs from official pretrained weights with fresh optimizers. The table uses each model's best Mac-validation checkpoint, not necessarily epoch 100.
- CodeSCAN split-v1 is frozen at 9,600 train / 1,200 validation / 1,200 test. Mac recordings and CodeSCAN validation/test are excluded from training. No test-set score is claimed here.
- Quality covers 24 Mac validation clips, three fixed frames per clip (72 frames), with equal weighting per clip. The seven gallery scenes are illustrative examples from this validation set, not the entire evaluation set.
- Input is offline HEVC/x265 encoding at 1280×720, target 600 Kbps, yuv420p. 4:2:0 was selected for broader GPU video-decoder compatibility; support still depends on codec, profile and device. These are not captured Apple Screen Sharing bitstreams.
- Quality uses FP32 inference. Native ×2 SR produces 2560×1440, followed by area resizing to 1920×1080. Reference images are the existing processed RGB8 validation references. PSNR is RGB pixel fidelity, not an OCR or human readability score.
- Latency was measured serially on one RTX 4090: batch 1, AMP FP16, 50 warmups and 1,000 timed calls per fine-tuned checkpoint. A fixed synthetic 720p input is reused; the calls are not 1,000 distinct evaluation images. CUDA events measure forward inference, excluding decoding, transfers, final 1080p resizing, queues and display. Pretrained latency was not measured separately.
- Quality and latency were measured separately with different precisions. AMP quality equivalence over the entire validation set is not established by this table. No RTX 5090 or Mac latency claim follows from the RTX 4090 results.
- Checkpoints were selected on the same Mac validation set used for this report. These are development-validation results, not an independently held-out final test estimate.

## Visual failure case

The scene previously labeled “Dense web table” is a Chinese webpage with tables (recording 2026-10-15_17-44-40, encoded frame 326). It remains in the gallery. Independent sequential decoding confirms the reference and encoded frames match the intended source frame; this is not an HTML alignment or seek error. The frame is shortly after a content change, rather than a long-settled screen.

Smearing is already present in the decoded low-bitrate input. Downsampling and compression discard text detail, and SR can retain or amplify incorrect edges. The matched 720p comparison improves from 24.669 dB at 600 Kbps to 26.031 dB at 1 Mbps for this frame. This supports compression damage as a contributor, but does not isolate one encoder setting as the sole cause. Encoder configuration remains part of our experimental responsibility. No claim is made that every scene becomes clearly readable.

## Scope of this final result

This release presents receiver-side SR without original-image supplements or source-detail blending. Earlier supplemental-transfer experiments are historical work and are excluded from these results. Adaptive switching between fast and high-quality models is a possible future extension, not a delivered or evaluated result.

## Evidence and reproducibility

`results.csv` contains the summary table. `training-curves.svg` plots all seven validation trajectories. `evidence/training-curves.csv` retains quality and hardware-labeled historical training/performance observations; use the dedicated RTX 4090 latency JSON for cross-model timing comparisons. `evidence/latency.json` retains all 7,000 raw model timing samples and checkpoint hashes. Bicubic measurements, encoding protocol and failure audits are also included. Each scene includes its original manifest and PNGs. `SHA256SUMS.json` binds every release file except itself.

Datasets and model weights remain in their existing experiment locations; this presentation bundle does not duplicate them. Source tools in `benchmarks/super_resolution`: `seven_model_gallery.py`, `benchmark_best_latency.py`, `benchmark_bicubic_latency.py`, `bicubic_quality.py`. This release consolidates completed measurements; it does not launch new training or claim a new evaluation run.

## Static presentation

This directory is the single maintained results release in the repository.
Open `index.html` directly, or publish this directory as the root of any static
website. No build step, CDN, model runtime or server-side code is needed. Keep
all seven scene directories beside `index.html`. The report, CSV, SVG and
measurement evidence are part of the same release.

Historical experiments are archived separately on the development host and
are not needed to view or host this site. The repository root README links here;
there are no separate legacy results or performance reports to choose between.
