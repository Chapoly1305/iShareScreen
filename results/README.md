# Final results: low-bitrate desktop super-resolution

The primary results are now the completed **independent CodeSCAN test-set evaluation**, released on 2026-11-03. [Open the interactive presentation](index.html). Its seven visual examples remain Mac validation scenes and are labeled separately.

## Main findings

At 720p / 600 Kbps / 4:2:0, fine-tuned **EDSR achieves 36.213 dB** on 1,200 held-out CodeSCAN samples: **+3.674 dB** over its official pretrained version and **+5.163 dB** over Bicubic (31.049 dB). IMDN reaches 36.114 dB, only 0.099 dB below EDSR. EDSR also has the best average SSIM (0.98428) and LPIPS (0.01614) among the evaluated methods.

All seven fine-tuned models improve mean PSNR, SSIM and LPIPS over their pretrained versions. These averages do not imply improvement on every sample. QuickSRNet has the highest test quality among models with measured RTX 4090 forward p95 below 2 ms: **33.561 dB at 1.780 ms**. EDSR's forward p95 is **64.246 ms**; IMDN's is **51.808 ms**. Quality and latency are separate measurements, not end-to-end performance.

## Primary results: independent test set

| Method | Epoch | Pretrained PSNR (dB) | Fine-tuned PSNR (dB) | Gain (dB) | Fine-tuned SSIM ↑ | Fine-tuned LPIPS ↓ | RTX 4090 p95 (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EDSR | 25 | 32.538 | 36.213 | +3.674 | 0.98428 | 0.01614 | 64.246 |
| IMDN | 85 | 32.506 | 36.114 | +3.608 | 0.98375 | 0.01708 | 51.808 |
| SAFMN | 85 | 32.471 | 35.402 | +2.931 | 0.98073 | 0.02042 | 82.472 |
| EfRLFN | 100 | 30.422 | 33.985 | +3.563 | 0.97454 | 0.02742 | 37.541 |
| QuickSRNet | 100 | 32.273 | 33.561 | +1.287 | 0.97262 | 0.03076 | 1.780 |
| PlainUSR | 30 | 32.153 | 33.153 | +1.001 | 0.96942 | 0.03386 | 1.428 |
| RT4KSR | 95 | 31.868 | 32.497 | +0.629 | 0.96300 | 0.04064 | 1.382 |

Bicubic control: **31.049 dB PSNR / 0.95620 SSIM / 0.07095 LPIPS**. Higher PSNR and SSIM are better; lower LPIPS is better. Model latency is the existing same-checkpoint RTX 4090 measurement, not a new timing run on test images.

## All test quality measurements

| Method | Weights | PSNR (dB) ↑ | SSIM ↑ | LPIPS ↓ |
| --- | --- | --- | --- | --- |
| Bicubic | No SR | 31.049 | 0.95620 | 0.07095 |
| EDSR | pretrained | 32.538 | 0.96481 | 0.03840 |
| EDSR | finetuned | 36.213 | 0.98428 | 0.01614 |
| IMDN | pretrained | 32.506 | 0.96450 | 0.03756 |
| IMDN | finetuned | 36.114 | 0.98375 | 0.01708 |
| SAFMN | pretrained | 32.471 | 0.96411 | 0.04005 |
| SAFMN | finetuned | 35.402 | 0.98073 | 0.02042 |
| EfRLFN | pretrained | 30.422 | 0.95215 | 0.08462 |
| EfRLFN | finetuned | 33.985 | 0.97454 | 0.02742 |
| QuickSRNet | pretrained | 32.273 | 0.96271 | 0.04282 |
| QuickSRNet | finetuned | 33.561 | 0.97262 | 0.03076 |
| PlainUSR | pretrained | 32.153 | 0.96262 | 0.04427 |
| PlainUSR | finetuned | 33.153 | 0.96942 | 0.03386 |
| RT4KSR | pretrained | 31.868 | 0.96073 | 0.04740 |
| RT4KSR | finetuned | 32.497 | 0.96300 | 0.04064 |

## Frozen test protocol

- CodeSCAN split-v1 contains 9,600 train / 1,200 validation / 1,200 test samples. All 1,200 test samples are included (50 for each of 24 languages), with equal sample weighting. Fifteen methods produce 18,000 per-sample records.
- All seven models completed 100 training epochs from official pretrained initialization. Checkpoints were selected using Mac validation and locked by SHA256 before test inference. No checkpoint, hyperparameter or input recipe was selected using test scores.
- Source manifest hashes, repository/group separation, source-file hashes and generated LR/HR file and pixel hashes were checked for overlap with train/validation. Generated test pairs also passed within-test uniqueness checks. These checks establish the recorded split isolation, not universal semantic independence of all code.
- Each screenshot is fitted to 1920×1080 with preserved aspect ratio and corner-color padding. The frozen training recipe creates 60 frames at 30 fps: 30 static frames followed by a synthetic vertical translation of up to 30 pixels. Aligned frame 45 is evaluated. This is synthetic screenshot motion, not a real screen recording.
- Whole frames are encoded with libx265 fast / zerolatency, 720p, target 600 Kbps, yuv420p, full-range BT.709, GOP 60 and no B-frames. Every sample passed the existing bitrate/VBV checks: no reported underflow and at most 630 Kbps after the first second. 4:2:0 is used for broader GPU video-decoder compatibility, subject to codec, profile and device support.
- Inference and quality scoring use FP32 model inference with TF32 disabled. Native ×2 SR produces 2560×1440, followed by area resizing to 1920×1080. Bicubic directly resizes 720p to 1080p. Metrics use complete frames, including padding: RGB PSNR, RGB SSIM with valid 7×7 windows and sample covariance, and native-1080p LPIPS AlexNet v0.1. The RGB SSIM implementation uses float64 integral sums.
- Means are per-image metric means, not pooled-pixel PSNR. Machine-readable results also retain language-level means. PSNR is pixel fidelity, not an OCR accuracy or human readability measurement. No claim is made that every scene becomes readable.

## Latency protocol

Timing is from the existing serial RTX 4090 benchmark: batch 1, AMP FP16, 50 warmups and 1,000 timed calls per fine-tuned checkpoint. A fixed synthetic 720p input is reused. CUDA events measure resident forward inference to 1440p, excluding decoding, transfers, final 1080p resizing, queues and display. Pretrained latency was not measured separately.

Quality uses FP32; timing uses AMP FP16. Full-test AMP quality equivalence has not been measured. These timings do not establish RTX 5090, Mac or live end-to-end performance. Direct GPU PyTorch bicubic p95 is 0.304 ms (actual FP32); CPU OpenCV bicubic p95 is 2.048 ms on the same host. The visual gallery uses CPU OpenCV bicubic and labels its timing accordingly.

## Supplementary Mac validation and visual examples

The existing Mac validation evaluation covers 24 clips × 3 fixed frames = 72 frames, with equal clip weighting. EDSR achieves 31.504 dB versus its pretrained 29.581 dB and Bicubic 27.485 dB. These numbers are retained in [validation-results.csv](validation-results.csv), but they are not the primary test results. The validation data were used to select checkpoints.

The seven interactive scenes come from that Mac validation set. Each card's PSNR describes its own frame; it must not be labeled as a CodeSCAN test score. [Training curves](training-curves.svg) also show Mac validation history, not test-set trajectories. The Mac images and CodeSCAN test data have different content and generation procedures; their absolute PSNR values must not be compared as an improvement from one dataset to the other. Neither evaluation is a captured Apple Screen Sharing bitstream.

## Visual failure case


The scene previously labeled “Dense web table” is a Chinese webpage with tables (recording 2026-10-15_17-44-40, encoded frame 326). It remains in the gallery. Independent sequential decoding confirms the reference and encoded frames match the intended source frame; this is not an HTML alignment or seek error. The frame is shortly after a content change, rather than a long-settled screen.

Smearing is already present in the decoded low-bitrate input. Downsampling and compression discard text detail, and SR can retain or amplify incorrect edges. The matched 720p comparison improves from 24.669 dB at 600 Kbps to 26.031 dB at 1 Mbps for this frame. This supports compression damage as a contributor, but does not isolate one encoder setting as the sole cause. Encoder configuration remains part of our experimental responsibility. No claim is made that every scene becomes clearly readable.

## Scope

This release evaluates receiver-side SR with no original-image supplements or source-detail blending. Adaptive switching between fast and high-quality models remains future work, not an evaluated result. The independent test improves the evidence for held-out CodeSCAN generalization; real remote-desktop readability, temporal artifacts and end-to-end latency still need separate evaluation.

## Evidence and reproduction

[results.csv](results.csv) is the primary test summary. [test-quality.svg](test-quality.svg) plots test PSNR. The complete test summary, frozen plan, runtime provenance and 18,000 records are in `evidence/test/`. `AUDIT.json` records verification of identities, aggregate metrics and the checkpoint match between test quality and existing latency. Frozen source images and full input manifests remain in the research archive; their hashes are bound in the plan and are not bundled in this static presentation.

`evidence/latency.json` retains 7,000 model timing samples. `validation-results.csv`, `training-curves.svg`, the original scene manifests and prior evidence remain supplementary. `SHA256SUMS.json` binds all release files except itself. The evaluator is `benchmarks/super_resolution/codescan_test.py`; it uses the unchanged codec recipe implementation and fixed checkpoint hashes.

## Static presentation

This is the repository's single maintained results release. Open `index.html` directly or publish this directory as the root of a static website. No build, CDN, Python runtime, GPU or backend is required. Keep all scene folders and evidence files beside the HTML pages. The repository root README points here.
