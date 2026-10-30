# HEVC Chroma Sampling Comparison

The evaluation covers 24 Mac validation videos, four input resolutions
(540p, 576p, 720p, and 1080p), two target bitrates, and eight reconstruction
methods: seven official pretrained SR models and a no-SR baseline. Three
matched frames per video are evaluated at a common 1920 × 1080 display size.
SR inference uses FP32. The table averages videos, frames, resolutions, and
methods equally.

| Target bitrate | Metric | 4:2:0 | 4:4:4 | Difference (444 − 420) | Preferred format |
| --- | --- | ---: | ---: | ---: | --- |
| 600 Kbps | PSNR ↑ (dB) | 28.130 | 28.351 | +0.221 | 4:4:4 |
| 600 Kbps | SSIM ↑ | 0.92833 | 0.92943 | +0.00110 | 4:4:4 |
| 600 Kbps | LPIPS ↓ | 0.09640 | 0.09761 | +0.00121 | 4:2:0 |
| 1 Mbps | PSNR ↑ (dB) | 29.280 | 29.745 | +0.465 | 4:4:4 |
| 1 Mbps | SSIM ↑ | 0.94352 | 0.94519 | +0.00167 | 4:4:4 |
| 1 Mbps | LPIPS ↓ | 0.07346 | 0.07370 | +0.00025 | 4:2:0 |

Differences are calculated before rounding.

At the same target bitrate, 4:4:4 provides modest gains in PSNR and SSIM,
while 4:2:0 achieves slightly lower mean LPIPS. Inspection of the selected
text crops, both before and after SR, showed limited visible differences.
These results do not demonstrate a consistent perceptual advantage for 4:4:4.

**We select 4:2:0 for subsequent evaluations to prioritize broader compatibility
with GPU hardware video decoders and client playback pipelines.** NVIDIA's
decoder matrix lists HEVC Main/Main10 support on older GPU generations where
HEVC 4:4:4 decoding is unavailable, while Microsoft's standard HEVC decoder
documents 4:2:0 as its supported chroma format.
See the [NVIDIA NVDEC capability matrix](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvdec-application-note/index.html#nvdec-capabilities)
and [Microsoft HEVC decoder constraints](https://learn.microsoft.com/en-us/windows/win32/medfound/h-265---hevc-video-decoder#format-constraints).
The small image-quality trade-off favors using 4:2:0 as the practical default.

## Scope and reproducibility

- All 9,216 requested method/frame observations are included. Twelve of the
  384 encodes failed the predefined bitrate/VBV check and remain explicitly
  flagged in the full experiment. This is a comparison at requested video
  bitrates, not a proof of constrained-link compliance.
- Encoding settings were identical across two CPU hosts, but their encoder
  builds differed: x265 3.5+1 on H100 and x265 4.1+1 on lab. Each matched
  4:2:0/4:4:4 pair was encoded on the same host with the same build. The
  aggregate therefore represents this two-build campaign.
- The videos belong to one capture session. These image-quality metrics and
  selected screenshots do not establish text-recognition accuracy, statistical
  equivalence, or live-client performance.
- The final machine-readable values are retained in
  [the paired comparison data](analysis/chroma-paired-validation-2026-10-29.json).
  Full inputs, predictions, logs, and checks are local artifacts under
  `benchmarks/super_resolution/experiments/validation-baselines-2026-10-29/`.
