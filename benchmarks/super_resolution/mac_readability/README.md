# Mac text readability experiment

This package contains the source preparation, real Mac VideoToolbox encoding,
GPU replay, source-detail transport, and fixed OCR evaluation for the November
2026 text-readability campaign. It is an experimental protocol, not a deployed
replacement for Apple Screen Sharing.

The completed independent confirmation is in
`../experiments/mac-readability-v2-20261102/confirmation-v1/REPORT.html`.
Its `RESULTS.json` contains per-category quality, errors introduced and repaired,
traffic, and local versus end-to-end timing. `IDLE-TAIL-RESULTS.json` measures
completion after the final distinct decoded input for all 32 source states.

## Frozen conditions

- Sender: the existing M1 Mac; receiver: lab RTX 4090, UUID
  `GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a`.
- Actual Mac HEVC: 720p, requested 600 Kbps, 8-bit 4:2:0, BT.709 limited range.
  Frames are decoded to RGB and EDSR runs at ×2 with AMP FP16, then area-resizes
  to 1920 × 1080. HEVC decoding is software; inference and rendering use the GPU.
- The selected trained checkpoint SHA256 is
  `01d99513d182f669e1e1e92c88e0e17a9a053b6e2d046a8733119047e61db4cd`.
- One forward application/framing budget of 600 Kbps carries video and source
  detail, including an 80-byte allowance per fragment. OS SSH counters are
  measured separately. This is not a strict physical IP/VPN bandwidth cap.
- Sources are announced by generation. Old work is rejected; accepted SR stays
  visible until a newer valid same-source result arrives. Exact source detail
  is encoded using identical luma-component reuse and lossless chroma coding.
- The frozen candidate blends 25% native source with 75% current-source SR,
  and recomposes when a newer valid SR result finishes. This ratio refers to
  pixels, not to bandwidth. The complete source supplement is transmitted.

## Independent data and selection

`datasets/sources/mac-readability-confirmation-v2-20261103` contains 16 new
document groups and 32 native Mac captures. All font variants remain in the
same group. Definitions record repository, normalized-text, near-duplicate,
and native-training isolation checks. The earlier v1 definitions and failed
capture are preserved: one mixed-language code line exceeded native column
width and was shortened before any model scoring. The original text and width
measurements are recorded in the v2 definition.

`INDEPENDENT-CONFIRMATION-PLAN-V1.json` and
`INDEPENDENT-COMPARATOR-ADDENDUM.json` in the parent experiment define the test.
Both existing model controls were scored before the candidate. The lower
full-set document CER selected the trained EDSR; its 13 positive-error groups
were frozen in `confirmation-v1/COMPARATOR-FROZEN.json`. Candidate settings
were not tuned on this confirmation set. Historical final test was not opened.

OCR preprocessing remains source-box cropping, source-defined dark inversion,
Lanczos ×3, a 12-pixel white border, Tesseract OEM 1 / PSM 7 / 300 dpi, with
frozen English and Chinese language files. Raw OCR outputs and hashes are kept.
Code nonspace exactness and raw whitespace exactness are separate; the latter
does not reliably measure rendered indentation. Clean-source OCR errors are
retained. OCR is not a human preference measurement.

## Reproduce analysis from saved evidence

From `/home/jchen73/iShareScreen`, activate the project environment:

```bash
source dev-environment.sh
export MPLCONFIGDIR=/tmp/isharescreen-matplotlib
```

Rebuild the independent report without changing source data or rescoring OCR:

```bash
python -m benchmarks.super_resolution.mac_readability.confirmation_report \
  --root benchmarks/super_resolution/experiments/mac-readability-v2-20261102/confirmation-v1 \
  --source datasets/sources/mac-readability-confirmation-v2-20261103/reference
```

Audit source/SR operands and actual final GPU images:

```bash
python -m benchmarks.super_resolution.mac_readability.stream_mix_audit \
  --root benchmarks/super_resolution/experiments/mac-readability-v2-20261102/confirmation-v1 \
  --run-name confirm-refresh-01
python -m benchmarks.super_resolution.mac_readability.confirmation_latency \
  --root benchmarks/super_resolution/experiments/mac-readability-v2-20261102/confirmation-v1
```

To rescore images, use `dense_quality --score-index` with the corresponding
`OFFICIAL-IMAGES.json`, `TRAINED-IMAGES.json`, `NATIVE-IMAGES.json`, or
`stream-runs/confirm-refresh-01/FINAL-IMAGES.json`. Always choose a **new output
directory**. `score_lines` evaluates the timed image indexes at receiver ages
0.5, 1, 2, and 2.8 seconds, retaining blank and late displays.

## Run another transport replay

The Mac experiment is
`/Users/alex/iShareScreen-experiments/text-confirmation-v1-20261103`.
Its primary `tools/stream_sender.py` is the frozen candidate sender; the
separate `stream_sender_idle_probe.py` adds a one-second idle tail for timing.
The source videos and raw 4:2:0 frames are hash-checked before sending. Use a
new run name. The receiver verifies the GPU UUID and refuses competing GPU
compute processes before loading the checkpoint.

```bash
/Users/alex/iShareScreen/.venv/bin/python tools/stream_sender.py \
  --root /Users/alex/iShareScreen-experiments/text-confirmation-v1-20261103 \
  --run-name your-new-run-name \
  --mode enhanced --policy hold --source-weight 0.25 --refresh-source-mix
```

Run names must be lowercase letters, digits, underscores, or hyphens; replace
the placeholder accordingly. The sender uses authenticated SSH to
`chen@10.193.124.90`. Receiver results go under the main experiment's
`stream-runs`, not automatically under `confirmation-v1`. Copy the completed
sender and receiver evidence into the analysis directory before indexing it.
Do not synchronize large files while measuring the link.

The idle-tail probe changes only replay scheduling: original packet spacing
and bytes remain intact, with one extra second before switching sources.
Its results validate local completion, and do not replace primary quality
scores or establish a 200 ms end-to-end response.

## Interpretation

Source detail is the main practical improvement. On independent material,
source detail alone and the frozen blend have similar OCR accuracy; exact
source detail is substantially more faithful in PSNR/SSIM. The experiment
does not establish an advantage from blending once complete source pixels
are available. Cell error rate can regress despite a lower line CER.

`REPORT.html`, `BLIND-KEY.json`, fixed native-pixel crops, and the full
regression lists preserve these tradeoffs. Actual capture/encoding failures,
frame drops, missed deadlines, and censored observations remain in their
original records. No production deployment, NVDEC, physical-panel latency,
or human-preference result is claimed.
