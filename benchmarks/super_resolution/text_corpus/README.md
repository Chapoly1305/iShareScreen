# Desktop text corpus and scratch experiment

Historical scratch-training corpus. This 10,000-view corpus and its split are
not the CodeSCAN dataset used for the final seven-model results. The current
training/test protocol is in [TRAINING.md](../TRAINING.md), and the only published
results are in [results/](../../../results/README.md). The commands below are
retained for reproducing the archived corpus, not for restarting a current queue.

This pipeline creates **10,000 new FHD desktop views**, not 10,000 independent
documents. The first source snapshot contains 293 unique local Zotero PDFs and
5,518 source files from this project and the local Matter SDK. PDFs are read
without modification. Frozen sources, generated data and weights stay in the
Git-ignored `benchmarks/super_resolution/artifacts/text-corpus-v1/` directory.

The generation buckets contain 6,000 paper viewports, 1,000 specification-related viewports,
2,000 code views, 500 HTML reader views, 250 numerical tables and 250 terminals.
The specification-related bucket is heuristic: 31 of its 42 PDF sources are
normative specifications; 11 are research papers about specifications. The
admitted bucket has 737 normative views and 263 research-paper views, recorded
in `source-role-review.json`. The bucket name is not a normative-document label. Canonical protocol
families in this snapshot fall in training, so this release does not establish
generalization to an unseen specification family.
The frozen split contains 8,068 training, 752 validation and 1,180 test views.
`dataset-views-v3.json` is the admitted release: two cross-split conference-cover
boilerplate crops were replaced, and one rotated PDF page had its geometric
labels corrected with exact replay of the unchanged source pixels. Earlier
indices and failed audit records remain available as provenance.
DOI/title/protocol families, identical source files and identical full-page
text are grouped before rendering; canonical protocol aliases are checked again
before the final split is admitted. Every page, crop and zoom of a source stays
with its group. The full audit also checks exact visible-text duplicates across
splits. This does not establish absence of every partially shared passage.

PDF words and geometric boxes come from the same embedded text layer and
viewport transform used to render the pixels. HTML code, numbers, paths and
paragraphs are escaped source strings with browser-measured boxes. Only fully
visible words are labeled. No OCR prediction is used as ground truth. PDF text
layers can still contain ligature/reading-order quirks; source truth and visual
geometry are distinct from a claim of perfect transcription on every page.

Each new view contributes one aligned LR/HR training pair. HR is 1920x1080 RGB;
LR is 960x540 RGB decoded from actual 30 fps HEVC 420 episodes. Episodes contain
one-second page holds, with a small synthetic content scroll in a subset.
The retained sample is at local frame 0, 5, 15 or 29; actual I/P frame identities
are recorded. Requested video-rate weights are 10% at 128 Kbps, 20% at 256,
35% at 600, 25% at 1,000 and 10% at 2,000. Rates are assigned deterministically,
so the exact realized counts are in `hevc-v1/pairs.json`.

The encoder uses libx265 fast/zerolatency, no B frames or lookahead, a fixed
two-second GOP and one-second VBV/HRD/strict-CBR requests. Measured encoded rates
and VBV warnings are explicit metadata. These episodes do **not** establish a
physical-link bandwidth limit, Apple encoder equivalence, latency or usability.

The existing 4,155 video frame pairs remain intact in their original frozen
dataset; `legacy-video-pool.json` binds their identities. They are a separate
pixel-supervision pool with no fabricated OCR labels. The first scratch run
uses the new corpus only, so unverified document overlap with old recordings
does not contaminate the new held-out document evaluation.

## Reproduce

Use the project's Python environment with Pillow, NumPy, PyMuPDF, Playwright,
OpenCV, PyAV and PyTorch. PDF rendering uses PyMuPDF; HTML rendering uses the
installed Chrome in a private headless context with external requests blocked.
Zotero must be running with its read-only local API enabled. FFmpeg must include
libx265. Tesseract is required only for the separate OCR evaluation.

```sh
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.catalog \
  --output benchmarks/super_resolution/artifacts/text-corpus-v1
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.render \
  --root benchmarks/super_resolution/artifacts/text-corpus-v1
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.finish_views \
  --root benchmarks/super_resolution/artifacts/text-corpus-v1
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.audit \
  --root benchmarks/super_resolution/artifacts/text-corpus-v1
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.repair_views \
  --root benchmarks/super_resolution/artifacts/text-corpus-v1
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.ocr_protocol \
  --root benchmarks/super_resolution/artifacts/text-corpus-v1
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.repair_rotation \
  --root benchmarks/super_resolution/artifacts/text-corpus-v1
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.audit \
  --root benchmarks/super_resolution/artifacts/text-corpus-v1 \
  --index dataset-views-v3.json --output quality-audit-v3.json
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.degrade \
  --root benchmarks/super_resolution/artifacts/text-corpus-v1
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.degrade \
  --root benchmarks/super_resolution/artifacts/text-corpus-v1 \
  --index repair-views.json --stage hevc-repairs-v1
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.assemble \
  --root benchmarks/super_resolution/artifacts/text-corpus-v1
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.pack \
  --root benchmarks/super_resolution/artifacts/text-corpus-v1
PYTHONPATH=.:src .venv/bin/python -m benchmarks.super_resolution.text_corpus.train \
  --packed benchmarks/super_resolution/artifacts/text-corpus-v1/packed-v1/manifest.json \
  --output benchmarks/super_resolution/artifacts/text-scratch-quicksrnet-v1 \
  --device cuda:0 --epochs 100
```

These commands name distinct stages; do not restart an active stage. Fresh
catalogues, pack outputs and training outputs are required. Completed render
and encoded episode identities are verified before reuse; incomplete episodes
require inspection rather than silent retry. `render --limit 36` creates a
separate pilot index covering every content type. The final view index fills
rejected blank/duplicate layouts using an explicit supplementary plan.

## Initial training protocol

QuickSRNet-Small x2 has 22,860 parameters. The scratch constructor verifies and
builds the pinned architecture **without loading a pretrained checkpoint**.
It retains the official constructor's identity initialization together with
seeded random convolution parameters. Initial tensors and their fingerprint
are saved separately from every trained state.

The first fixed budget is 100 epochs, batch 32, fresh 96x96 LR / 192x192 HR
crops, Adam at 0.001 with cosine decay to 0.00001, FP32 with TF32 disabled.
Every training view is visited once per epoch. 80% of crops use source-known
word anchors (numbers/punctuation get 3x anchor probability); 20% are uniform.
The objective is normalized RGB L1 with a 3x weight inside text boxes, plus
0.05 times horizontal/vertical derivative L1. Validation uses two fixed crops
per held-out view. Minimum validation proxy loss selects `best.pt`; the test
split never participates in training or selection. This proxy must be followed
by source-truth OCR and image inspection, including digits and punctuation.

`initial.pt`, `best.pt`, ten-epoch snapshots, `last.pt`, protocol/source hashes,
host identity, per-epoch losses and completion receipts make the run reviewable.
A completion receipt means the fixed training budget finished; it does not mean
the model improves OCR, generalizes to all documents, or solves 128 Kbps links.

The frozen OCR protocol has 64 validation and 96 test English line regions,
selected from source geometry before training. Clean-reference OCR failures are
retained rather than filtering for easy cases. Its hash is bound into the
training protocol. After training, `evaluate render` generates identical crops
for bilinear, bicubic, official pretrained, scratch initialization and the
validation-proxy-selected scratch weights; `evaluate ocr` scores them locally
with the same Tesseract build and language model. These single-line PSM7 scores
are distinct from the earlier PSM6 multi-line benchmark.

## Hopper overnight campaign

`hopper submit` records six independent QuickSRNet-Small x2 scratch runs:
ordinary RGB L1 versus text-weighted RGB L1 plus edge loss, each at seeds
682, 683 and 684, with 300 epochs per run. Both losses retain the same text
anchor sampler, initialization rules, optimizer, LR schedule and training split.
All six use validation seed 682 and the common text/edge validation objective
to select checkpoints. Report all three seeds for each loss; test OCR must not
select a seed. The separate H100 100-epoch pilot has a different training budget.

A CPU Slurm preparation job verifies and packs the copied immutable corpus.
The six-task GPU array depends on its success and allows three concurrent A100
tasks. Each task gets four CPUs, 32 GiB host RAM and a 12-hour limit. Queued jobs
are distinct from running jobs. Contributed GPU resources can be preempted;
automatic requeue and implicit training resume are disabled. Failure logs and
partial outputs remain available for review.

`campaign.json` freezes all code, assets and source-index identities;
`submission.json` records actual Slurm IDs. Per-run logs, initial/trained weights,
source and host receipts, and separately named frozen OCR crops stay under the
isolated campaign root. OCR scoring uses the frozen local Tesseract after
downloading predictions; a completed GPU job alone does not establish OCR gain.

`collect_hopper` runs locally and follows the authorized SSH route through H100.
It copies completed runs, verifies their prediction/training bindings, scores
the frozen OCR cases, and reports all three seeds per loss. Failed Slurm tasks
are recorded; training is never resubmitted by this collector. Local collection
requires the Mac to keep running, while Slurm training runs independently.
Historical launch IDs, pilot results and file links are recorded in
`artifacts/text-corpus-v1/EXPERIMENT_STATUS.md`.

The October 21 runtime amendment explicitly migrates this campaign to two H100
GPUs after shared-scratch random reads caused long I/O stalls. Consistent L1
checkpoints at epochs 178/181/160 retain model, Adam, cosine scheduler and NumPy
crop-RNG state. `resume.py` verifies the original source/data policy and resumes
at the next epoch; its optimizer/scheduler/RNG recovery is tested against an
uninterrupted trajectory. Cross-GPU execution is not claimed bitwise identical.
The original Hopper jobs and checkpoint snapshots remain recorded; frozen data,
300-epoch budgets, seeds, losses and held-out selection rules are unchanged.
`h100_worker.py` runs two explicit queues against verified host-local packed
arrays; `collect_hopper --direct` collects the migrated H100 execution.

## Official weights and matched text training

The October 21 `official-transfer-v1` comparison uses the pinned official
QuickSRNet-Small x2 directly, official initialized text training, and a fresh
random initialized control. Both trained runs use seed 682, 100 epochs, the
same 7,288 training / 686 validation views, crop RNG, text/edge objective and
fixed validation selection. Official fine-tuning uses Adam LR 1e-4; scratch
uses 1e-3, both with cosine decay to 1% LR. This compares practical recipes;
the different learning rates prevent a pure initialization-only conclusion.

Only the allowed 256/600/1000/2000 Kbps subset is used. The frozen source-group
splits are retained, including 1,049 test views; 59 validation / 83 test English
line regions survive the new rate restriction. No line is replaced based on
OCR success. The earlier 300-epoch results have a different rate mixture and
remain historical comparisons, rather than substituting for the new control.

`transfer_campaign` freezes an isolated code tree and training plan, verifies
packed arrays, and runs an explicit GPU queue on `chen@lab`. `collect_transfer`
downloads completed weights and predictions, checks matching baseline pixels
and official initial tensor identity, and runs the unchanged local OCR engine.
It reports every eligible fixed test case and paired source bootstrap estimates.
`transfer_preview` applies the actual saved weights to the existing same-page
600 Kbps / 1 Mbps decoded HEVC frames; native pixel crops remain unsharpened.

The official checkpoint contains old optimizer metadata rejected by some newer
weights-only readers. `official.py` saves only its verified unchanged model
tensors using the existing compatible reader. The original checkpoint SHA256,
normalized file SHA256 and every model tensor's combined identity are bound and
rechecked on the training host. No unrestricted pickle reader or production
model-loader change is introduced. Tests verify preserved weights, actual
updates, tamper rejection, and rate filtering in every split.

Generated results and receipts live under
`artifacts/text-corpus-v1/official-transfer-v1/`. Completed training, OCR gain,
same-page visual appearance and live constrained-network usability are separate
claims; this experiment establishes only the first three when their artifacts
and checks have actually completed.
