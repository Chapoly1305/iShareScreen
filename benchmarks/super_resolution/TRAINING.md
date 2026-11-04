# Completed CodeSCAN training and independent evaluation

Status reviewed 2026-11-03. The seven-model 100-epoch campaign and independent
1,200-sample test evaluation are complete. The canonical report is
[results/README.md](../../results/README.md); the [static presentation](../../results/index.html)
contains the primary test table and supplementary Mac validation examples.

## Data and training

CodeSCAN split-v1 is frozen at 9,600 train / 1,200 validation / 1,200 test
samples, grouped by source repository and duplicate relationships before
rendering. Training uses CodeSCAN train only. Mac recordings and CodeSCAN
validation/test samples are excluded from training. See [data curation](CURATION.md)
and the [dataset guide](../../datasets/README.md).

EDSR, QuickSRNet, IMDN, SAFMN, PlainUSR, RT4KSR and EfRLFN each completed 100
epochs from official pretrained weights with fresh optimizers. The implementation
is in `codescan_campaign/`; original configurations, checkpoints and curves
remain in the local experiment archive. Historical service files describe
those completed jobs and should not be enabled as a new training queue.

The codec recipe fits screenshots to 1920×1080, generates 60 frames with a
static first half and synthetic vertical translation in the second half, and
encodes complete frames at 720p / 600 Kbps / yuv420p. The aligned sample is frame
45. `codescan_campaign/data.py` implements the recipe. Black-and-white source
images are not the low-resolution inputs.

## Checkpoint selection

Mac validation covers 24 clips and 72 fixed frames. Highest equal-clip RGB PSNR
at epochs 0, 5, …, 100 selects a checkpoint. CodeSCAN validation crops provide
additional observations; the independent test does not select checkpoints.

| Model | Selected epoch |
| --- | ---: |
| EDSR | 25 |
| IMDN | 85 |
| SAFMN | 85 |
| EfRLFN | 100 |
| QuickSRNet | 100 |
| PlainUSR | 30 |
| RT4KSR | 95 |

The final weights and official initialization hashes were fixed before test
inference in [the test plan](../../results/evidence/test/PLAN.json). Training
curves show Mac validation observations, not repeated test evaluations.

## Independent test

`codescan_test.py` evaluates all 1,200 held-out screenshots using the frozen
codec recipe. It compares direct bicubic resizing and all seven models before
and after fine-tuning: 15 methods and 18,000 metric records. Complete 1080p
images are scored with RGB PSNR, RGB SSIM and LPIPS AlexNet v0.1. Model inference
uses FP32; no test score is used to tune parameters or select a checkpoint.

The runner checks protocol, source, input and weight hashes; split overlap;
codec bitrate/VBV conditions; and per-method sample coverage. Its `prepare`,
`evaluate` and `all` commands require a frozen plan and source manifests.
Use `python -m benchmarks.super_resolution.codescan_test --help` for arguments.
The GPU guard is bound to the recorded lab RTX 4090. Reproduction requires
private/local datasets and checkpoint assets; the static presentation does not.

Preserve completed metrics and protocols. A new experiment needs an independent
output directory and an explicit protocol; do not tune on the final test data.
Any subsequent development informed by these test scores requires a fresh
held-out test to support a new final generalization claim.

## Historical tools

`train_epochs.py`, `train_candidates_epochs.py`, `folds.py`, and the Slurm epoch
scripts belong to earlier recording-based or candidate campaigns. The earlier
4,155-pair / five-fold / 1,500-epoch procedure is not the recipe for these results.
Its documentation is retained in Git history and the external materials archive.
The original data and checkpoints remain unchanged.
