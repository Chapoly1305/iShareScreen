# Epoch-based fine-tuning

## Data and decisions

EDSR-baseline, IMDN and SAFMN start independently from pinned official
pretrained x2 weights. They use the same frozen dataset, content folds and
training recipe. There is one maintained training scheme: true epochs over
the frozen curated data.

The current dataset contains 4,155 aligned frame pairs from 24 recordings:
683 clean baseline anchors and 3,472 curated additions. Twenty-one flagged
baseline frames are excluded. Recordings and their derivatives stay in their
assigned content fold; related recordings must share a group. Five-fold
cross-validation uses four folds for training and one for validation.

Training uses every eligible frame outside the held-out fold. Validation uses
only its clean baseline anchors; new frames in the held-out fold enter neither
side. This keeps validation membership fixed while adding training material.
Separate future evaluation recordings must not enter curation or checkpoint
selection. The current scores are development validation, not independent tests.

## Recipe

| Setting | Value |
|---|---|
| Initialization | Official pretrained weights, independently per model/fold |
| Optimizer / loss | Adam defaults / normalized RGB L1 |
| Learning rate | Constant 1e-5 |
| Batch / crop | Up to 32 paired crops; LR 64x64, HR 128x128 |
| Precision | FP32; training TF32 enabled, validation TF32 disabled |
| Sampling | Seeded shuffle without replacement each epoch; retain partial batch |
| Selection / full validation | Every 10 / 50 epochs, plus the final epoch |
| Selection rule | Highest fixed-subset equal-recording RGB PSNR, including epoch zero |
| Metrics | PSNR, SSIM, LPIPS |
| Recovery save | Every 100 updates and each epoch boundary |

Crops reduce memory and computation while preserving paired x2 coordinates.
Each visit draws a new random crop from a frozen full frame. An epoch means one
visit to every training frame, not enumeration of every possible patch.
All architectures receive the same frame order within a fold, not equal compute.

L1 is calculated between predicted and reference training patches. Validation
runs on full held-out frames without gradient updates. Its fixed small subset
(up to three anchors per recording) selects checkpoints; full validation describes
quality across the complete held-out inventory.

| Fold | Training frames | Validation frames | Updates/epoch |
|---|---:|---:|---:|
| 0 | 3,316 | 143 | 104 |
| 1 | 3,252 | 150 | 102 |
| 2 | 3,352 | 139 | 105 |
| 3 | 3,259 | 138 | 102 |
| 4 | 3,441 | 113 | 108 |

The completed campaign started with a 500-epoch ceiling and continued to 1,500.
All 15 trajectories total 2,344,500 optimizer updates. A ceiling is a budget,
not a convergence test. There is no automatic early stopping.

## Prepare frozen data

For new recordings, first run `prepare_video`, then assign content groups with
`folds` and extract baseline pairs with `training_data`. See each module's
`--help` and [curation](CURATION.md) for candidate extraction and quality review.
Freeze only a signed-off pool:

```sh
python -m benchmarks.super_resolution.freeze_training \
  --candidates /path/to/candidates.json --proposal /path/to/proposal.json \
  --pool /path/to/fold-additions.json --qc /path/to/audit.json \
  --folds /path/to/folds.json --baseline /path/to/baseline/manifest.json \
  --output /path/to/frozen

python -m benchmarks.super_resolution.pack_training \
  --pairs /path/to/frozen/manifest.json --output /path/to/packed
```

Freezing copies and verifies source pixels, geometry and group assignments.
Packing changes storage to read-only memory maps, not training pixels.
`restore_packed` verifies compressed transport copies before use.
Do not change data or folds in an existing run.

## Train and resume

One command owns one model/fold. It prints a plan unless `--execute` is added:

```sh
python -m benchmarks.super_resolution.train_epochs \
  --pairs /path/to/packed/manifest.json --folds /path/to/frozen/folds.json \
  --output /path/to/runs/imdn-fold-0 --model imdn --fold 0 --epochs 1500 \
  --batch 32 --patch 64 --lr 0.00001 --validate-every 10 --full-every 50 \
  --checkpoint-every 100 --selection-per-clip 3 --device cuda \
  --pixel-metrics cuda --train-tf32
```

Select GPU visibility explicitly. For Slurm, `slurm_train_epochs.sh` maps indices
0-4 to EDSR, 5-9 to IMDN and 10-14 to SAFMN. Its header defines required paths;
set `ISS_TRAIN_EPOCHS` to the desired ceiling. Scheduler defaults are site-specific.

Send SIGUSR1 to the verified worker PID for a safe pause (exit 75), then explicitly
resume with identical arguments plus `--resume --execute`. Use an output-specific
`flock` to prevent concurrent writers. A running validation finishes before pause;
do not infer a stall from a temporarily unchanged update counter.
Check `status.json`, `progress.jsonl` and stderr. Failed runs are not retried
automatically.

`last.pt` contains CURRENT weights, Adam, RNG, epoch and within-epoch cursor.
`best.pt` contains the selected weights. Periodic snapshots contain the actual
weights at each observation. Resume never combines earlier best weights with
later optimizer state.

To extend a completed budget, `extend_epochs` requires unchanged sources/data
and a fresh output. It preserves CURRENT weights, Adam, RNG and the earlier best,
changing only the cumulative ceiling. Resume and extension reject source drift.
Original-source checkpoints must use their original Git revision for further
training; the cleaned checkout can audit and render the completed epoch campaign
using its pinned historical source-map identity, without relabelling it as current.

## Report and select

```sh
python -m benchmarks.super_resolution.epoch_report \
  --run /path/to/completed-runs --folds /path/to/frozen/folds.json \
  --pairs /path/to/packed/manifest.json --output /path/to/fresh-report

python -m benchmarks.super_resolution.epoch_plots \
  --audit /path/to/fresh-report/audit.json --output /path/to/fresh-plots \
  --continuation-epoch 500
```

The report checks all model/fold outputs, source identities, frame membership,
per-frame scores, update counts and selected weights. It reconstructs summaries
from saved scores; it does not rerun metric inference.
Curves show CURRENT snapshots, not a running-best curve. Training-loss curves show
frame-weighted batch L1 per epoch. Selected epochs can differ by model and fold.
A five-fold model collection is not a single all-data deployment checkpoint.
