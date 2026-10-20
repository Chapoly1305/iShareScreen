# Fine-tuning results

All three models completed 1,500 epochs across five content folds, totaling
2,344,500 optimizer updates. The [training protocol](TRAINING.md) defines the
shared data, recipe and checkpoint selection.

Scores use 683 held-out frames from 24 recordings: frames are averaged within
each recording, then recordings are weighted equally.

| Method | PSNR (dB), higher | SSIM, higher | LPIPS, lower |
|---|---:|---:|---:|
| Bicubic | 26.019 | 0.92118 | 0.14649 |
| EDSR pretrained | 28.738 | 0.94877 | 0.05493 |
| EDSR fine-tuned | 35.633 | 0.98501 | 0.01289 |
| IMDN pretrained | 28.597 | 0.94736 | 0.05031 |
| IMDN fine-tuned | 34.322 | 0.98241 | 0.01491 |
| SAFMN pretrained | 28.591 | 0.94766 | 0.05629 |
| SAFMN fine-tuned | 32.594 | 0.97509 | 0.02153 |

| Model | Selected epochs, folds 0-4 |
|---|---|
| EDSR | 1480, 1500, 1500, 1490, 1500 |
| IMDN | 1500, 1490, 1500, 1490, 1500 |
| SAFMN | 1000, 1500, 1500, 1500, 1500 |

Selection chooses each trajectory's highest fixed-subset PSNR; the last epoch is
not necessarily best. The comparison media instead use exact CURRENT epoch-1,500
weights to show the same budget across models.

## Local material

Generated material is ignored by Git and available in the original workspace:

- [Viewing gallery](artifacts/epoch-1500-comparison/training-comparison/index.html)
- [Media description](artifacts/epoch-1500-comparison/README.md)
- [Validation curves](artifacts/epochs-1500-v1-plots-v3/full_curves.png)
- [Checkpoint selection curves](artifacts/epochs-1500-v1-plots-v3/selection_curves.png)
- [Training L1 curves](artifacts/epochs-1500-v1-plots-v3/loss_curves.png)
- [Unrounded validation data](artifacts/epochs-1500-v1-report/curves.csv)
- [Unrounded epoch losses](artifacts/epochs-1500-v1-report/losses.csv)
- [Saved-artifact audit](artifacts/epochs-1500-v1-report/audit.json)
- [Trained checkpoints](artifacts/epochs-1500-v1-completed/)

The audit checked 2,344,500 update records, 546 aggregate validation observations
and 22,500 per-fold epoch-loss records, plus checkpoint identities, selection
and coverage. It reconstructed summaries from saved per-frame scores without
rerunning model inference or recomputing pixel metrics.

Between epochs 1,450 and 1,500, mean full-validation PSNR increased by 0.0327 dB
for EDSR, 0.0314 dB for IMDN and 0.0204 dB for SAFMN.

## Interpretation

These are development-validation scores, not independent-test performance.
Recorded content shares a capture environment; content folds do not establish
session independence. Curation was not compared with uniform sampling in a
separate controlled training experiment. References preserve decoded recording
pixels, not lossless acquisition.

This is a collection of five fold-specific checkpoints per model, not one final
all-data model. Quality scores and offline videos do not establish live latency,
FPS, temporal stability or text correctness. Live testing and a separately
recorded evaluation set remain separate next steps.
