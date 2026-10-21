# Fine-tuning results

Four lightweight candidates with official native ×2 pretrained weights completed
1,500 epochs across five grouped content folds. Scores below average the five
held-out folds. `Before` is the epoch-zero full validation; `After` is the
selected checkpoint's full validation from the same fold.

| Method | PSNR (dB), higher | SSIM, higher | LPIPS, lower |
|---|---:|---:|---:|
| PlainUSR, before → after | 27.759 → 29.205 | 0.9408 → 0.9549 | 0.0740 → 0.0525 |
| QuickSRNet, before → after | 28.049 → **30.044** | 0.9439 → **0.9629** | 0.0671 → **0.0354** |
| RT4KSR, before → after | 27.087 → 27.436 | 0.9314 → 0.9370 | 0.0884 → 0.0822 |
| EfRLFN, before → after | 25.625 → 27.098 | 0.9122 → 0.9318 | 0.1577 → 0.0858 |

CASR and SPANV2 are evaluated separately: their public weights are native ×4,
so their deterministic ×4-to-×2 initialization is not presented as a
pretrained ×2 baseline.

## Local material

Generated material is ignored by Git and remains in the local workspace:

- [Aggregate curves and fold-level data](artifacts/candidate-training-1500-report/)
- [Unified candidate-model comparison](artifacts/candidate-pretrained-vs-finetuned-v1/comparison-v2/index.html)
- [Original-model comparison gallery](artifacts/epoch-1500-comparison/training-comparison/index.html)
- [Cross-device latency measurements](artifacts/cross-device-video-latency/)

The unified gallery preserves the original code, terminal and desktop crops and
adds selected light-theme interfaces, web content, charts, document text and
video texture as still-image comparisons. Each model stays in a fixed column,
with pretrained above fine-tuned; Original/Bicubic and the unenhanced 540p input
provide shared references. Fine-tuned panels use the selected checkpoint from
the completed 1,500-epoch held-out fold trajectory.

- [All recording previews and scene review](artifacts/recording-midpoint-review/index.html)

## Interpretation

These are development-validation scores, not independent-test performance.
The data use grouped content folds but share a capture environment. Quality
scores and offline videos do not establish live latency, FPS, temporal
stability or text correctness. Live testing and a separately recorded
evaluation set remain separate next steps.
