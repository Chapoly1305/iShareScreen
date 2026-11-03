# Super-resolution research tools

The only published results are in [`../../results/`](../../results/README.md).
Open the [static presentation](../../results/index.html) for the seven-scene
before/after comparison. The presentation is self-contained and tracked with
its images, summary table, training curves and measurement evidence.

## Current evaluation

The final campaign trains seven official pretrained ×2 models for 100 epochs
on frozen CodeSCAN training data. Evaluation uses 720p / 600 Kbps / yuv420p
inputs and final 1080p output. The report explicitly separates FP32 validation
quality from AMP FP16 forward timing on one RTX 4090. Test data are not used to
select checkpoints. See the results report for the full protocol and limits.

## Source map

- `codescan_campaign/`: frozen-pair preparation, training and monitoring.
- `seven_model_gallery.py`: reference, bicubic and pretrained/fine-tuned PNGs.
- `benchmark_best_latency.py`: repeated resident-forward timing on RTX 4090.
- `benchmark_bicubic_latency.py` and `bicubic_quality.py`: no-SR controls.
- `validation_baselines.py` and related modules: codec and quality evaluation.
- `text_corpus/`, `low_bandwidth_e2e/`, `mac_readability/`: retained research
  tooling, including historical approaches not used in the final presentation.

Older training and curation procedures remain in [TRAINING.md](TRAINING.md)
and [CURATION.md](CURATION.md) as technical documentation, not current results.
The [development notes](DEVELOPMENT.md) describe regression checks.

## Local research artifacts

Experimental outputs, checkpoints and datasets are not published with the code.
On the development machine, historical outputs have been archived outside the
repository. The ignored `experiments` and `artifacts` links preserve existing
local paths; they are not part of the static site or required by it. Do not
rewrite frozen manifests or hashes. New experiments need fresh directories.

Activate the project environment with `source dev-environment.sh` on the
configured development host. Follow `AGENTS.md` before using GPU resources.
No model download or GPU is required to view the final presentation.
