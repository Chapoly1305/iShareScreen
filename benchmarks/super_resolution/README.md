# Super-resolution research tools

The only published results are in [`../../results/`](../../results/README.md).
Open the [static presentation](../../results/index.html) for the seven-scene
before/after comparison. The presentation is self-contained and tracked with
its images, summary table, training curves and measurement evidence.

## Current evaluation

The completed campaign trained seven official pretrained ×2 models for 100 epochs
on frozen CodeSCAN training data. Evaluation uses 720p / 600 Kbps / yuv420p
inputs and final 1080p output. The report separates FP32 test quality from AMP FP16 forward timing on one
RTX 4090. The primary results cover 1,200 independent CodeSCAN test samples. Test
data are not used to select checkpoints or tune parameters. The Mac gallery
and training curves remain supplementary validation evidence. See the results report for the full protocol and limits.

## Source map

- `codescan_test.py`: frozen independent test preparation and 15-method evaluation.
- `codescan_campaign/`: frozen-pair preparation, training and monitoring.
- `seven_model_gallery.py`: reference, bicubic and pretrained/fine-tuned PNGs.
- `benchmark_best_latency.py`: repeated resident-forward timing on RTX 4090.
- `benchmark_bicubic_latency.py` and `bicubic_quality.py`: no-SR controls.
- `edsr_cuda/`: optional GPU-resident NVDEC/EDSR build, validation and timing
  tools; see its [scope and retained measurements](../../docs/analysis/edsr-cuda/README.md).
- `validation_baselines.py` and related modules: codec and quality evaluation.
- `text_corpus/`, `low_bandwidth_e2e/`, `mac_readability/`: retained research
  tooling, including historical approaches not used in the final presentation.

[TRAINING.md](TRAINING.md) describes the completed campaign and fixed test
protocol; [CURATION.md](CURATION.md) describes split isolation.
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
