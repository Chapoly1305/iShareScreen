# Training-frame curation

Use denser paired samples from existing desktop recordings, remove redundant
frames and review questionable content before freezing training data.
Curation does not launch training or demonstrate improved model quality.

## Environment

FiftyOne Brain provides candidate near-duplicate selection; native-resolution
HR/LR checks protect fine desktop details. Keep it separate from the viewer
and training environment:

```sh
python3.11 -m venv benchmarks/super_resolution/.venv-curation
benchmarks/super_resolution/.venv-curation/bin/python -m pip install \
  -r benchmarks/super_resolution/requirements-curation.txt
```

Processing is local, with telemetry disabled and a run-specific MongoDB directory.
Only package installation and explicit `--download-weights` fetch public assets;
recordings and features are not sent to an inference service.

## Extract, select and audit

Start with complete reference/HEVC pairs from `prepare_video`, a content-group
split from `folds`, and baseline PNG pairs from `training_data`:

```sh
python -m benchmarks.super_resolution.folds \
  --manifest /path/to/video-pairs/manifest.json --output /path/to/folds.json

python -m benchmarks.super_resolution.training_data \
  --manifest /path/to/video-pairs/manifest.json --folds /path/to/folds.json \
  --output /path/to/baseline --every 60

python -m benchmarks.super_resolution.prepare_curation \
  --manifest /path/to/video-pairs/manifest.json --folds /path/to/folds.json \
  --baseline /path/to/baseline/manifest.json --output /path/to/candidates \
  --every 10 --workers 4

benchmarks/super_resolution/.venv-curation/bin/python \
  -m benchmarks.super_resolution.curate \
  --candidates /path/to/candidates/candidates.json --folds /path/to/folds.json \
  --output /path/to/review --device mps --download-weights

python -m benchmarks.super_resolution.audit_curation \
  --review /path/to/review --folds /path/to/folds.json \
  --output /path/to/qc --workers 4
```

Use fresh output directories. Select `cuda`, `mps` or `cpu` explicitly.
All existing baseline indices are included even when off the denser schedule.
Candidate IDs are zero-based source-frame indices. Extraction checks complete
frame/timestamp alignment, source hashes, x2 geometry and paired decoded pixels.
Every derivative inherits its recording's group and fold.

The selector:

1. Embeds whole frames at 224x224 using pretrained MobileNetV3-Small.
2. Asks FiftyOne Brain for within-capture near-duplicate candidates at distance
   0.005. Independently checks cosine distance in the same centered feature space;
   the pinned Brain version's representative-only neighbor map can recenter data.
3. Resolves exact paired duplicates using both HR and LR pixel hashes, preferring
   an existing baseline anchor, otherwise the earliest frame.
4. Checks suggested duplicates at native resolution on both HR and LR. Mean
   absolute RGB difference must be at most 0.25 intensity units and the maximum
   16x16 tile mean at most 1.0, including partial edge tiles.
5. Preserves baseline anchors and quarantines new references shared across
   content groups rather than introducing cross-fold duplicates.

These conservative thresholds are practical guards, not universal perceptual
criteria. Embeddings can miss tiny text changes; the reviewed policy below adds
an exact-HR guard. Different LR degradation is meaningful even with identical HR.

## Review and freeze

Review `index.html`, coverage sheets and native PNGs. The proposal retains every
decision, representative and source identity; the independent audit rechecks
coverage, fold membership, pairing, hashes and confirmed-duplicate pixel guards.

Generate detail images and optional timelines for suspicious captures:

```sh
python -m benchmarks.super_resolution.curation_views \
  --review /path/to/review --output /path/to/details

python -m benchmarks.super_resolution.curation_views \
  --review /path/to/review --output /path/to/timelines \
  --timeline-captures capture-a capture-b

python -m benchmarks.super_resolution.review_curation \
  --review /path/to/review --audit /path/to/qc/audit.json \
  --quality-review /path/to/quality-review.json --output /path/to/final-review

python -m benchmarks.super_resolution.audit_curation \
  --review /path/to/final-review --folds /path/to/folds.json \
  --output /path/to/final-qc --workers 4
```

The dataset-specific quality policy binds `candidates_sha256`, lists inclusive
`capture`, `start`, `end`, `reason` ranges, and enables
`require_exact_hr_for_near_dedup: true`. Any changed HR pixel then restores a
suggested near duplicate. Reject capture overlays, transitions and unsuitable
content by reviewed ranges rather than generic embedding distance.

The review writes `proposal.json` and `fold-additions.json`; these are not
direct training manifests. Flagged baseline frames stay in the historical
inventory but are excluded when `freeze_training` builds the current clean
dataset. [Training](TRAINING.md) defines the final 683-frame validation boundary.

The accepted local inputs are `curation-review-v4` and `curation-qc-v4`.
Do not edit a frozen dataset or resume a run with changed membership.
New independent evaluation recordings remain outside this process.

## Checks

`tests/test_super_resolution_curation.py` covers pairing, baseline retention,
small/edge-tile changes, fold leakage, representative survival and quality policies.
The real FiftyOne API smoke test needs the optional curation environment.
Visual spot checks do not prove that every retained frame is useful, and denser
sampling can still miss brief events. No extra uniform-versus-curated training
comparison is required or claimed.

Components: [FiftyOne](https://github.com/voxel51/fiftyone),
[pinned Brain backend](https://github.com/voxel51/fiftyone-brain/blob/v0.25.0/fiftyone/brain/internal/core/sklearn.py),
[torchvision MobileNetV3-Small](https://docs.pytorch.org/vision/0.21/models/generated/torchvision.models.mobilenet_v3_small.html).
