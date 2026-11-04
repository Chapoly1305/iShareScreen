# Data isolation for the final campaign

The current campaign uses frozen CodeSCAN split-v1. See [training and test
protocol](TRAINING.md) and the [dataset guide](../../datasets/README.md). The
canonical results are in [results/](../../results/README.md).

## Source grouping and freeze

The 12,000 source screenshots are split into 9,600 train, 1,200 validation and
1,200 test samples. Each sample's seven files remain together. Repository
identity and duplicate relationships are grouped before split assignment;
source and directory hashes are audited. Original ZIP contents are preserved.

`datasets/split_codescan.py` provides `inventory`, `split`, and `verify` stages.
The existing split is frozen: use `verify` for auditing; do not overwrite it
or add future samples to the same version. Exact hashes do not prove absence
of all semantically similar code fragments.

## Codec-derived pairs

`codescan_campaign/data.py` prepares train/validation pairs only. The independent
`codescan_test.py` uses the same rendering and encoding functions for test
preparation in a separate output directory. Neither the test samples nor Mac
validation enter the training loader.

The test evaluator checks repository/group and source-hash overlap, then checks
generated LR/HR file and pixel hashes against train/validation pairs. It also
checks within-test uniqueness. All 1,200 test inputs passed the checks before
the full evaluation. [The published audit](../../results/evidence/test/AUDIT.json)
and [frozen plan](../../results/evidence/test/PLAN.json) bind the results.

## Mac and DVR material

Mac recordings remain development validation for the final CodeSCAN campaign.
The seven-scene gallery is a subset of that validation set. Older campaigns
used some Mac material for training, so the same recordings must not be called
unseen data for those historical models.

DVR excerpts are separate candidate material. Selecting a usable clip does not
admit it to train, validation or test. Future assignments must group source
sessions and shared document/project content before extracting frames, and
check overlap against existing splits.

## Historical curation tooling

`prepare_curation`, `curate`, `audit_curation`, `review_curation`, and
`freeze_training` implement the earlier video-frame/FiftyOne curation workflow.
The old 683-anchor and 4,155-pair inventories are historical data, not the final
CodeSCAN split. Their original procedures remain in Git history and the external
materials archive; the frozen inputs and audit receipts remain unchanged.
