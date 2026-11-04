# Development hosts and local archives

Reviewed 2026-11-03. H100 `/home/jchen73/iShareScreen` is the development checkout.
The canonical published results are [results/](../results/README.md). Host paths
below are local research infrastructure and are not needed for static hosting.
GPU permissions and occupancy checks are defined by `AGENTS.md`.

## Archive and compatibility paths

- Original artifacts: `/home/jchen73/iShareScreen-archives/before-retrain-2026-10-29_192818/`.
- Completed experiments and superseded result pages: `/home/jchen73/iShareScreen-archives/results-cleanup-20261103/`.
- Pre-review documentation: `/home/jchen73/iShareScreen-archives/materials-review-20261103/`.
- H100 `benchmarks/super_resolution/artifacts` and `experiments` are ignored
  compatibility links into external archives. Preserve frozen manifests and
  source bindings; create a separate output for new work.

These are moves on the same machine, not independent backups. The original Mac
MOV files were deleted by the user; processed RGB8 references remain available.

## Completed independent test

The test ran on `chen@lab` (`/home/chen/iShareScreen`), RTX 4090 UUID
`GPU-f3d8b210-4eb5-a38b-a81d-983611f83a6a`. All 15 methods completed 1,200 samples.
The remote output is `benchmarks/super_resolution/experiments/codescan-test-720p-600k-420-v1/`.
Metrics and protocol were collected to H100
`datasets/derived/codescan-test-720p-600k-420-v1/`; published evidence is under
`results/evidence/test/`. The completed systemd unit was
`isharescreen-codescan-test-v1.service`; it is not an active evaluation queue.

The earlier 4:2:0/4:4:4 resolution/bitrate sweep is historical. New evaluation
defaults to 4:2:0. Use the final report for quality and timing comparisons.

## Synchronizing future work

Sync explicit code and input files. Exclude host `.venv` directories, private
machine settings, caches and active experiment outputs. Do not broadly overwrite
remote projects or assume their dataset links match H100. Validate copied input
and checkpoint hashes before running. Keep private SSH keys on their own hosts.

Activate each host's project-local environment with `source dev-environment.sh`.
Recorded package versions describe the completed run; inspect the actual runtime
for any new work. Existing public-key authorization does not imply GPU idleness.
