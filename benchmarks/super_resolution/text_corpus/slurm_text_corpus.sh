#!/usr/bin/env bash
# Resources/dependencies are recorded by hopper.submit; no automatic requeue.
set -euo pipefail
: "${ISS_TEXT_ROOT:?Isolated campaign root required}"
: "${ISS_TEXT_PYTHON:?Verified compatible Python required}"
: "${ISS_TEXT_STAGE:?pack or train required}"
export PYTHONPATH="$ISS_TEXT_ROOT/code:$ISS_TEXT_ROOT/code/src"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$ISS_TEXT_ROOT/code"
text_lock="$ISS_TEXT_ROOT/$ISS_TEXT_STAGE-${SLURM_ARRAY_TASK_ID:-single}.lock"
flock --nonblock --no-fork "$text_lock" "$ISS_TEXT_PYTHON" -u \
    -m benchmarks.super_resolution.text_corpus.hopper worker \
    --root "$ISS_TEXT_ROOT" --python "$ISS_TEXT_PYTHON" --stage "$ISS_TEXT_STAGE"
