#!/usr/bin/env bash
# One independent model/fold trajectory per allocated GPU, fresh or resumed.
#SBATCH --job-name=sr-epochs
#SBATCH --partition=contrib-gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@180
#SBATCH --no-requeue
set -euo pipefail
: "${ISS_TRAIN_ROOT:?Set new campaign root}"
: "${ISS_TRAIN_PYTHON:?Set existing compatible Python executable}"
: "${ISS_TRAIN_HOME:?Set verified official assets directory}"
: "${ISS_TRAIN_TORCH_HOME:?Set cached LPIPS backbone directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit explicit model/fold schedule indices}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 14 )); then exit 2; fi
if [[ -n "${ISS_TRAIN_MODULES:-}" ]]; then
    read -r -a training_modules <<< "$ISS_TRAIN_MODULES"
    module load "${training_modules[@]}"
fi
# Catch site-environment failures before starting a Python training worker.
command -v git >/dev/null
command -v flock >/dev/null
"$ISS_TRAIN_PYTHON" --version
training_models=(edsr imdn safmn)
training_model="${training_models[$((SLURM_ARRAY_TASK_ID / 5))]}"
training_fold=$((SLURM_ARRAY_TASK_ID % 5))
training_runs="${ISS_TRAIN_RUNS:-$ISS_TRAIN_ROOT/runs}"
training_output="$training_runs/$training_model-fold-$training_fold"
mkdir -p "$training_runs"
export PYTHONPATH="$ISS_TRAIN_ROOT/code/src:$ISS_TRAIN_ROOT/code"
export TORCH_HOME="$ISS_TRAIN_TORCH_HOME"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$ISS_TRAIN_ROOT/code"
training_args=()
if [[ "${ISS_TRAIN_RESUME:-0}" == 1 ]]; then training_args+=(--resume); fi
training_pid=""
request_pause() {
    if [[ -n "$training_pid" ]]; then kill -USR1 "$training_pid" 2>/dev/null || true; fi
}
trap request_pause USR1 TERM
flock --nonblock --no-fork "$training_output.worker.lock" \
    "$ISS_TRAIN_PYTHON" -u -m benchmarks.super_resolution.train_epochs \
    --model "$training_model" --fold "$training_fold" --output "$training_output" \
    --home "$ISS_TRAIN_HOME" --folds "$ISS_TRAIN_ROOT/data/folds.json" \
    --pairs "$ISS_TRAIN_ROOT/packed/manifest.json" --epochs "${ISS_TRAIN_EPOCHS:-1500}" --batch 32 --patch 64 \
    --lr 0.00001 --validate-every 10 --full-every 50 --checkpoint-every 100 \
    --selection-per-clip 3 --threads "$OMP_NUM_THREADS" --device cuda --pixel-metrics cuda \
    --train-tf32 --execute "${training_args[@]}" &
training_pid=$!
set +e
wait "$training_pid"
training_exit=$?
if kill -0 "$training_pid" 2>/dev/null; then
    wait "$training_pid"
    training_exit=$?
fi
set -e
printf 'WORKER_EXIT_CODE=%s\n' "$training_exit"
exit "$training_exit"  # No automatic retry/requeue, including after a safe pause.
