#!/usr/bin/env bash
# One independent lightweight-candidate/fold trajectory per Slurm GPU.
# Submit only unstarted indices; each output has an exclusive flock.
#SBATCH --job-name=sr-candidates
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
: "${ISS_CANDIDATE_ROOT:?Set campaign root}"
: "${ISS_CANDIDATE_PYTHON:?Set compatible Python executable}"
: "${SLURM_ARRAY_TASK_ID:?Submit explicit candidate/fold indices}"
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID > 29 )); then exit 2; fi
if [[ -n "${ISS_CANDIDATE_MODULES:-}" ]]; then
    read -r -a candidate_modules <<< "$ISS_CANDIDATE_MODULES"
    module load "${candidate_modules[@]}"
fi
candidate_models=(plainusr quicksrnet rt4ksr efrlfn casr spanv2)
candidate_model="${candidate_models[$((SLURM_ARRAY_TASK_ID / 5))]}"
candidate_fold=$((SLURM_ARRAY_TASK_ID % 5))
candidate_output="$ISS_CANDIDATE_ROOT/runs/$candidate_model-fold-$candidate_fold"
export PYTHONPATH="$ISS_CANDIDATE_ROOT/code/src:$ISS_CANDIDATE_ROOT/code"
export TORCH_HOME="${ISS_CANDIDATE_TORCH_HOME:-$ISS_CANDIDATE_ROOT/torch-cache}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
cd "$ISS_CANDIDATE_ROOT/code"
candidate_args=()
if [[ "${ISS_CANDIDATE_RESUME:-0}" == 1 ]]; then candidate_args+=(--resume); fi
candidate_pid=""
request_pause() {
    if [[ -n "$candidate_pid" ]]; then kill -USR1 "$candidate_pid" 2>/dev/null || true; fi
}
trap request_pause USR1 TERM
flock --nonblock --no-fork "$candidate_output.worker.lock" \
    "$ISS_CANDIDATE_PYTHON" -u -m benchmarks.super_resolution.train_candidates_epochs \
    --model "$candidate_model" --fold "$candidate_fold" --output "$candidate_output" \
    --home "$ISS_CANDIDATE_ROOT/assets" --folds "$ISS_CANDIDATE_ROOT/data/folds.json" \
    --pairs "$ISS_CANDIDATE_ROOT/packed/manifest.json" --epochs 1500 --batch 32 --patch 64 \
    --lr 0.00001 --validate-every 10 --full-every 50 --checkpoint-every 100 \
    --selection-per-clip 3 --threads "$OMP_NUM_THREADS" --device cuda --pixel-metrics cuda \
    --train-tf32 --execute "${candidate_args[@]}" &
candidate_pid=$!
set +e
wait "$candidate_pid"
candidate_exit=$?
set -e
printf 'WORKER_EXIT_CODE=%s\n' "$candidate_exit"
exit "$candidate_exit"
