#!/usr/bin/env bash
# One exclusive, fixed-budget scratch experiment. No retries or early stopping.
set -uo pipefail
text_task_root="$1"
text_python="$2"
export PYTHONPATH="$text_task_root/code:$text_task_root/code/src"
export CUDA_VISIBLE_DEVICES=GPU-3bf9ecb9-5729-7ad7-1a71-197d4c00415e
export OMP_NUM_THREADS=4
cd "$text_task_root/code" || exit 2
"$text_python" -u -m benchmarks.super_resolution.text_corpus.pack \
    --root "$text_task_root/data" > "$text_task_root/pack.log" 2>&1
text_exit=$?
printf '%s\n' "$text_exit" > "$text_task_root/pack.exit"
if [[ "$text_exit" != 0 ]]; then exit "$text_exit"; fi
"$text_python" -u -m benchmarks.super_resolution.text_corpus.train \
    --packed "$text_task_root/data/packed-v1/manifest.json" \
    --output "$text_task_root/runs/quicksrnet-scratch-v1" \
    --home "$text_task_root/assets" --device cuda:0 --epochs 100 \
    --batch 32 --patch 96 --lr 0.001 --seed 682 \
    > "$text_task_root/train.log" 2>&1
text_exit=$?
printf '%s\n' "$text_exit" > "$text_task_root/train.exit"
if [[ "$text_exit" != 0 ]]; then exit "$text_exit"; fi
"$text_python" -u -m benchmarks.super_resolution.text_corpus.evaluate render \
    --root "$text_task_root/data" --training "$text_task_root/runs/quicksrnet-scratch-v1" \
    --home "$text_task_root/assets" --device cuda:0 \
    > "$text_task_root/evaluation-render.log" 2>&1
text_exit=$?
printf '%s\n' "$text_exit" > "$text_task_root/evaluation-render.exit"
exit "$text_exit"
