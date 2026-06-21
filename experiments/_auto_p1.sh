#!/bin/bash
# Auto-launch P1 experiments
set -e
cd /home/room115/alternating-optimization-lora

# Kill any leftover python experiments
pkill -f "_param_matched_baseline" 2>/dev/null || true
pkill -f "_eval_c4" 2>/dev/null || true
pkill -f "_eval_downstream" 2>/dev/null || true
sleep 2

source .venv/bin/activate
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

echo "=== $(date): Launching parameter-matched baseline ==="
mkdir -p runs/param_matched_baseline
python experiments/_param_matched_baseline.py > runs/param_matched_baseline/run.log 2>&1

echo "=== $(date): Parameter-matched baseline done ==="
echo "=== $(date): Launching C4 evaluation ==="
python experiments/_eval_c4.py > runs/qwen25_7b_800s/c4_eval.log 2>&1

echo "=== ALL P1 DONE $(date) ==="
