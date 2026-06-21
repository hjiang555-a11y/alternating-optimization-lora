#!/bin/bash
# Finalize: multi-seed HellaSwag + MMLU + ARC + multi-seed C4
# Runs on GPU0. Requires proxy for C4 downloads.

set -e
cd /home/room115/alternating-optimization-lora
source .venv/bin/activate

export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# Proxy for C4
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
unset all_proxy ALL_PROXY

LOG_DIR=runs/qwen25_7b_800s/finalize_logs
mkdir -p "$LOG_DIR"

echo "============================================================" | tee "$LOG_DIR/run.log"
echo "FINALIZE: multi-seed downstream eval $(date)" | tee -a "$LOG_DIR/run.log"
echo "============================================================" | tee -a "$LOG_DIR/run.log"

# ── Step 0: Kill leftovers ──
pkill -f "_eval_downstream" 2>/dev/null || true
pkill -f "_eval_c4" 2>/dev/null || true
sleep 2

# ── Step 1: Remaining HellaSwag (B_s123, B_s456, D_s42, D_s123, D_s456) ──
echo "" | tee -a "$LOG_DIR/run.log"
echo "=== STEP 1: HellaSwag remaining 5 checkpoints $(date) ===" | tee -a "$LOG_DIR/run.log"

# Skip B_s42 (already done: acc=0.550, acc_norm=0.734)
python experiments/_eval_downstream.py \
    --protocols B --seeds 123,456 \
    --tasks hellaswag --num_fewshot 0 2>&1 | tee "$LOG_DIR/hellaswag_B123_B456.log"

python experiments/_eval_downstream.py \
    --protocols D --seeds 42,123,456 \
    --tasks hellaswag --num_fewshot 0 2>&1 | tee "$LOG_DIR/hellaswag_D_all.log"

echo "=== HellaSwag done $(date) ===" | tee -a "$LOG_DIR/run.log"

# ── Step 2: MMLU on B,D (seed 42, 5-shot, 200 samples per task) ──
echo "" | tee -a "$LOG_DIR/run.log"
echo "=== STEP 2: MMLU B+D s42 $(date) ===" | tee -a "$LOG_DIR/run.log"

python experiments/_eval_downstream.py \
    --protocols B,D --seeds 42 \
    --tasks mmlu --num_fewshot 5 --limit 200 2>&1 | tee "$LOG_DIR/mmlu.log"

echo "=== MMLU done $(date) ===" | tee -a "$LOG_DIR/run.log"

# ── Step 3: ARC-Challenge on B,D (seed 42, 0-shot) ──
echo "" | tee -a "$LOG_DIR/run.log"
echo "=== STEP 3: ARC-Challenge B+D s42 $(date) ===" | tee -a "$LOG_DIR/run.log"

python experiments/_eval_downstream.py \
    --protocols B,D --seeds 42 \
    --tasks arc_challenge --num_fewshot 0 2>&1 | tee "$LOG_DIR/arc.log"

echo "=== ARC done $(date) ===" | tee -a "$LOG_DIR/run.log"

# ── Step 4: C4 multi-seed (B+D × 3 seeds, 500 samples each) ──
echo "" | tee -a "$LOG_DIR/run.log"
echo "=== STEP 4: C4 multi-seed eval $(date) ===" | tee -a "$LOG_DIR/run.log"

for proto in B D; do
    for seed in 42 123 456; do
        echo "--- C4: P$proto s$seed $(date) ---" | tee -a "$LOG_DIR/run.log"
        python experiments/_eval_c4.py \
            --protocol "$proto" --seed "$seed" --n_eval 300 2>&1 | tee -a "$LOG_DIR/c4_${proto}_${seed}.log"
    done
done

echo "" | tee -a "$LOG_DIR/run.log"
echo "============================================================" | tee -a "$LOG_DIR/run.log"
echo "ALL DONE $(date)" | tee -a "$LOG_DIR/run.log"
echo "============================================================" | tee -a "$LOG_DIR/run.log"
