#!/bin/bash
# P1 Sequential Task Runner
# 1. Kill zombie GPU process, clean memory
# 2. Run C4 perplexity evaluation
# 3. Run MMLU + ARC downstream evaluation

set -e
cd /home/room115/alternating-optimization-lora
source .venv/bin/activate
export HF_HUB_OFFLINE=1

echo "=== STEP 0: Clean GPU memory ==="
# Kill known zombie (PID file from previous runs)
fuser -k /dev/nvidia0 2>/dev/null || true
sleep 3
nvidia-smi

echo "=== STEP 1: C4 Perplexity Evaluation ==="
echo "Note: C4 dataset may need download. Will try local cache first."
python experiments/_eval_c4.py 2>&1 | tee runs/c4_eval.log

echo ""
echo "=== STEP 2: Run full HellaSwag on remaining checkpoints ==="
python experiments/_eval_downstream.py --protocols D --seeds 42 --tasks hellaswag 2>&1 | tee -a runs/qwen25_7b_800s/hellaswag_extra.log

echo ""
echo "=== STEP 3: MMLU evaluation on B,D checkpoints (seed 42) ==="
python experiments/_eval_downstream.py --protocols B,D --seeds 42 --tasks mmlu --num_fewshot 5 --limit 200 2>&1 | tee runs/qwen25_7b_800s/mmlu_eval.log

echo ""
echo "=== ALL P1 DONE ==="
