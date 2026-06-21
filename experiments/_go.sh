#!/bin/bash
cd /home/room115/alternating-optimization-lora
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=1

pkill -f "python.*_param_matched" 2>/dev/null
pkill -f "python.*_eval_c4" 2>/dev/null
sleep 2

source .venv/bin/activate
python experiments/_param_matched_baseline.py
