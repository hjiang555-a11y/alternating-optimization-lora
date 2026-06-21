#!/usr/bin/env bash
cd /home/room115/alternating-optimization-lora
exec env -i HOME="$HOME" PATH="$PATH" SHELL="$SHELL" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/python experiments/_eval_full_test.py 2>&1 | tee /tmp/eval_full_test.log
