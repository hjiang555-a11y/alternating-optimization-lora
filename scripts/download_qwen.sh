#!/bin/bash
source /home/room115/pytorch_env/bin/activate
hf download Qwen/Qwen2.5-7B 2>&1
echo "Download completed"