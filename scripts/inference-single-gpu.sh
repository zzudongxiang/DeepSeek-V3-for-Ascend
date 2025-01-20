#!/bin/bash
set -e
mkdir -p log

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

python inference.py                             \
    --ckpt-path ckpt/                           \
    --input-file scripts/inputs.txt             \
    --config configs/config_16B.json            \
    --use-random-weights                        \
    | tee log/inference-single-gpu.log
