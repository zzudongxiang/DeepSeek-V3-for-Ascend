#!/bin/bash
set -e
mkdir -p log

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

python pretrain.py                              \
    --ckpt-path ckpt/                           \
    --ckpt-saved-path ckpt-saved/               \
    --input-file scripts/shakespeare.txt        \
    --config configs/config_16B.json            \
    --use-random-weights                        \
    | tee log/pretrain-single-gpu.log
