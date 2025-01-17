#!/bin/bash
set -e
mkdir -p log

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# --profiling
python inference/generate.py                    \
    --ckpt-path ckpt/                           \
    --input-file scripts/inputs.txt             \
    --config inference/configs/config_16B.json  \
    --use-random-weights                        \
    | tee log/single-gpu-inference.log
