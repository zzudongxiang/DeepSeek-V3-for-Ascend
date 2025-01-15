#!/bin/bash
set -e
mkdir -p log

python inference/generate.py                    \
    --ckpt-path ckpt/                           \
    --input-file scripts/inputs.txt             \
    --config inference/configs/config_671B.json \
    | tee log/single-gpu-inference.log
