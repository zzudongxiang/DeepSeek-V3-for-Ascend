#!/bin/bash
set -e
mkdir -p log

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

# read modelarts env
MASTER_ADDR="${VC_WORKER_HOSTS%%,*}"
NODE_RANK="$VC_TASK_INDEX"
NNODES="$MA_NUM_HOSTS"

# if not in modelarts
MASTER_ADDR=${MASTER_ADDR:="localhost"}
NODE_RANK=${NODE_RANK:="0"}
NNODES=${NNODES:="1"}

# others var
GPUS_PER_NODE=8
MASTER_PORT=6000
WORLD_SIZE=$((GPUS_PER_NODE*NNODES))

echo ----------------------------
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo NODE_RANK=${NODE_RANK}
echo NNODES=${NNODES}
echo ----------------------------

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES                \
    --node_rank $NODE_RANK          \
    --master_addr $MASTER_ADDR      \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS pretrain.py              \
    --ckpt-path ../ckpt-ep256-mp32/                 \
    --ckpt-saved-path ckpt-saved/                   \
    --input-file scripts/shakespeare.txt            \
    --config configs/config_671B.json               \
    | tee log/pretrain-multi-npu.log
