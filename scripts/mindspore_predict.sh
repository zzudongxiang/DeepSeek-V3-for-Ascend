#!/bin/bash
set -e
mkdir -p log

source /usr/local/Ascend/ascend-toolkit/set_env.sh

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
MASTER_PORT=8888
WORLD_SIZE=$((GPUS_PER_NODE*NNODES))

echo ----------------------------
echo MASTER_ADDR=${MASTER_ADDR}
echo MASTER_PORT=${MASTER_PORT}
echo NODE_RANK=${NODE_RANK}
echo NNODES=${NNODES}
echo ----------------------------

export PYTHONPATH=/home/ma-user/llm/mindformers/:$PYTHONPATH
export HCCL_OP_EXPANSION_MODE=AIV
export MS_ENABLE_LCCL=off

cd /home/ma-user/work/mindspore-dsv3/examples
bash msrun_launcher.sh          \
    "run_deepseekv3_predict.py" \
    $WORLD_SIZE                 \
    $WORLD_SIZE                 \
    $MASTER_ADDR                \
    $MASTER_PORT                \
    $NODE_RANK                  \
    output/msrun_log            \
    False                       \
    300
