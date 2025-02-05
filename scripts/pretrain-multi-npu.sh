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

# # 设置HCCL和ACL的相关参数（仅适用于调试场景抓取日志）
# export HCCL_EXEC_TIMEOU=7200
# export HCCL_CONNECT_TIMEOUT=3600
# export HCCL_ENTRY_LOG_ENABLE=1
# export ASCEND_GLOBAL_LOG_LEVEL=1
# export ASCEND_SLOG_PRINT_TO_STDOUT=1

# # 打印详细的日志（仅适用于调试场景抓取日志）
# ma_vj_name=`echo ${MA_VJ_NAME} | sed 's:ma-job:modelarts-job:g'`
# task_name="worker-${VC_TASK_INDEX}"
# task_plog_path=${MA_LOG_DIR}/${ma_vj_name}/${task_name}
# mkdir -p ${task_plog_path}
# export ASCEND_PROCESS_LOG_PATH=${task_plog_path}
# echo "plog path: ${ASCEND_PROCESS_LOG_PATH}"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES                \
    --node_rank $NODE_RANK          \
    --master_addr $MASTER_ADDR      \
    --master_port $MASTER_PORT
"

torchrun $DISTRIBUTED_ARGS pretrain.py              \
    --ckpt-path ../ckpt-ep256-mp64/                 \
    --ckpt-saved-path ckpt-saved/                   \
    --input-file scripts/shakespeare.txt            \
    --config configs/config_671B.json               \
    | tee log/pretrain-multi-npu.log
