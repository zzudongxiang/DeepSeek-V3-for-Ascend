set -e

source /usr/local/Ascend/ascend-toolkit/set_env.sh

OUT_PUT="log"
rm -rf ${OUT_PUT} && mkdir -p ${OUT_PUT}

ARGS="
    --ckpt-path ../ckpt/v3-int4-mp2     \
    --config configs/config_671B.json   \
    --model-name deepseek_pp            \
    --startup-type ./scripts/inputs.txt \
    --pp-layer-list 17,15,15,14         \
    --use_random_weights true           \
    --max-batch-size 4                  \
    --mini-batch-size 1                 \
    --max-new-tokens 10                 \
"

msprof --application="bash scripts/inference.sh ${ARGS}" \
    --output=${OUT_PUT} \
    --ascendcl=on \
    --model-execution=on \
    --runtime-api=on \
    --task-time=on \
    --aicpu=on \
    --hccl=on \
    --aicpu=on \
    --aic-mode=task-based \
    --aic-metrics=PipeUtilization \
    --host-sys=cpu,mem \
    --sys-hardware-mem=on \
    --sys-hardware-mem-freq=100 \
    --sys-cpu-profiling=on \
    --sys-cpu-freq=50 \
    --sys-profiling=on \
    --sys-sampling-freq=10 \
    --sys-interconnection-profiling=on \
    --sys-interconnection-freq=50 \
    --dvpp-profiling=on \
    --dvpp-freq=100 \
    --l2=on \
    --ai-core=on

# tar -zcvf log.tar.gz log/xxx
# msprof --export=on --output=log/xxx