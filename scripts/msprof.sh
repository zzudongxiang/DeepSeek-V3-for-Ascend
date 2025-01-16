set -e

OUT_PUT="log"
rm -rf ${OUT_PUT} && mkdir -p ${OUT_PUT}
msprof --application="bash scripts/single-gpu-inference.sh" \
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
