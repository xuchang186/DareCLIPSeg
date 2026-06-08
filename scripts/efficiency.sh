#!/bin/bash
set -e

OUTPUT=$1
DATASET=$2
DATA_PERCENTAGE=$3
SEED=$4

if [ -z "${OUTPUT}" ] || [ -z "${DATASET}" ] || [ -z "${DATA_PERCENTAGE}" ] || [ -z "${SEED}" ]; then
    echo "Usage: bash efficiency.sh <OUTPUT> <DATASET> <DATA_PERCENTAGE> <SEED>"
    echo "Example: bash efficiency.sh /root/autodl-tmp/xuchang/DareCLIPSeg/output BTMRI 100 24"
    exit 1
fi

gpu_id=${GPU_ID:-1}
export CUDA_VISIBLE_DEVICES=${gpu_id}
echo "=================================="
echo "Using GPU ${gpu_id}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "=================================="

CONFIG=${DATASET}

if [ "${DATA_PERCENTAGE}" = "100" ]; then
    OUTPUT_DATASET=${DATASET}
else
    OUTPUT_DATASET="${DATASET}_${DATA_PERCENTAGE}"
fi

BASE_RUN_NAME="seed${SEED}"
RUN_NAME="${BASE_RUN_NAME}"
RUN_ROOT="${OUTPUT}/${OUTPUT_DATASET}"

idx=2
while [ -d "${RUN_ROOT}/trained_models/${RUN_NAME}" ] || \
      [ -d "${RUN_ROOT}/seg_results/${RUN_NAME}" ] || \
      [ -d "${RUN_ROOT}/unc_results/${RUN_NAME}" ]; do
    RUN_NAME="${BASE_RUN_NAME}(${idx})"
    idx=$((idx + 1))
done

echo "Processing dataset: ${DATASET} using config: ${CONFIG}.yaml with data_percentage=${DATA_PERCENTAGE}"
echo "Seed: ${SEED}"
echo "Run name: ${RUN_NAME}"
echo "Output dataset directory: ${RUN_ROOT}"

python train.py --config-file configs/${CONFIG}.yaml \
    --output-dir "${OUTPUT}" \
    --data_percentage "${DATA_PERCENTAGE}" \
    --seed "${SEED}" \
    --run_name "${RUN_NAME}"

python test.py --config-file configs/${CONFIG}.yaml \
    --output-dir "${OUTPUT}" \
    --source_dataset "${DATASET}" \
    --data_percentage "${DATA_PERCENTAGE}" \
    --seed "${SEED}" \
    --run_name "${RUN_NAME}"

python utils/eval.py --config-file configs/${CONFIG}.yaml \
    --output-dir "${OUTPUT}" \
    --data_percentage "${DATA_PERCENTAGE}" \
    --seed "${SEED}" \
    --run_name "${RUN_NAME}"
