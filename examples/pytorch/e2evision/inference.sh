#!/bin/bash
set -euo pipefail  # 严格的错误处理模式

# 创建日志和结果目录
LOG_DIR="logs"
RESULTS_DIR="results"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

# 推理配置
TEST_LIST="test_list.txt"
BATCH_SIZE=1
NUM_WORKERS=4
ACCELERATOR="gpu"
DEVICES=1
PRECISION="16-mixed"
BACKBONE="resnet18"
NUM_QUERIES=16  # 与训练时保持一致
FEATURE_DIM=256
NUM_DECODER_LAYERS=6
SEQUENCE_LENGTH=10
CONFIDENCE_THRESHOLD=0.5

# 创建带时间戳的实验名称
TIMESTAMP=$(date +%Y%m%d || echo "default")
EXP_NAME="e2e_perception_inference_${TIMESTAMP}"

# 设置日志文件
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

# 设置检查点路径 - 使用最新的训练检查点
CHECKPOINT_DIR="logs/e2e_perception_${TIMESTAMP}/version_104/checkpoints"
CHECKPOINT=$(ls -t "${CHECKPOINT_DIR}"/*.ckpt 2>/dev/null | head -n 1 || echo "")

if [ -z "${CHECKPOINT}" ]; then
    echo "Error: No checkpoint found in ${CHECKPOINT_DIR}"
    exit 1
fi

# 捕获输出并记录日志
{
    echo "======== System Resources Before Inference ========"
    echo "CPU Info:"
    lscpu | grep "Model name\|Socket(s)\|Core(s) per socket\|Thread(s) per core"
    
    echo -e "\nMemory Info:"
    free -h
    
    echo -e "\nDisk Space:"
    df -h

    echo -e "\nGPU Resources:"
    nvidia-smi

    echo -e "\n======== Inference Configuration ========"
    echo "Starting inference at $(date)"
    echo "Experiment Name: ${EXP_NAME}"
    echo "  Test List: ${TEST_LIST}"
    echo "  Checkpoint: ${CHECKPOINT}"
    echo "  Batch Size: ${BATCH_SIZE}"
    echo "  Num Workers: ${NUM_WORKERS}"
    echo "  Accelerator: ${ACCELERATOR}"
    echo "  Devices: ${DEVICES}"
    echo "  Precision: ${PRECISION}"
    echo "  Backbone: ${BACKBONE}"
    echo "  Num Queries: ${NUM_QUERIES}"
    echo "  Confidence Threshold: ${CONFIDENCE_THRESHOLD}"

    # 设置可见GPU
    export CUDA_VISIBLE_DEVICES=0

    echo -e "\n======== Inference Start ========"
    # 推理命令
    python inference.py \
        --test-list "${TEST_LIST}" \
        --sequence-length "${SEQUENCE_LENGTH}" \
        --output-dir "${RESULTS_DIR}/${EXP_NAME}" \
        --checkpoint "${CHECKPOINT}" \
        --feature-dim "${FEATURE_DIM}" \
        --num-queries "${NUM_QUERIES}" \
        --num-decoder-layers "${NUM_DECODER_LAYERS}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --confidence-threshold "${CONFIDENCE_THRESHOLD}" \
        --accelerator "${ACCELERATOR}" \
        --devices "${DEVICES}" \
        --precision "${PRECISION}"

    echo -e "\n======== System Resources After Inference ========"
    echo "CPU Load:"
    uptime
    
    echo -e "\nMemory Info:"
    free -h
    
    echo -e "\nDisk Space:"
    df -h

    echo -e "\nGPU Resources:"
    nvidia-smi

    echo -e "\nInference completed at $(date)"
} 2>&1 | tee "${LOG_FILE}"

echo "Inference log saved to ${LOG_FILE}"
echo "Results saved to ${RESULTS_DIR}/${EXP_NAME}" 