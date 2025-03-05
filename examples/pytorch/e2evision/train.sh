#!/bin/bash
set -euo pipefail  # 严格的错误处理模式

# 创建日志目录
LOG_DIR="logs"
mkdir -p "${LOG_DIR}"

# 训练配置
TRAIN_LIST="train_clips.txt"
VAL_LIST="val_clips.txt"
BATCH_SIZE=1
NUM_WORKERS=4
MAX_EPOCHS=100
ACCELERATOR="gpu"
DEVICES=8
PRECISION="32-true"

# 创建带时间戳的实验名称
TIMESTAMP=$(date +%Y%m%d_%H%M%S || echo "default")
EXP_NAME="e2e_perception_${TIMESTAMP}"

# 设置日志文件
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

# 捕获输出并记录日志
{
    echo "======== System Resources Before Training ========"
    echo "CPU Info:"
    lscpu | grep "Model name\|Socket(s)\|Core(s) per socket\|Thread(s) per core"
    
    echo -e "\nMemory Info:"
    free -h
    
    echo -e "\nDisk Space:"
    df -h

    echo -e "\nGPU Resources:"
    nvidia-smi

    echo -e "\n======== Training Configuration ========"
    echo "Starting training at $(date)"
    echo "Experiment Name: ${EXP_NAME}"
    echo "  Train List: ${TRAIN_LIST}"
    echo "  Val List: ${VAL_LIST}"
    echo "  Batch Size: ${BATCH_SIZE}"
    echo "  Num Workers: ${NUM_WORKERS}"
    echo "  Max Epochs: ${MAX_EPOCHS}"
    echo "  Accelerator: ${ACCELERATOR}"
    echo "  Devices: ${DEVICES}"
    echo "  Precision: ${PRECISION}"

    # 设置可见GPU
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9

    echo -e "\n======== Training Start ========"
    # 训练命令
    # 注意: 如果需要使用预训练权重(需要网络连接)，请添加 --pretrained-weights 参数
    python train.py \
        --train-list "${TRAIN_LIST}" \
        --val-list "${VAL_LIST}" \
        --batch-size "${BATCH_SIZE}" \
        --num-workers "${NUM_WORKERS}" \
        --max-epochs "${MAX_EPOCHS}" \
        --accelerator "${ACCELERATOR}" \
        --devices "${DEVICES}" \
        --precision "${PRECISION}" \
        --experiment-name "${EXP_NAME}"

    echo -e "\n======== System Resources After Training ========"
    echo "CPU Load:"
    uptime
    
    echo -e "\nMemory Info:"
    free -h
    
    echo -e "\nDisk Space:"
    df -h

    echo -e "\nGPU Resources:"
    nvidia-smi

    echo -e "\nTraining completed at $(date)"
} 2>&1 | tee "${LOG_FILE}"

echo "Training log saved to ${LOG_FILE}"