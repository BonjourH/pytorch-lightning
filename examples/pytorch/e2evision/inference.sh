#!/bin/bash
set -euo pipefail  # 严格的错误处理模式

# 创建日志和结果目录
LOG_DIR="logs"
RESULTS_DIR="results"
CHECKPOINT_DIR="checkpoints"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}" "${CHECKPOINT_DIR}"

# 推理参数，可通过环境变量覆盖默认值
# 注意：这些参数会覆盖配置文件中的参数
TEST_LIST=${TEST_LIST:-"test_list.txt"}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-4}
ACCELERATOR=${ACCELERATOR:-"gpu"}
DEVICES=${DEVICES:-1}
PRECISION=${PRECISION:-"16-mixed"}
BACKBONE=${BACKBONE:-"resnet18"}
NUM_QUERIES=${NUM_QUERIES:-16}
SEQUENCE_LENGTH=${SEQUENCE_LENGTH:-10}
CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.5}
CHECKPOINT=${CHECKPOINT:-"${CHECKPOINT_DIR}/model.ckpt"}  # 默认使用保存的最新模型

# 创建带时间戳的实验名称
TIMESTAMP=$(date +%Y%m%d || echo "default")
EXP_NAME="e2e_perception_inference_${TIMESTAMP}"

# 设置日志文件
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

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
    
    # 构建推理命令
    INFERENCE_CMD="python inference.py"
    
    # 添加基本参数
    INFERENCE_CMD="${INFERENCE_CMD} --output-dir ${RESULTS_DIR}/${EXP_NAME}"
    [ -n "${CHECKPOINT}" ] && INFERENCE_CMD="${INFERENCE_CMD} --checkpoint ${CHECKPOINT}"
    [ -n "${TEST_LIST}" ] && INFERENCE_CMD="${INFERENCE_CMD} --test-list ${TEST_LIST}"
    
    # 可选指定自定义配置模块
    # CONFIG_MODULE="configs.custom_config"
    # [ -n "${CONFIG_MODULE}" ] && INFERENCE_CMD="${INFERENCE_CMD} --config-module ${CONFIG_MODULE}"
    
    # 使用config-override添加其他参数
    CONFIG_OVERRIDES=()
    [ -n "${SEQUENCE_LENGTH}" ] && CONFIG_OVERRIDES+=("inference.sequence_length=${SEQUENCE_LENGTH}")
    [ -n "${BATCH_SIZE}" ] && CONFIG_OVERRIDES+=("inference.batch_size=${BATCH_SIZE}")
    [ -n "${NUM_WORKERS}" ] && CONFIG_OVERRIDES+=("inference.num_workers=${NUM_WORKERS}")
    [ -n "${CONFIDENCE_THRESHOLD}" ] && CONFIG_OVERRIDES+=("inference.confidence_threshold=${CONFIDENCE_THRESHOLD}")
    [ -n "${ACCELERATOR}" ] && CONFIG_OVERRIDES+=("inference.accelerator=${ACCELERATOR}")
    [ -n "${DEVICES}" ] && CONFIG_OVERRIDES+=("inference.devices=${DEVICES}")
    [ -n "${PRECISION}" ] && CONFIG_OVERRIDES+=("inference.precision=${PRECISION}")
    [ -n "${BACKBONE}" ] && CONFIG_OVERRIDES+=("model.backbone=${BACKBONE}")
    [ -n "${NUM_QUERIES}" ] && CONFIG_OVERRIDES+=("model.num_queries=${NUM_QUERIES}")
    
    # 添加配置覆盖参数
    if [ ${#CONFIG_OVERRIDES[@]} -gt 0 ]; then
        INFERENCE_CMD="${INFERENCE_CMD} --config-override ${CONFIG_OVERRIDES[@]}"
    fi
    
    # 执行推理命令
    echo "Running command: ${INFERENCE_CMD}"
    eval ${INFERENCE_CMD}

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