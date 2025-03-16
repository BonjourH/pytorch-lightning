#!/bin/bash
set -euo pipefail # 严格的错误处理模式

# 创建日志和结果目录
RESULTS_DIR="predict_results"
mkdir -p "${RESULTS_DIR}"

# 推理参数，可通过环境变量覆盖默认值
# 注意：这些参数会覆盖配置文件中的参数
TEST_LIST=${TEST_LIST:-"test_clips.txt"}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-20}
ACCELERATOR=${ACCELERATOR:-"gpu"}
DEVICES=${DEVICES:-1}
PRECISION=${PRECISION:-"32"}
CONFIDENCE_THRESHOLD=${CONFIDENCE_THRESHOLD:-0.5}
CHECKPOINT=${CHECKPOINT:-"last.ckpt"} # 默认使用保存的最新模型
CONFIG_FILE=${CONFIG_FILE:-"configs/one_cycle.json"}
USE_OVERRIDES=${USE_OVERRIDES:-1}

# 创建带时间戳的实验名称
TIMESTAMP=$(date +%Y%m%d || echo "default")
EXP_NAME="e2e_perception" #_${TIMESTAMP}"

# 设置日志文件
LOG_FILE="${RESULTS_DIR}/${EXP_NAME}.log"
CHECKPOINT="checkpoints/${EXP_NAME}/last.ckpt"

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
    echo "  Confidence Threshold: ${CONFIDENCE_THRESHOLD}"
    echo "  Config File: ${CONFIG_FILE}"

    # 设置可见GPU
    export CUDA_VISIBLE_DEVICES=0

    echo -e "\n======== Inference Start ========"

    # 构建推理命令
    INFERENCE_CMD="python predict.py"

    # 添加基本参数
    INFERENCE_CMD="${INFERENCE_CMD} --checkpoint ${CHECKPOINT}"
    INFERENCE_CMD="${INFERENCE_CMD} --test_list ${TEST_LIST}"
    INFERENCE_CMD="${INFERENCE_CMD} --config_file ${CONFIG_FILE}"

    # 创建配置覆盖数组
    CONFIG_OVERRIDES=()

    # 添加每个参数到覆盖数组
    CONFIG_OVERRIDES+=("predict.batch_size=${BATCH_SIZE}")
    CONFIG_OVERRIDES+=("predict.num_workers=${NUM_WORKERS}")
    CONFIG_OVERRIDES+=("predict.accelerator=${ACCELERATOR}")
    CONFIG_OVERRIDES+=("predict.devices=${DEVICES}")
    CONFIG_OVERRIDES+=("predict.precision=${PRECISION}")
    CONFIG_OVERRIDES+=("predict.confidence_threshold=${CONFIDENCE_THRESHOLD}")
    CONFIG_OVERRIDES+=("predict.output_dir=${RESULTS_DIR}/${EXP_NAME}")

    # 添加配置覆盖参数
    if [ "$USE_OVERRIDES" -eq 1 ]; then
        if [ ${#CONFIG_OVERRIDES[@]} -gt 0 ]; then
            OVERRIDE_STR=$(
                IFS=" "
                echo "${CONFIG_OVERRIDES[*]}"
            )
            INFERENCE_CMD="${INFERENCE_CMD} --config-override ${OVERRIDE_STR}"
        fi
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
