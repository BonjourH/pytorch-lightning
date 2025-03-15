#!/bin/bash
set -euo pipefail # Strict error handling mode

# Create a timestamped experiment name
TIMESTAMP=$(date +%Y%m%d || echo "default")
# EXP_NAME="e2e_perception_${TIMESTAMP}"
EXP_NAME="e2e_perception"

# Training parameters, can be overridden by environment variables
# Note: These parameters will override those in the configuration file
TRAIN_LIST=${TRAIN_LIST:-"train_clips.txt"}
VAL_LIST=${VAL_LIST:-"val_clips.txt"}
BATCH_SIZE=${BATCH_SIZE:-1}    # 默认减小批量大小为1
NUM_WORKERS=${NUM_WORKERS:-20} # 默认减少工作线程数为4
MAX_EPOCHS=${MAX_EPOCHS:-5000000}
ACCELERATOR=${ACCELERATOR:-"gpu"}
DEVICES=${DEVICES:-1}
PRECISION=${PRECISION:-32} # 16-mixed, 32, 64
ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES:-4}
NUM_QUERIES=${NUM_QUERIES:-64}
SEED=${SEED:-42}
RESUME=${RESUME:-0}
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-"true"}
RUN_ID=${RUN_ID:-0}
CONFIG_FILE=${CONFIG_FILE:-"configs/e2e_perception.yaml"}
VALIDATE_ONLY=${VALIDATE_ONLY:-0}             # 0: train and validate, 1: validate only
LIMIT_VAL_BATCHES=${LIMIT_VAL_BATCHES:-1}     # 1.0: validate all batches, 0.1: validate 10% of batches, 1: one batch
GRADIENT_CLIP_VAL=${GRADIENT_CLIP_VAL:-1.0}   # 添加梯度裁剪值
MEMORY_EFFICIENT=${MEMORY_EFFICIENT:-1}       # 添加内存效率选项
CLEAN_WANDB_HISTORY=${CLEAN_WANDB_HISTORY:-1} # 是否清理 W&B 历史数据，1: 清理, 0: 不清理
# Create log directory
LOG_DIR="logs/${EXP_NAME}"
CHECKPOINT_DIR="checkpoints/${EXP_NAME}"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

# Set log file
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

# Capture output and log
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
    echo "  Num Queries: ${NUM_QUERIES}"
    echo "  Seed: ${SEED}"
    echo "  Resume: ${RESUME}"

    # Set visible GPUs
    export CUDA_VISIBLE_DEVICES=0

    echo -e "\n======== Training Start ========"

    # Add memory optimization options
    if [ "${MEMORY_EFFICIENT}" = "1" ]; then
        echo "Enabling memory optimization options"
        # Set environment variables for PyTorch memory efficiency
        export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
        # Disable CUDA graph capture which can use extra memory
        export CUDA_LAUNCH_BLOCKING=1
    fi

    # Build training command
    TRAIN_CMD="python train.py"
    TRAIN_CMD="${TRAIN_CMD} --config_file ${CONFIG_FILE}"
    TRAIN_CMD="${TRAIN_CMD} --experiment_name ${EXP_NAME}"
    TRAIN_CMD="${TRAIN_CMD} --resume ${RESUME}"

    # Add validate-only flag if needed
    if [ "${VALIDATE_ONLY}" = "1" ]; then
        TRAIN_CMD="${TRAIN_CMD} --validate_only"
        echo "Running in validation-only mode"
    fi

    # Create config override array
    CONFIG_OVERRIDES=()

    # Add each parameter to the override array
    CONFIG_OVERRIDES+=("validate_only=${VALIDATE_ONLY}")
    CONFIG_OVERRIDES+=("training.train_list=${TRAIN_LIST}")
    CONFIG_OVERRIDES+=("training.val_list=${VAL_LIST}")
    CONFIG_OVERRIDES+=("training.batch_size=${BATCH_SIZE}")
    CONFIG_OVERRIDES+=("training.num_workers=${NUM_WORKERS}")
    CONFIG_OVERRIDES+=("training.max_epochs=${MAX_EPOCHS}")
    CONFIG_OVERRIDES+=("training.accelerator=${ACCELERATOR}")
    CONFIG_OVERRIDES+=("training.devices=${DEVICES}")
    CONFIG_OVERRIDES+=("training.precision=${PRECISION}")
    CONFIG_OVERRIDES+=("training.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}")
    CONFIG_OVERRIDES+=("training.seed=${SEED}")
    CONFIG_OVERRIDES+=("training.pretrained_weights=${PRETRAINED_WEIGHTS}")
    CONFIG_OVERRIDES+=("training.limit_val_batches=${LIMIT_VAL_BATCHES}")
    CONFIG_OVERRIDES+=("training.gradient_clip_val=${GRADIENT_CLIP_VAL}") # 添加梯度裁剪配置
    CONFIG_OVERRIDES+=("model.memory_efficient=${MEMORY_EFFICIENT}")      # 添加内存效率配置
    CONFIG_OVERRIDES+=("model.decoder.num_queries=${NUM_QUERIES}")
    CONFIG_OVERRIDES+=("logging.checkpoint_dir=${CHECKPOINT_DIR}")
    CONFIG_OVERRIDES+=("logging.last_checkpoint_dir=$(dirname "$(realpath "$0")")")
    CONFIG_OVERRIDES+=("logging.log_dir=${LOG_DIR}")
    CONFIG_OVERRIDES+=("logging.run_id=${RUN_ID}")
    CONFIG_OVERRIDES+=("logging.use_tensorboard=true")
    CONFIG_OVERRIDES+=("logging.use_csv=false")
    CONFIG_OVERRIDES+=("logging.use_wandb=true")
    CONFIG_OVERRIDES+=("logging.wandb_project=e2e_perception")
    CONFIG_OVERRIDES+=("logging.use_optional_metrics=false")
    CONFIG_OVERRIDES+=("logging.clean_wandb_history=${CLEAN_WANDB_HISTORY}")

    # Add config override parameters
    if [ ${#CONFIG_OVERRIDES[@]} -gt 0 ]; then
        OVERRIDE_STR=$(
            IFS=" "
            echo "${CONFIG_OVERRIDES[*]}"
        )
        TRAIN_CMD="${TRAIN_CMD} --config-override ${OVERRIDE_STR}"
    fi

    # Execute training command
    echo "Running command: ${TRAIN_CMD}"
    eval ${TRAIN_CMD}

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
