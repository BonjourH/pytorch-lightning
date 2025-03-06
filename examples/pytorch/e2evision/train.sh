#!/bin/bash
set -euo pipefail  # Strict error handling mode

# Create log directory
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"
mkdir -p "${LOG_DIR}" "${CHECKPOINT_DIR}"

# Training parameters, can be overridden by environment variables
# Note: These parameters will override those in the configuration file
TRAIN_LIST=${TRAIN_LIST:-"train_clips.txt"}
VAL_LIST=${VAL_LIST:-"val_clips.txt"}
BATCH_SIZE=${BATCH_SIZE:-1}
NUM_WORKERS=${NUM_WORKERS:-1}
MAX_EPOCHS=${MAX_EPOCHS:-100}
ACCELERATOR=${ACCELERATOR:-"gpu"}
DEVICES=${DEVICES:-1}
PRECISION=${PRECISION:-"16-mixed"}
ACCUMULATE_GRAD_BATCHES=${ACCUMULATE_GRAD_BATCHES:-4}
BACKBONE=${BACKBONE:-"resnet18"}
NUM_QUERIES=${NUM_QUERIES:-16}
SEED=${SEED:-42}
RESUME=${RESUME:-""}  # Set to a non-empty value to use the --resume flag
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:-"true"}

# Create a timestamped experiment name
TIMESTAMP=$(date +%Y%m%d || echo "default")
EXP_NAME="e2e_perception_${TIMESTAMP}"

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
    echo "  Backbone: ${BACKBONE}"
    echo "  Num Queries: ${NUM_QUERIES}"
    echo "  Seed: ${SEED}"
    echo "  Resume: ${RESUME}"

    # Set visible GPUs
    export CUDA_VISIBLE_DEVICES=0

    echo -e "\n======== Training Start ========"
    
    # Build training command
    TRAIN_CMD="python train.py"
    
    # Add basic parameters
    TRAIN_CMD="${TRAIN_CMD} --experiment-name ${EXP_NAME} --save_dir ${LOG_DIR}"
    [ -n "${RESUME}" ] && TRAIN_CMD="${TRAIN_CMD} --resume"

    # Optional specify custom config module
    # CONFIG_MODULE="configs.custom_config"
    # [ -n "${CONFIG_MODULE}" ] && TRAIN_CMD="${TRAIN_CMD} --config-module ${CONFIG_MODULE}"
    
    # Create config override array
    CONFIG_OVERRIDES=()
    
    # Add each parameter to the override array
    CONFIG_OVERRIDES+=("training.train_list=${TRAIN_LIST}")
    CONFIG_OVERRIDES+=("training.val_list=${VAL_LIST}")
    CONFIG_OVERRIDES+=("training.batch_size=${BATCH_SIZE}")
    CONFIG_OVERRIDES+=("training.num_workers=${NUM_WORKERS}")
    CONFIG_OVERRIDES+=("training.max_epochs=${MAX_EPOCHS}")
    CONFIG_OVERRIDES+=("training.accelerator=${ACCELERATOR}")
    CONFIG_OVERRIDES+=("training.devices=${DEVICES}")
    CONFIG_OVERRIDES+=("training.precision=${PRECISION}")
    CONFIG_OVERRIDES+=("training.accumulate_grad_batches=${ACCUMULATE_GRAD_BATCHES}")
    CONFIG_OVERRIDES+=("model.backbone=${BACKBONE}")
    CONFIG_OVERRIDES+=("model.num_queries=${NUM_QUERIES}")
    CONFIG_OVERRIDES+=("training.seed=${SEED}")
    CONFIG_OVERRIDES+=("training.pretrained_weights=${PRETRAINED_WEIGHTS}")
    
    # Add config override parameters
    if [ ${#CONFIG_OVERRIDES[@]} -gt 0 ]; then
        OVERRIDE_STR=$(IFS=" " ; echo "${CONFIG_OVERRIDES[*]}")
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
