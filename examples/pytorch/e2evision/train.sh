#!/bin/bash
set -e

# Training configuration
TRAIN_LIST="train_clips.txt"
VAL_LIST="val_clips.txt"
BATCH_SIZE=2
NUM_WORKERS=4
MAX_EPOCHS=100
ACCELERATOR="gpu"
DEVICES=1
PRECISION="32-true"

# Create experiment name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_NAME="e2e_perception_${TIMESTAMP}"

# Training command
python train.py \
    --train-list ${TRAIN_LIST} \
    --val-list ${VAL_LIST} \
    --batch-size ${BATCH_SIZE} \
    --num-workers ${NUM_WORKERS} \
    --max-epochs ${MAX_EPOCHS} \
    --accelerator ${ACCELERATOR} \
    --devices ${DEVICES} \
    --precision ${PRECISION} \
    --experiment-name ${EXP_NAME}