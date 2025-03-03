#!/bin/bash
set -e

python train.py \
    --train-list train_clips.txt \
    --val-list val_clips.txt \
    --batch-size 2 \
    --num-epochs 100