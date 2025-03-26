#! /bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 运行训练脚本
python ./xinnovation/apis/train.py --config_file ./xinnovation/configs/4d/sparse4D/sparse4dv3_temporal_base.py