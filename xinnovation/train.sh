#! /bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

rm -rf ./xinnovation_visualize_intermediate_results 
rm -rf ./xinnovation_checkpoints
# 运行训练脚本
python ./xinnovation/apis/train.py --config_file ./xinnovation/examples/detector4D/sparse4dv3_temporal_base.py