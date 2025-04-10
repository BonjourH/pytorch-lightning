#! /bin/bash

# 设置环境变量
export PYTHONPATH=$PYTHONPATH:$(pwd)

config_file=${1:-"./xinnovation/examples/detector4D/sparse4dv3_temporal_base.py"}
mode=${2:-"train"}
delete_checkpoints=${3:-0}


rm -rf ./xinnovation_visualize_intermediate_results
if [ $delete_checkpoints -eq 1 ]; then
    rm -rf ./xinnovation_checkpoints
fi
# 运行训练脚本
python ./xinnovation/apis/train.py --config_file $config_file --mode $mode
