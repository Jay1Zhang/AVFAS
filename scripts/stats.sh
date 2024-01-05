#!/bin/bash

task=avvp
mode=stats

scale=[10,5]
num_cells=4

epochs=40
batch_size=32
lr=3e-4
wd=1e-4
drpt=0.4

search_dir=/home/xxx/research/mm-nas/results/avvp/search-M3T-20230223-093735

CUDA_VISIBLE_DEVICES=4 python \
    stats.py --task $task --mode $mode --tensorboard \
        --scale $scale --num_cells $num_cells \
        --epochs $epochs --batch_size $batch_size --lr $lr --weight_decay $wd --drpt $drpt --search_dir $search_dir
