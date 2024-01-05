#!/bin/bash

task=avvp
mode=estimate_ma # estimate_ma_search  # estimate_noise_search

scale=[1,3,5,7,9]
num_cells=4

batch_size=32

search_dir=/home/xxx/research/mm-nas/results/avvp/sota-search-M3T-20230416-091935/
result_dir=/home/xxx/research/mm-nas/results/avvp/sota-search-M3T-20230416-091935/train-M3T-20230420-131042

CUDA_VISIBLE_DEVICES=4 python \
        main_avvp.py --task $task --mode $mode \
            --scale $scale --num_cells $num_cells \
            --batch_size $batch_size \
            --search_dir $search_dir --result_dir $result_dir 