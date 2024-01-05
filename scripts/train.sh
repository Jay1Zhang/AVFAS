#!/bin/bash

task=avvp
mode=train # retrain_ma, retrain_denoise

scale=[1,3,5,7,9]
num_cells=4

epochs=100
batch_size=32
lr=1e-4
wd=1e-5
drpt=0.4

label_denoise=/home/xxx/research/audio-visual/JoMoLD/noise_ratios.npz
label_ma=/home/xxx/research/audio-visual/ma/step3_retrain/need_to_change.pkl
# label_ma=/home/xxx/research/audio-visual/CM-Co-Occurrence-AVVP/need_to_change.pkl

search_dir=/home/xxx/research/mm-nas/results/avvp/sota-search-M3T-20230416-091935


if [ $1 ]; then
    echo DDP
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 --master_port 56668 \
        main_avvp.py --task $task --mode $mode --parallel --tensorboard \
            --scale $scale --num_cells $num_cells \
            --epochs $epochs --batch_size $batch_size --lr $lr --weight_decay $wd --drpt $drpt \
            --search_dir $search_dir --label_denoise $label_denoise --label_ma $label_ma
else
    CUDA_VISIBLE_DEVICES=9 python \
        main_avvp.py --task $task --mode $mode --tensorboard \
            --scale $scale --num_cells $num_cells \
            --epochs $epochs --batch_size $batch_size --lr $lr --weight_decay $wd --drpt $drpt \
            --search_dir $search_dir --label_denoise $label_denoise 
fi