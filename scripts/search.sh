#!/bin/bash

task=avvp
mode=search

scale=[1,3,5,7,9]
# scale=[5,4,2,1]
num_cells=4

epochs=100
batch_size=32
lr=1e-4
wd=1e-3
drpt=0.4

label_ma=/home/xxx/research/audio-visual/ma/step3_retrain/need_to_change.pkl
label_denoise=/home/xxx/research/audio-visual/JoMoLD/noise_ratios.npz


if [ $1 ]; then
    echo DDP
    CUDA_VISIBLE_DEVICES=1,3 torchrun --nproc_per_node=2 --master_port 52416 \
        main_avvp.py --task $task --mode $mode --parallel --tensorboard \
            --scale $scale --num_cells $num_cells \
            --epochs $epochs --batch_size $batch_size --lr $lr --weight_decay $wd --drpt $drpt --label_ma $label_ma --label_denoise $label_denoise
else
    CUDA_VISIBLE_DEVICES=0 python \
        main_avvp.py --task $task --mode $mode --tensorboard \
            --scale $scale --num_cells $num_cells \
            --epochs $epochs --batch_size $batch_size --lr $lr --weight_decay $wd --drpt $drpt --label_ma $label_ma --label_denoise $label_denoise
fi