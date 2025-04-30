#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
common="--checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --rep --is-train"
bechmark="sintel"
# bechmark="sintel_full"
num_blocks=2
python src/valid.py $common \
    --dataroot   ../datasets/Sintel \
    --benchmark  $bechmark \
    --num-blocks $num_blocks