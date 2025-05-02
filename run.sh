#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
common="--checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --rep --is-train"
bechmark="sintel"
bechmark="sintel_full"
fwd=flow_cat
num_blocks=3
python src/valid.py $common \
    --dataroot      ../datasets/Sintel \
    --benchmark     $bechmark \
    --num-blocks    $num_blocks \
    --forward-opt   $fwd