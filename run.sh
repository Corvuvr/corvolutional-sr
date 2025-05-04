#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
common="--checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --is-train"
# rep="--rep"
# benchmark="sintel_mountains"
benchmark="sintel_full"
num_blocks=4
num_epochs_per_round=1
num_rounds=35

fwd=flow_cat
loss=l2
python src/valid.py $rep $common \
    --dataroot      ../datasets/Sintel \
    --benchmark     $benchmark \
    --num-blocks    $num_blocks \
    --forward-opt   $fwd \
    --loss          $loss \
    --num-rounds    $num_rounds \
    --num-epochs-per-round $num_epochs_per_round
