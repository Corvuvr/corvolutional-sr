#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
python src/valid.py --checkpoint-id rt4ksr_x2 --scale 2 --arch rt4ksr_rep --rep --is-train --dataroot ../datasets/Sintel