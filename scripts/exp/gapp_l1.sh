#!/bin/bash

F="l1"
save_dir="exp_gapp_pr/cbtcn/"

for rl in 0.1 0.01 0.001; do
    mkdir -p ${save_dir}${F}_${rl}/
    python train.py --save_path ${save_dir}${F}_${rl}/ --regularizer $F --lambda $rl
done
