#!/bin/bash
SEEDS=(7)
LRS=(0.0001 0.0005 0.001)
EPS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1)
STEPS=20
ENT_WEIGHTS=(1)
EXP_WEIGHTS=(1)
COL_WEIGHTS=(1)


for lr in ${LRS[@]}; do
    for ep in ${EPS[@]}; do
        for seed in ${SEEDS[@]}; do
            export WANDB_NAME=vip_v2_seed_${seed}_lr_${lr}_ep_${ep}
            sbatch --job-name=vip_v2 run_vip_v2.slurm ${seed} ${STEPS} ${lr} ${ep}
        done
    done
done