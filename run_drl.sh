#!/bin/bash
SEEDS=(1 3 5 7 11)
LRS=(0.0001 0.0005 0.001 0.005)
STEPS=(10)

for lr in ${LRS[@]}; do
    for step in ${STEPS[@]}; do
        for seed in ${SEEDS[@]}; do
            export WANDB_NAME=vip_cg_eg_seed_${seed}_lr_${lr}_step_${step}
            sbatch --job-name=cg_eg_vip run_drl.slurm ${seed} ${step} ${lr}
        done
    done
done