#!/bin/bash
SEEDS=(1 5 7)
LRS=(0.000001 0.000005 0.00001 0.00005 0.0001)
STEPS=(1 5 10)

for seed in ${SEEDS[@]}; do
    for step in ${STEPS[@]}; do
        for lr in ${LRS[@]}; do
            export WANDB_NAME=drl_seed_${seed}_lr_${lr}_step_${step}
            sbatch --job-name=ipd_drl run_drl.slurm ${seed} ${step} ${lr}
        done
    done
done