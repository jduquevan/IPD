#!/bin/bash
SEEDS=(1 5 7)
LRS=(0.00001 0.00005 0.001 0.005 0.01)
STEPS=(10)

for lr in ${LRS[@]}; do
    for step in ${STEPS[@]}; do
        for seed in ${SEEDS[@]}; do
            export WANDB_NAME=drl_seed_${seed}_lr_${lr}_step_${step}
            sbatch --job-name=ipd_sgd_drl run_drl.slurm ${seed} ${step} ${lr}
        done
    done
done