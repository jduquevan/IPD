#!/bin/bash
SEEDS=(7)
LRS=(0.0001 0.0005 0.001)
EPS=(0.1 0.2 0.3)
STEPS=20
ENT_WEIGHTS=(0.5 1 2)
EXP_WEIGHTS=(1)
COL_WEIGHTS=(1)


for lr in ${LRS[@]}; do
    for ep in ${EPS[@]}; do
        for seed in ${SEEDS[@]}; do
            for ent in ${ENT_WEIGHTS[@]}; do
                for exp in ${EXP_WEIGHTS[@]}; do
                    for col in ${COL_WEIGHTS[@]}; do
                        export WANDB_NAME=vip_v2_seed_${seed}_lr_${lr}_ep_${ep}
                        sbatch --job-name=vip_v2 run_vip_v2.slurm ${seed} ${STEPS} ${lr} ${ep} ${ent} ${exp} ${col}
                    done
                done
            done
        done
    done
done