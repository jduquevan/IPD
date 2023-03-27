#!/bin/bash
SEEDS=(7)
LRS=(0.0001)
STEPS=(20)
ENT_WEIGHTS=(0 0.25 0.5 1)
EXP_WEIGHTS=(0 0.25 0.5 1)
COL_WEIGHTS=(0 0.25 0.5 1)


for lr in ${LRS[@]}; do
    for step in ${STEPS[@]}; do
        for seed in ${SEEDS[@]}; do
            for ent in ${ENT_WEIGHTS[@]}; do
                for exp in ${EXP_WEIGHTS[@]}; do
                    for col in ${COL_WEIGHTS[@]}; do
                        export WANDB_NAME=vip_cg_eg_seed_${seed}_lr_${lr}_ent_${ent}_exp_${exp}_col_${col}
                        sbatch --job-name=cg_eg_vip run_drl.slurm ${seed} ${step} ${lr} ${ent} ${exp} ${col} 
                    done
                done
            done
        done
    done
done