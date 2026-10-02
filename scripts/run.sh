#!/bin/bash

#[ ! -d "logs" ] && mkdir -p logs
[ ! -d "logs" ] && mkdir -p logs


JOBID=$1
FID=$(printf "%02d" $JOBID)

python -u main.py \
    --output_dir "./results" \
    --model "EquiformerV2-31M-S2EF-OC20-All+MD" \
    --model_ckpt_dir "./ocp_checkpoints" \
    --bulk_atoms_file "data/alloy/copper_alloy_atoms_10_jobs/bulk_chemical_formula_${FID}.pkl" \
    --job_id $JOBID \
    > logs/run_${FID}.log 2>&1
