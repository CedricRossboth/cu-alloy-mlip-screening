#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=3-0:0:0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --partition=h100
#SBATCH --job-name=elecatml
#SBATCH --array=0-9
#SBATCH --output=logs/slurm_%A_%a.out

# load cuda 12.4
module load gcc/13.2.0 cuda/12.4.1
# load conda env
source /path/to/miniconda3/bin/activate ecat

# commands
idx=${SLURM_ARRAY_TASK_ID:?no array id}
JOBID1=$((10#$idx * 2))
JOBID2=$((JOBID1 + 1))
FID1=$(printf "%02d" $JOBID1)
FID2=$(printf "%02d" $JOBID2)

python -u main.py \
    --output_dir "./results" \
    --model "EquiformerV2-31M-S2EF-OC20-All+MD" \
    --model_ckpt_dir "./ocp_checkpoints" \
    --bulk_atoms_file "data/alloy/copper_alloy_atoms_20_jobs/bulk_chemical_formula_${FID1}.pkl" \
    --job_id $JOBID1 \
    > logs/run_${FID1}.log 2>&1 &


python -u main.py \
    --output_dir "./results" \
    --model "EquiformerV2-31M-S2EF-OC20-All+MD" \
    --model_ckpt_dir "./ocp_checkpoints" \
    --bulk_atoms_file "data/alloy/copper_alloy_atoms_20_jobs/bulk_chemical_formula_${FID2}.pkl" \
    --job_id $JOBID2 \
    > logs/run_${FID2}.log 2>&1 &


wait
echo "TWO JOBS FINISHED"
