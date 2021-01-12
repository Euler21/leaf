#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -q regular 
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --array=1-400:10

module load tensorflow
source activate rayleaf

CASE_NUM=$SLURM_ARRAY_TASK_ID
SEED=${SLURMD_NODENAME: -3} # Seed is based on the last three digits of node name; bit of a hack, but it quickly allows us to run multiple trials

python ../../models/main.py -dataset femnist -model cnn --num-rounds 80 --eval-every 1 --clients-per-round 2 --num-epochs 5 -lr 0.06 --num-client-servers 5 --metrics-name svd_rank$CASE_NUM_$SEED --seed $SEED --sketcher SVD --rank $CASE_NUM