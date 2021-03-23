#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -q regular 
#SBATCH -N 1
#SBATCH -t 2:00:00
#SBATCH --array=5-100:5

module load tensorflow
source activate rayleaf

CASE_NUM=$SLURM_ARRAY_TASK_ID

python ../../../models/main.py -dataset femnist -model cnn --num-rounds 80 --eval-every 1 --clients-per-round 2 --num-epochs 5 -lr 0.06 --num-client-servers 5 --metrics-name svd_rank$CASE_NUM --sketcher RandomizedSVD --rank $CASE_NUM