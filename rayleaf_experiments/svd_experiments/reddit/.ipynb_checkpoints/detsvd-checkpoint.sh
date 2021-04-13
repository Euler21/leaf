#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -q regular 
#SBATCH -N 1
#SBATCH -t 5:00:00
#SBATCH --array=0-50:10

module load tensorflow
source activate rayleaf

CASE_NUM=$SLURM_ARRAY_TASK_ID

python ../../../models/main.py -dataset reddit -model stacked_lstm --num-rounds 100 --eval-every 1 --clients-per-round 1 --num-epochs 1 -lr 8 --num-client-servers 10 --metrics-name reddit_detsvd --batch-size 5 --sketcher SVD --rank $CASE_NUM