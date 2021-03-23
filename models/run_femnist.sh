#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -q premium 
#SBATCH -N 1
#SBATCH -t 1:00:00

module load tensorflow
source activate rayleaf


python main.py -dataset femnist -model cnn --num-rounds 80 --eval-every 1 --clients-per-round 2 --num-epochs 5 -lr 0.06 --num-client-servers 5