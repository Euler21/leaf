#!/bin/bash

#SBATCH -p shared # partition (queue)
#SBATCH -N 1 # number of nodes
#SBATCH -t 0-4:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --constraint=haswell
#SBATCH -A mp156

. modules.sh

python main_baseline.py -dataset femnist -model cnn --minibatch 0.1 --clients-per-round 10 --num-rounds 3000 --metrics-name cnn 

