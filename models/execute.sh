#!/bin/bash

#SBATCH -t 00:02:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --constraint=haswell
#SBATCH --cpus-per-task=16
#SBATCH -A mp156
#SBATCH -N 1 # number of nodes
#SBATCH --cpus-per-task=16

. modules.sh

python main.py -dataset femnist -model tt_cnn --minibatch 0.1 --clients-per-round 10 --num-rounds 3000 --metrics-name tt_cnn

