#!/bin/bash -l
#SBATCH -C haswell
#SBATCH -q premium 
#SBATCH -N 5
#SBATCH -t 7:00:00

module load tensorflow

SEED=${SLURMD_NODENAME: -3} # Seed is based on the last three digits of node name; bit of a hack, but it quickly allows us to run multiple trials

python main.py --eval-every 2 -dataset femnist -model tt_cnn2 -lr 0.06 --num-epochs 5 --clients-per-round 10 --num-rounds 50 --metrics-name tt_cnn2_avg_5_$SEED --seed $SEED
python main.py --eval-every 2 -dataset femnist -model tt_cnn4 -lr 0.06 --num-epochs 5 --clients-per-round 10 --num-rounds 50 --metrics-name tt_cnn4_avg_5_$SEED --seed $SEED
python main.py --eval-every 2 -dataset femnist -model tt_cnn6 -lr 0.06 --num-epochs 5 --clients-per-round 10 --num-rounds 50 --metrics-name tt_cnn6_avg_5_$SEED --seed $SEED
python main.py --eval-every 2 -dataset femnist -model tt_cnn8 -lr 0.06 --num-epochs 5 --clients-per-round 10 --num-rounds 50 --metrics-name tt_cnn8_avg_5_$SEED --seed $SEED

python main.py --eval-every 2 -dataset femnist -model tucker_cnn -lr 0.06 --num-epochs 5 --clients-per-round 10 --num-rounds 50 --metrics-name tucker_cnn_avg_5_$SEED --seed $SEED
python main.py --eval-every 2 -dataset femnist -model cnn -lr 0.06 --num-epochs 5 --clients-per-round 10 --num-rounds 50 --metrics-name cnn_avg_5_$SEED --seed $SEED
python main.py --eval-every 2 -dataset femnist -model cnn32 -lr 0.06 --num-epochs 5 --clients-per-round 10 --num-rounds 50 --metrics-name cnn32_avg_5_$SEED --seed $SEED
python main.py --eval-every 2 -dataset femnist -model cnn16 -lr 0.06 --num-epochs 5 --clients-per-round 10 --num-rounds 50 --metrics-name cnn16_avg_5_$SEED --seed $SEED
python main.py --eval-every 2 -dataset femnist -model cnn8 -lr 0.06 --num-epochs 5 --clients-per-round 10 --num-rounds 50 --metrics-name cnn8_avg_5_$SEED --seed $SEED
python main.py --eval-every 2 -dataset femnist -model cnn4 -lr 0.06 --num-epochs 5 --clients-per-round 10 --num-rounds 50 --metrics-name cnn4_avg_5_$SEED --seed $SEED

# python main.py -dataset femnist -model cnn -lr 0.06 --minibatch 0.1 --clients-per-round 10 --num-rounds 2500 --metrics-name cnn_mbsgd_pt1
# python main.py -dataset femnist -model cnn32 -lr 0.06 --minibatch 0.1 --clients-per-round 10 --num-rounds 2500 --metrics-name cnn32_mbsgd_pt1
# python main.py -dataset femnist -model cnn16 -lr 0.06 --minibatch 0.1 --clients-per-round 10 --num-rounds 2500 --metrics-name cnn16_mbsgd_pt1
# python main.py -dataset femnist -model cnn8 -lr 0.06 --minibatch 0.1 --clients-per-round 10 --num-rounds 2500 --metrics-name cnn8_mbsgd_pt1
# python main.py -dataset femnist -model cnn4 -lr 0.06 --minibatch 0.1 --clients-per-round 10 --num-rounds 2500 --metrics-name cnn4_mbsgd_pt1

