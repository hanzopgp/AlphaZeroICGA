#!/bin/bash

#SBATCH --partition=funky

#SBATCH --job-name=alphazero

#SBATCH --nodes=2

#SBATCH --gpus-per-node=1

#SBATCH --time=60

#SBATCH --output=%x-%j.out

#SBATCH --error=%x-%j.err

srun --gpus-per-node=1 bash alphazero.sh &

srun --gpus-per-node=1 bash alphazero.sh 