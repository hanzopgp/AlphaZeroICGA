#!/bin/bash

#SBATCH --partition=funky

#SBATCH --job-name=alphazero

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=60

#SBATCH --output=%x-%j.out

#SBATCH --error=%x-%j.err

srun --gpus-per-node=1 bash cluster_scripts/alphazero_dojos.sh