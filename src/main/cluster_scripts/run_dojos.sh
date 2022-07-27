#!/bin/bash

#SBATCH --partition=funky

#SBATCH --job-name=alphazero_dojos

#SBATCH --nodes=1

#SBATCH --gpus-per-node=0

#SBATCH --time=120

#SBATCH --output=cluster_logs/%x-%j.out

#SBATCH --error=cluster_logs/%x-%j.err

srun --gpus-per-node=0 bash cluster_scripts/alphazero_dojos.sh $1