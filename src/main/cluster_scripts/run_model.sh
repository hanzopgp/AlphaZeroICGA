#!/bin/bash

#SBATCH --partition=funky

#SBATCH --job-name=alphazero_model

#SBATCH --nodes=1

#SBATCH --gpus-per-node=1

#SBATCH --time=60

#SBATCH --output=cluster_logs/%x-%j.out

#SBATCH --error=cluster_logs/%x-%j.err

srun --gpus-per-node=1 bash cluster_scripts/alphazero_train.sh "${1}" "${2}"