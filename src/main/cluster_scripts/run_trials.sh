#!/bin/bash

#SBATCH --partition=funky

#SBATCH --job-name=alphazero_trials

#SBATCH --nodes=2

#SBATCH --gpus-per-node=0

#SBATCH --time=120

#SBATCH --output=cluster_logs/%x-%j.out

#SBATCH --error=cluster_logs/%x-%j.err

srun --gpus-per-node=0 bash cluster_scripts/alphazero_trials.sh & srun --gpus-per-node=0 bash cluster_scripts/alphazero_trials.sh & wait

# command=""
# for i in  {0. .$(($1))}
# do
# 	command+="srun --gpus-per-node=0 bash cluster_scripts/alphazero_trials.sh ${2} &"
# done
# command+=" wait"
# eval $command