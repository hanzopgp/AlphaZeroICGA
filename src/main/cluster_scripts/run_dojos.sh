#!/bin/bash

#SBATCH --partition=funky

#SBATCH --job-name=alphazero_dojos

#SBATCH --gpus-per-node=0

#SBATCH --nodes=3

#SBATCH --time=120

#SBATCH --output=cluster_logs/%x-%j.out

#SBATCH --error=cluster_logs/%x-%j.err

# srun --nodes="$1" bash cluster_scripts/alphazero_dojos.sh "$2"
srun bash cluster_scripts/alphazero_dojos.sh "$2"

# command=""
# for i in  {0. .$(($1))}	
# do
# 	command+="srun --gpus-per-node=0 bash cluster_scripts/alphazero_dojos.sh ${2} &"
# done
# command+=" wait"
# eval $command