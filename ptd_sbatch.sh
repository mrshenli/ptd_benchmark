#!/bin/bash

#SBATCH --job-name=ptd

#SBATCH --open-mode=append

#SBATCH --partition=train

#SBATCH --nodes=8

#SBATCH --gpus-per-node=8

#SBATCH --gpus-per-task=1

#SBATCH --time=24:00:00

srun --label ptd.sh

