#!/bin/bash

#SBATCH --job-name=ptd

#SBATCH --open-mode=append

#SBATCH --partition=train

#SBATCH --nodes=32

#SBATCH --cpus-per-gpu=12

#SBATCH --gpus-per-node=8

#SBATCH --gpus-per-task=1

#SBATCH --time=24:00:00

srun --label fsdp.sh

