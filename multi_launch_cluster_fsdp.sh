#!/bin/bash

srun --label launch_cluster_fsdp.sh 16

srun --label sbatch_cluster_fsdp.sh 24