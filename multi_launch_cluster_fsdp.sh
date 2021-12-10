#!/bin/bash

srun --label launch_cluster_fsdp.sh 48 offload

srun --label launch_cluster_fsdp.sh 64 offload

srun --label launch_cluster_fsdp.sh 80 offload

srun --label launch_cluster_fsdp.sh 96 offload

srun --label launch_cluster_fsdp.sh 112 offload

srun --label launch_cluster_fsdp.sh 128 offload

srun --label launch_cluster_fsdp.sh 144 offload