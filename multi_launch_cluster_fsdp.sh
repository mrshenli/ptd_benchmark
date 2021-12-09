#!/bin/bash

srun --label launch_cluster_fsdp.sh 4 noop

srun --label launch_cluster_fsdp.sh 6 noop

srun --label launch_cluster_fsdp.sh 8 noop

srun --label launch_cluster_fsdp.sh 10 noop

srun --label launch_cluster_fsdp.sh 12 noop

srun --label launch_cluster_fsdp.sh 14 noop

srun --label launch_cluster_fsdp.sh 8 checkpoint

srun --label launch_cluster_fsdp.sh 16 checkpoint

srun --label launch_cluster_fsdp.sh 24 checkpoint

srun --label launch_cluster_fsdp.sh 32 checkpoint

srun --label launch_cluster_fsdp.sh 40 checkpoint

srun --label launch_cluster_fsdp.sh 48 checkpoint

srun --label launch_cluster_fsdp.sh 56 checkpoint

srun --label launch_cluster_fsdp.sh 64 checkpoint

srun --label launch_cluster_fsdp.sh 72 checkpoint

srun --label launch_cluster_fsdp.sh 80 checkpoint

srun --label launch_cluster_fsdp.sh 8 offload

srun --label launch_cluster_fsdp.sh 16 offload

srun --label launch_cluster_fsdp.sh 32 offload

srun --label launch_cluster_fsdp.sh 40 offload

srun --label launch_cluster_fsdp.sh 48 offload

srun --label launch_cluster_fsdp.sh 56 offload

srun --label launch_cluster_fsdp.sh 64 offload

srun --label launch_cluster_fsdp.sh 72 offload

srun --label launch_cluster_fsdp.sh 80 offload