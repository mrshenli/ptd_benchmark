#!/bin/bash


srun --label launch_cluster_pipeline.sh 40 4 checkpoint

srun --label launch_cluster_pipeline.sh 48 4 checkpoint

srun --label launch_cluster_pipeline.sh 56 4 checkpoint

srun --label launch_cluster_pipeline.sh 64 4 checkpoint

srun --label launch_cluster_pipeline.sh 40 4 offload

srun --label launch_cluster_pipeline.sh 48 4 offload

srun --label launch_cluster_pipeline.sh 56 4 offload

srun --label launch_cluster_pipeline.sh 64 4 offload

srun --label launch_cluster_pipeline.sh 40 2 checkpoint

srun --label launch_cluster_pipeline.sh 48 2 checkpoint

srun --label launch_cluster_pipeline.sh 56 2 checkpoint

srun --label launch_cluster_pipeline.sh 64 2 checkpoint

srun --label launch_cluster_pipeline.sh 40 2 offload

srun --label launch_cluster_pipeline.sh 48 2 offload

srun --label launch_cluster_pipeline.sh 56 2 offload

srun --label launch_cluster_pipeline.sh 64 2 offload