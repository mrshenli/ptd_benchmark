#!/bin/bash


srun --label launch_cluster_pipeline.sh GPTSmall 40 2 noop

srun --label launch_cluster_pipeline.sh GPTSmall 40 4 noop

srun --label launch_cluster_pipeline.sh GPTLarge 16 2 noop

srun --label launch_cluster_pipeline.sh GPTLarge 40 2 noop

srun --label launch_cluster_pipeline.sh GPTLarge 40 4 noop

srun --label launch_cluster_pipeline.sh GPTXXL 40 2 noop

srun --label launch_cluster_pipeline.sh GPTXXL 40 4 noop














