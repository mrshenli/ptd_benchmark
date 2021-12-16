#!/bin/bash

srun --label launch_cluster_ddp.sh GPTLarge 24 noop

srun --label launch_cluster_ddp.sh GPTLarge 32 noop

srun --label launch_cluster_ddp.sh GPTLarge 48 noop

srun --label launch_cluster_ddp.sh GPTLarge 64 noop

srun --label launch_cluster_ddp.sh GPTLarge 80 noop



srun --label launch_cluster_fsdp.sh GPTLarge 32 noop

srun --label launch_cluster_fsdp.sh GPTLarge 48 noop

srun --label launch_cluster_fsdp.sh GPTLarge 64 noop

srun --label launch_cluster_fsdp.sh GPTLarge 80 noop

srun --label launch_cluster_fsdp.sh GPTLarge 96 noop


srun --label launch_cluster_pipeline.sh GPTLarge 48 2 noop

srun --label launch_cluster_pipeline.sh GPTLarge 48 4 noop

srun --label launch_cluster_pipeline.sh GPTLarge 64 2 noop

srun --label launch_cluster_pipeline.sh GPTLarge 64 4 noop

srun --label launch_cluster_pipeline.sh GPTLarge 96 2 noop

srun --label launch_cluster_pipeline.sh GPTLarge 96 4 noop

srun --label launch_cluster_pipeline.sh GPTLarge 128 2 noop

srun --label launch_cluster_pipeline.sh GPTLarge 128 4 noop

srun --label launch_cluster_pipeline.sh GPTLarge 160 2 noop

srun --label launch_cluster_pipeline.sh GPTLarge 160 4 noop



srun --label launch_cluster_ddp.sh GPTXXL 24 noop

srun --label launch_cluster_ddp.sh GPTXXL 32 noop

srun --label launch_cluster_ddp.sh GPTXXL 40 noop

srun --label launch_cluster_fsdp.sh GPTXXL 32 noop

srun --label launch_cluster_fsdp.sh GPTXXL 40 noop

srun --label launch_cluster_fsdp.sh GPTXXL 48 noop

srun --label launch_cluster_pipeline.sh GPTXXL 48 2 noop

srun --label launch_cluster_pipeline.sh GPTXXL 48 4 noop

srun --label launch_cluster_pipeline.sh GPTXXL 64 2 noop

srun --label launch_cluster_pipeline.sh GPTXXL 64 4 noop

srun --label launch_cluster_pipeline.sh GPTXXL 80 2 noop

srun --label launch_cluster_pipeline.sh GPTXXL 80 4 noop

srun --label launch_cluster_pipeline.sh GPTXXL 96 2 noop

srun --label launch_cluster_pipeline.sh GPTXXL 96 4 noop