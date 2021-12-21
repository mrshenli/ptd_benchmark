#!/bin/bash


srun --label launch_cluster_ddp.sh GPTSmall 32 noop 1

srun --label launch_cluster_ddp.sh GPTMedium 32 noop 1

srun --label launch_cluster_ddp.sh GPTLarge 32 noop 1

srun --label launch_cluster_ddp.sh GPTXL 32 noop 1

srun --label launch_cluster_ddp.sh GPTXXL 32 noop 1


srun --label launch_cluster_ddp.sh GPTSmall 60 noop 1

srun --label launch_cluster_ddp.sh GPTMedium 60 noop 1

srun --label launch_cluster_ddp.sh GPTLarge 60 noop 1

srun --label launch_cluster_ddp.sh GPTXL 60 noop 1

srun --label launch_cluster_ddp.sh GPTXXL 60 noop 1


srun --label launch_cluster_ddp.sh GPTSmall 64 noop 1

srun --label launch_cluster_ddp.sh GPTMedium 64 noop 1

srun --label launch_cluster_ddp.sh GPTLarge 64 noop 1

srun --label launch_cluster_ddp.sh GPTXL 64 noop 1

srun --label launch_cluster_ddp.sh GPTXXL 64 noop 1


srun --label launch_cluster_ddp.sh GPTSmall 60 checkpoint 1

srun --label launch_cluster_ddp.sh GPTMedium 60 checkpoint 1

srun --label launch_cluster_ddp.sh GPTLarge 60 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXL 60 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXXL 60 checkpoint 1


srun --label launch_cluster_ddp.sh GPTSmall 64 checkpoint 1

srun --label launch_cluster_ddp.sh GPTMedium 64 checkpoint 1

srun --label launch_cluster_ddp.sh GPTLarge 64 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXL 64 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXXL 64 checkpoint 1


srun --label launch_cluster_ddp.sh GPTSmall 80 checkpoint 1

srun --label launch_cluster_ddp.sh GPTMedium 80 checkpoint 1

srun --label launch_cluster_ddp.sh GPTLarge 80 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXL 80 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXXL 80 checkpoint 1


srun --label launch_cluster_ddp.sh GPTSmall 96 checkpoint 1

srun --label launch_cluster_ddp.sh GPTMedium 96 checkpoint 1

srun --label launch_cluster_ddp.sh GPTLarge 96 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXL 96 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXXL 96 checkpoint 1


srun --label launch_cluster_ddp.sh GPTSmall 112 checkpoint 1

srun --label launch_cluster_ddp.sh GPTMedium 112 checkpoint 1

srun --label launch_cluster_ddp.sh GPTLarge 112 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXL 112 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXXL 112 checkpoint 1


srun --label launch_cluster_ddp.sh GPTSmall 128 checkpoint 1

srun --label launch_cluster_ddp.sh GPTMedium 128 checkpoint 1

srun --label launch_cluster_ddp.sh GPTLarge 128 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXL 128 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXXL 128 checkpoint 1


srun --label launch_cluster_ddp.sh GPTXXXL 1 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXXXL 2 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXXXL 4 checkpoint 1

srun --label launch_cluster_ddp.sh GPTXXXL 8 checkpoint 1



