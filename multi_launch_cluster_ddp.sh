#!/bin/bash

srun --label launch_cluster_ddp.sh GPTXXL 2 noop

srun --label launch_cluster_ddp.sh GPTXXL 4 noop

srun --label launch_cluster_ddp.sh GPTXXL 6 noop

srun --label launch_cluster_ddp.sh GPTXXL 8 noop

srun --label launch_cluster_ddp.sh GPTXL 2 noop

srun --label launch_cluster_ddp.sh GPTXL 4 noop

srun --label launch_cluster_ddp.sh GPTXL 6 noop

srun --label launch_cluster_ddp.sh GPTXL 8 noop

srun --label launch_cluster_ddp.sh GPTXL 12 noop

srun --label launch_cluster_ddp.sh GPTXL 14 noop

srun --label launch_cluster_ddp.sh GPTXL 16 noop

srun --label launch_cluster_ddp.sh GPTLarge 2 noop

srun --label launch_cluster_ddp.sh GPTLarge 4 noop

srun --label launch_cluster_ddp.sh GPTLarge 6 noop

srun --label launch_cluster_ddp.sh GPTLarge 8 noop

srun --label launch_cluster_ddp.sh GPTLarge 12 noop

srun --label launch_cluster_ddp.sh GPTLarge 14 noop

srun --label launch_cluster_ddp.sh GPTLarge 16 noop

srun --label launch_cluster_ddp.sh GPTLarge 18 noop

srun --label launch_cluster_ddp.sh GPTLarge 20 noop

srun --label launch_cluster_ddp.sh GPTSmall 2 noop

srun --label launch_cluster_ddp.sh GPTSmall 4 noop

srun --label launch_cluster_ddp.sh GPTSmall 6 noop

srun --label launch_cluster_ddp.sh GPTSmall 8 noop

srun --label launch_cluster_ddp.sh GPTSmall 12 noop

srun --label launch_cluster_ddp.sh GPTSmall 14 noop

srun --label launch_cluster_ddp.sh GPTSmall 16 noop

srun --label launch_cluster_ddp.sh GPTSmall 18 noop

srun --label launch_cluster_ddp.sh GPTSmall 20 noop


