#!/bin/bash


srun --label launch_cluster_fsdp.sh GPT13B 256 offload

srun --label launch_cluster_fsdp.sh GPT39B 256 offload



srun --label launch_cluster_fsdp.sh GPT76B 8 offload

srun --label launch_cluster_fsdp.sh GPT76B 16 offload

srun --label launch_cluster_fsdp.sh GPT76B 32 offload

srun --label launch_cluster_fsdp.sh GPT76B 64 offload

srun --label launch_cluster_fsdp.sh GPT76B 128 offload



srun --label launch_cluster_fsdp.sh GPT100B 8 offload

srun --label launch_cluster_fsdp.sh GPT100B 16 offload

srun --label launch_cluster_fsdp.sh GPT100B 32 offload

srun --label launch_cluster_fsdp.sh GPT100B 64 offload

srun --label launch_cluster_fsdp.sh GPT100B 128 offload


srun --label launch_cluster_fsdp.sh GPT175B 8 offload

srun --label launch_cluster_fsdp.sh GPT175B 1 offload

srun --label launch_cluster_fsdp.sh GPT175B 2 offload

srun --label launch_cluster_fsdp.sh GPT175B 4 offload







