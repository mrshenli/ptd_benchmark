#!/bin/bash


srun --label launch_local_fsdp.sh GPT13B 8 checkpoint

srun --label launch_local_fsdp.sh GPT13B 8 offload