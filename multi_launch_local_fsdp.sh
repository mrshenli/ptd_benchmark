#!/bin/bash

srun --label launch_local_fsdp.sh GPTSmall 16 noop

srun --label launch_local_fsdp.sh GPTSmall 16 checkpoint

srun --label launch_local_fsdp.sh GPTSmall 16 offload

srun --label launch_local_fsdp.sh GPTLarge 16 noop

srun --label launch_local_fsdp.sh GPTLarge 16 checkpoint

srun --label launch_local_fsdp.sh GPTLarge 16 offload

srun --label launch_local_fsdp.sh GPTXL 16 noop

srun --label launch_local_fsdp.sh GPTXL 16 checkpoint

srun --label launch_local_fsdp.sh GPTXL 16 offload

srun --label launch_local_fsdp.sh GPTXXL 16 noop

srun --label launch_local_fsdp.sh GPTXXL 16 checkpoint

srun --label launch_local_fsdp.sh GPTXXL 16 offload

srun --label launch_local_fsdp.sh GPTXXXL 16 noop

srun --label launch_local_fsdp.sh GPTXXXL 16 checkpoint

srun --label launch_local_fsdp.sh GPTXXXL 16 offload

srun --label launch_local_fsdp.sh GPT13B 16 noop

srun --label launch_local_fsdp.sh GPT13B 16 checkpoint

srun --label launch_local_fsdp.sh GPT13B 16 offload


srun --label launch_local_fsdp.sh GPTXL 16 noop

srun --label launch_local_fsdp.sh GPTXL 16 checkpoint

srun --label launch_local_fsdp.sh GPTXL 16 offload

srun --label launch_local_fsdp.sh GPTXXL 16 noop

srun --label launch_local_fsdp.sh GPTXXL 16 checkpoint

srun --label launch_local_fsdp.sh GPTXXL 16 offload

srun --label launch_local_fsdp.sh GPTXXXL 16 noop

srun --label launch_local_fsdp.sh GPTXXXL 16 checkpoint

srun --label launch_local_fsdp.sh GPTXXXL 16 offload

srun --label launch_local_fsdp.sh GPT13B 16 noop

srun --label launch_local_fsdp.sh GPT13B 16 checkpoint

srun --label launch_local_fsdp.sh GPT13B 16 offload


srun --label launch_local_ddp.sh GPTSmall 16 noop

srun --label launch_local_fsdp.sh GPTLarge 16 noop

srun --label launch_local_fsdp.sh GPTXL 16 noop

srun --label launch_local_fsdp.sh GPTXXL 16 noop
