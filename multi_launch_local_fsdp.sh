#!/bin/bash

srun --label launch_local_fsdp.sh GPTXXXL 4 checkpoint

srun --label launch_local_fsdp.sh GPTXXXL 6 checkpoint

srun --label launch_local_fsdp.sh GPTXXXL 8 checkpoint

srun --label launch_local_fsdp.sh GPTXXXL 10 checkpoint

srun --label launch_local_fsdp.sh GPTXXXL 12 checkpoint

srun --label launch_local_fsdp.sh GPTXXXL 14 checkpoint

srun --label launch_local_fsdp.sh GPTXXXL 16 checkpoint

srun --label launch_local_fsdp.sh GPTXXXL 18 checkpoint

srun --label launch_local_fsdp.sh GPTXXXL 20 checkpoint

