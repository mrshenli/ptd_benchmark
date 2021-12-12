#!/bin/bash

srun --label launch_local_pipeline.sh GPTXXXL 24 4 noop

srun --label launch_local_pipeline.sh GPTXXXL 32 4 noop

srun --label launch_local_pipeline.sh GPTXXXL 40 4 noop

srun --label launch_local_pipeline.sh GPTXXXL 48 4 noop