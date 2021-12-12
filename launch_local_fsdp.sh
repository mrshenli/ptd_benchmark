#!/bin/bash

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint="localhost:5699" trainer.py  --mode=fsdp --model=$1 --batch_size=$2 --vocab_size=50000 --block_size=256 --activation=$3 --dtype=fp16 --wrap=transformer
