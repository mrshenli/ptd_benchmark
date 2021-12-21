#! /bin/bash

source /home/shenli_fb_com/.bashrc

ulimit -S -c unlimited -n unlimited

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=600 --rdzv_backend=c10d --rdzv_endpoint="127.0.0.1:6789" trainer.py --mode=fsdp --model=$1 --batch_size=$2 --vocab_size=50000 --block_size=256 --dtype=fp16 --activation=$3 --wrap=linear --cpu_offload=True
