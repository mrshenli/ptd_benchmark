#! /bin/bash

source /home/shenli_fb_com/.bashrc

ulimit -S -c unlimited

torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=200 --rdzv_backend=c10d --rdzv_endpoint="shen-64-a2-highgpu-8g-compute-0-0:5679" trainer.py --ndevice_per_proc=2 --mode=pdp --model=$1 --batch_size=$2 --chunks=$3 --activation=$4 --vocab_size=50000 --block_size=256 --dtype=fp16 --wrap=transformer