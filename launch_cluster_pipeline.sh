#! /bin/bash

source /home/shenli_fb_com/.bashrc

ulimit -S -c unlimited

torchrun --nnodes=16 --nproc_per_node=2 --rdzv_id=200 --rdzv_backend=c10d --rdzv_endpoint="shen-64-a2-highgpu-8g-compute-0-0:5679" trainer.py --ndevice_per_proc=4 --mode=pdp --model=GPT13B --batch_size=$1 --chunks=$2 --activation=$3 --vocab_size=50000 --block_size=256 --dtype=fp16