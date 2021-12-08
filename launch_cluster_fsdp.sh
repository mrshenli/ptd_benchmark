#! /bin/bash

source /home/shenli_fb_com/.bashrc

ulimit -S -c unlimited -n unlimited

torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint="shen-64-a2-highgpu-8g-compute-0-0:6789" trainer.py --mode=fsdp --model=GPTXXXL --batch_size=8 --vocab_size=50000 --block_size=256 --dtype=fp16 --activation=offload
