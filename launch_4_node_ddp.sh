#! /bin/bash

source /home/shenli_fb_com/.bashrc

ulimit -S -c unlimited

torchrun --nnodes=4 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint="10.128.0.38:5678" trainer.py
