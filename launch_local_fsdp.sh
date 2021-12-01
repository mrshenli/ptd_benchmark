torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=100 --rdzv_endpoint="localhost:5678" trainer.py  --mode=fsdp --model=GPTXXXL --batch_size=1 --vocab_size=3072 --block_size=64
