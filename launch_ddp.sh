torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=100 --rdzv_endpoint="localhost:5678" trainer.py