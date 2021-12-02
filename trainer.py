from argparse import ArgumentParser
from dataclasses import dataclass
from datetime import datetime
from posix import posix_spawn
from typing import Tuple
import gc
import os

from models import (
    GPT,
    GPTSmallConfig,
    GPTMediumConfig,
    GPTLargeConfig,
    GPTXLConfig,
    GPTXXLConfig,
    GPTXXXLConfig,
    GPT13BConfig,
    GPT175BConfig,
    ShardedGPT,
    sequential_gpt
)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.pipeline.sync import Pipe
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler


@dataclass
class TrainConfig:
    weight_decay : float = 0.01
    learning_rate : float  = 0.01
    betas : Tuple[float, float] = (0.9, 0.999)
    vocab_size : int = 3072
    block_size : int = 128
    batch_size : int = 10


def parse_args():
    parser = ArgumentParser(description="PyTorch Data Parallel Experiments")

    parser.add_argument(
        "--model",
        type=str,
        default="GPTSmall",
        help="specify the model to experiment with, e.g., GPTSmall, GPTLarge, ResNet50, etc."
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="ddp",
        help="ddp - DDP, pdp - Pipline and DDP, fsdp - FullyShardedDataParallel"
    )

    parser.add_argument(
        "--ndevice_per_proc",
        type=int,
        default=1,
        help="number of devices used by each process, only applies to pdp"
    )

    parser.add_argument(
        "--vocab_size",
        type=int,
        default=3072,
        help="vocabulary size"
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=128,
        help="the max number of tokens in one sample"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="input batch size"
    )

    parser.add_argument(
        "--chunks",
        type=int,
        default=2,
        help="the number of micro-batches for pipeline parallelism"
    )

    return parser.parse_args()


def print_peak_memory(prefix, device):
    rank = int(os.getenv("RANK"))
    if rank == 0:
        print(f"{prefix}: {torch.cuda.max_memory_allocated(device) // 1e6}MB ")


def get_gpt_config(args):
    assert args.model.startswith("GPT")
    config_class_name = args.model + "Config"
    assert config_class_name in globals()
    return globals()[config_class_name](args.vocab_size, args.block_size)


def build_ddp_model(args):
    # since we have set CUDA_VISIBLE_DEVICES, each process only sees one device,
    # hence using "cuda:0" here
    device = torch.device("cuda:0")

    # get local model, for DDP, the entire model resides on cuda:0
    if args.model.startswith("GPT"):
        # still needs to call to(device) because GPT buffer is still on CPU
        local_model = GPT(get_gpt_config(args), device=device).to(device)
    elif args.model.startswith("ResNet"):
        # TODO
        raise ValueError("ResNet Model Not Implementated")
    else:
        raise ValueError(f"Unrecognized Model {args.model}")

    ddp = DistributedDataParallel(
        local_model,
        device_ids=[device],
        # not 100% sure about this. Look like the buffer is only used as a mask.
        # So, we don't need to synchronize it across processes?
        broadcast_buffers=False,
        gradient_as_bucket_view=True,
    )
    ddp._set_static_graph()
    return ddp


def build_pdp_model(args):
    devices = [f"cuda:{d}" for d in range(args.ndevice_per_proc)]
    if not args.model.startswith("GPT"):
        raise ValueError("We shouldn't need pipeline for ResNet models")
    gpt = sequential_gpt(get_gpt_config(args), devices=devices)
    pipe = Pipe(gpt, chunks=args.chunks)
    ddp = DistributedDataParallel(
        pipe,
        broadcast_buffers=False,
        gradient_as_bucket_view=True
    )
    ddp._set_static_graph()
    return ddp


def build_fsdp_model(args):
    device = torch.device("cuda:0")

    if args.model.startswith("GPT"):
        # still needs to call to(device) because GPT buffer is still on CPU
        return ShardedGPT(get_gpt_config(args), device=device).to(device)
    elif args.model.startswith("ResNet"):
        # TODO
        raise ValueError("ResNet Model Not Implemented")
    else:
        raise ValueError(f"Unrecognized Model {args.model}")


def my_tensorboard_trace_handler(dir_name: str, rank, worker_name = None, use_gzip: bool = False):
    if 0 < rank and rank < 4:
        return tensorboard_trace_handler(dir_name, worker_name, use_gzip)
    else:
        return None


def train(args):
    rank = int(os.getenv("RANK"))
    ws = int(os.getenv("WORLD_SIZE"))

    if rank == 0:
        print(f"# of visible devices = {torch.cuda.device_count()}", flush=True)

    # build DDP/Pipeline/FSDP model
    if args.mode == "ddp":
        model = build_ddp_model(args)
    elif args.mode == "pdp":
        model = build_pdp_model(args)
    elif args.mode == "fsdp":
        model = build_fsdp_model(args)
        print_peak_memory("Memory allocation after model init", "cuda:0")

    # build dummy inputs
    if "GPT" in args.model:
        inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device="cuda:0")
    else:
        raise ValueError("Inputs not implemented for non-GPT models")

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    print_peak_memory("Memory allocation after optimizer", "cuda:0")

    # warmup
    for i in range(4):
        out = model(inputs)
        print_peak_memory(f"Step {i} Memory allocation after forward", "cuda:0")
        loss = out.sum() if isinstance(out, torch.Tensor) else out.local_value().sum()
        loss.backward()
        print_peak_memory(f"Step {i} Memory allocation after backward", "cuda:0")
        del loss
        del out
        print_peak_memory(f"Step {i} Memory allocation after del loss", "cuda:0")
        opt.step()
        print_peak_memory(f"Step {i} Memory allocation after optimizer", "cuda:0")
        opt.zero_grad()

    now = datetime.now()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=my_tensorboard_trace_handler(f"tb/{now.strftime('%Y_%m_%d_%H_%M_%S')}", rank, use_gzip=True)
    ) as prof:
        for i in range(4):
            out = model(inputs)
            loss = out.sum() if isinstance(out, torch.Tensor) else out.local_value().sum()
            loss.backward()
            del loss
            del out
            gc.collect()
            opt.step()
            opt.zero_grad()

    if rank == 0:
        prof.export_chrome_trace(f"chrome/{args.mode}_{args.model}_ws{ws}_bs{args.batch_size}_vs{args.vocab_size}_blk{args.block_size}.json")


def setup(args):
    if args.mode in ["ddp", "fsdp"]:
        local_rank = os.getenv("LOCAL_RANK")
        # set the visible devices so that each DDP process only sees one CUDA device
        # N.B.: this has to be done before using any CUDA API from torch
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{local_rank}"
    elif args.mode == "pdp":
        local_rank = int(os.getenv("LOCAL_RANK"))
        # N.B.: we cannot call torch.cuda.device_count() to figure out the
        # number of devices and then divide it by nproc_per_node, because calling
        # torch.cuda.* would already initialize cuda and read CUDA_VISIBLE_DEVICES
        ndevice_per_proc = int(args.ndevice_per_proc)
        start_device = local_rank * ndevice_per_proc
        end_device = (local_rank + 1) * ndevice_per_proc
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(start_device, end_device)))
    else:
        raise ValueError(f"Unrecognized mode {args.mode}")


    world_size = int(os.getenv("WORLD_SIZE"))
    rank = int(os.getenv("RANK"))
    if rank == 0:
        print(f"parsed args {args}")
        print(f"World size is {world_size}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if args.mode == "pdp":
        dist.rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)


def teardown(args):
    if args.mode == "pdp":
        dist.rpc.shutdown()
    rank = int(os.getenv("RANK"))
    if rank == 0:
        print("Finished")


def main():
    args=parse_args()

    setup(args)
    train(args)
    teardown(args)

if __name__=="__main__":
    main()
