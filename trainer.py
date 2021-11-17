from argparse import ArgumentParser
from dataclasses import dataclass
from posix import posix_spawn
from typing import Tuple
import os

from models import (
    GPT,
    GPTSmallConfig,
    GPTLargeConfig,
    configure_optimizers,
    sequential_gpt
)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.pipeline.sync import Pipe


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
        # still nees to call to(device) because GPT buffer is still on CPU
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


def train(args):
    # build DDP/Pipeline/FSDP model
    if args.mode == "ddp":
        model = build_ddp_model(args)
    elif args.mode == "pdp":
        model = build_pdp_model(args)

    # build dummy inputs and optimizer
    if args.model.startswith("GPT"):
        inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device="cuda:0")

        opt = configure_optimizers(
            model,
            TrainConfig(
                vocab_size=args.vocab_size,
                block_size=args.block_size,
                batch_size=args.batch_size,
            )
        )
    else:
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

    # warmup
    for i in range(2):
        out = model(inputs)
        loss = out.sum() if isinstance(out, torch.Tensor) else out.local_value().sum()
        loss.backward()
        opt.step()
        opt.zero_grad()

    # measure
    e_pre_step = torch.cuda.Event(enable_timing=True)
    e_post_fwd = torch.cuda.Event(enable_timing=True)
    e_post_bwd = torch.cuda.Event(enable_timing=True)
    e_post_opt = torch.cuda.Event(enable_timing=True)
    for i in range(2):
        e_pre_step.record()
        out = model(inputs)
        loss = out.sum() if isinstance(out, torch.Tensor) else out.local_value().sum()
        e_post_fwd.record()
        loss.backward()
        e_post_bwd.record()
        opt.step()
        opt.zero_grad()
        e_post_opt.record()

        # events can only be consumed when all prior CUDA ops are done.
        # N.B., we might need to store events and only call synchronize
        # after the for loop to avoid killing overlap between opt and next
        # forward
        e_post_opt.synchronize()

        # N.B., this is inaccurate for pipeline, as it only captures CUDA ops
        # on CUDA0. We will need sth like this:
        # https://gist.github.com/mrshenli/8c39c35612218e0d6d772910bca2f737
        print(
            f"Rank {os.getenv('RANK')} Iteration {i}: "
            f"FWD Time = {e_pre_step.elapsed_time(e_post_fwd)}, "
            f"BWD Time = {e_post_fwd.elapsed_time(e_post_bwd)}, "
            f"OPT Time = {e_post_bwd.elapsed_time(e_post_opt)}"
        )


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
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    dist.rpc.init_rpc(f"worker{rank}", rank=rank, world_size=world_size)


def teardown():
    dist.rpc.shutdown()


def main():
    args=parse_args()

    setup(args)
    train(args)
    teardown()

if __name__=="__main__":
    main()