from argparse import Action, ArgumentParser
import contextlib
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from pathlib import Path
from posix import posix_spawn
import psutil
from statistics import stdev
from typing import Tuple
import gc
import os
import time

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
    GPT1TConfig,
    ShardedGPT,
    sequential_gpt
)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.pipeline.sync import Pipe
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from torch.distributed._fsdp.wrap import enable_wrap, wrap
from torch.distributed._fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from torch.distributed._fsdp.fully_sharded_data_parallel import BackwardPrefetch_
from fairscale.nn.data_parallel import FullyShardedDataParallel as fairscale_fsdp

@dataclass
class TrainConfig:
    weight_decay : float = 0.01
    learning_rate : float  = 0.01
    betas : Tuple[float, float] = (0.9, 0.999)
    vocab_size : int = 3072
    block_size : int = 128
    batch_size : int = 10

class ParseDType(Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        super(ParseDType, self).__init__(option_strings, dest, **kwargs)
        self.str_to_dtype = {
            "fp32" : torch.float32,
            "fp16" : torch.float16,
        }

    def __call__(self, parser, namespace, values, option_string=None, first=[True]):
        if values in self.str_to_dtype:
            namespace.dtype = self.str_to_dtype[values]
        else:
            raise ValueError(f"cannot parse {values}")


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
        "--dtype",
        type=str,
        action=ParseDType,
        default=torch.float32,
        help="Tensor dtype"
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

    parser.add_argument(
        "--activation",
        type=str,
        default="noop",
        help=(
            "toggles the modes among "
            "noop: keep activation in GPU "
            "checkpoint: checkpoint inner activation and recompute"
            "offload: checkpoint + offload outer activation to CPU"
        )
    )

    parser.add_argument(
        "--profile",
        type=bool,
        default=False,
        help="enable profiling the iterations"
    )

    parser.add_argument(
        "--cpu-offload",
        type=bool,
        default=False,
        help="enable cpu offload params for FSDP"
    )

    parser.add_argument(
        "--prefetch",
        type=str,
        default="noop",
        help=(
            "toggles the modes among "
            "noop: no prefetching in the backward "
            "prehook: prefetching in the backward pre hook"
            "posthook: prefetching in the backward post hook"
        )
    )

    parser.add_argument(
        "--version",
        type=str,
        default="pytorch",
        help=(
            "toggles the modes among "
            "pytorch: train with pytorch fsdp"
            "fairscale: train with fairscale fsdp"
        )
    )

    return parser.parse_args()


def print_memory_summary(prefix, device):
    rank = int(os.getenv("RANK"))
    if rank == 0:
        print(f"{prefix}, GPU memory allocation: {torch.cuda.max_memory_allocated(device) // 1e9}GB "
              f"CPU used memory percent: {psutil.virtual_memory().percent}, "
              f"CPU memory available: {psutil.virtual_memory().available // 1e9}GB, ")
        torch.cuda.reset_peak_memory_stats(device)

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
    gpt = sequential_gpt(get_gpt_config(args), devices=devices, dtype=args.dtype)
    pipe = Pipe(gpt, chunks=args.chunks)
    ddp = DistributedDataParallel(
        pipe,
        broadcast_buffers=False,
        gradient_as_bucket_view=True
    )
    ddp._set_static_graph()
    return ddp


def build_fsdp_model(args):
    rank = int(os.getenv("RANK"))

    device = torch.device("cuda:0")

    cpu_offload_config = None
    if args.cpu_offload:
        if rank == 0: 
            print("Enabling cpu offloading")
        cpu_offload_config = CPUOffload(offload_params=True)
    
    backward_prefetch = None
    if args.prefetch == "prehook":
        backward_prefetch = BackwardPrefetch_.BACKWARD_PRE
    elif args.prefetch == "posthook":
        backward_prefetch = BackwardPrefetch_.BACKWARD_POST

    if args.model.startswith("GPT"):
        # still needs to call to(device) because GPT buffer is still on CPU
        if args.version == "pytorch":
            with enable_wrap(wrapper_cls=FSDP, cpu_offload=cpu_offload_config, backward_prefetch=backward_prefetch):
                if args.cpu_offload:
                    return wrap(ShardedGPT(get_gpt_config(args), device=device, dtype=args.dtype, activation=args.activation))
                else:
                    return wrap(ShardedGPT(get_gpt_config(args), device=device, dtype=args.dtype, activation=args.activation)).to(device)
        elif args.version == "fairscale":
            print("using fairscale fsdp")
            with enable_wrap(wrapper_cls=fairscale_fsdp, move_params_to_cpu=args.cpu_offload, compute_dtype=torch.float16):
                if args.cpu_offload:
                    return wrap(ShardedGPT(get_gpt_config(args), device=device, dtype=args.dtype, activation=args.activation, version=args.version, cpu_offload=True).cpu())
                else:
                    return wrap(ShardedGPT(get_gpt_config(args), device=device, dtype=args.dtype, activation=args.activation, version=args.version)).to(device)

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


def calc_flop(args):
    B = args.batch_size
    s = args.block_size
    conf = get_gpt_config(args)
    l = conf.n_layer
    h = conf.n_embd
    V = conf.vocab_size
    return 96 * B * s * l * h * h * (1 + s/6/h + V/16/l/h)


def train(args):
    rank = int(os.getenv("RANK"))
    ws = int(os.getenv("WORLD_SIZE"))

    def sync_all_device():
        # setup() has already configured CUDA_VISIBLE_DEVICES such that each
        # process exclusively works on its own set of devices. So it's safe to
        # do device sync here
        for d in range(torch.cuda.device_count()):
            torch.cuda.synchronize(d)

    FLOP = calc_flop(args)

    if rank == 0:
        print(f"# of visible devices = {torch.cuda.device_count()}", flush=True)
        print(f"TFLOP per iteration {FLOP // 10**12}")

    init_start_event = torch.cuda.Event(enable_timing=True)
    init_end_event = torch.cuda.Event(enable_timing=True)
    before_forward_event = torch.cuda.Event(enable_timing=True)
    after_forward_event = torch.cuda.Event(enable_timing=True)
    after_backward_event = torch.cuda.Event(enable_timing=True)
    after_step_event = torch.cuda.Event(enable_timing=True)
    after_zero_grad_event = torch.cuda.Event(enable_timing=True)

    init_start_event.record()

    now = datetime.now()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # record_shapes=True, # Causes seg fault in export_chrome_trace
        # with_stack=True, # Causes seg fault with EFA
        # with_flops=True, # Causes seg fault in export_chrome_trace
        on_trace_ready=my_tensorboard_trace_handler(f"tb/{now.strftime('%Y_%m_%d_%H_%M_%S')}", rank, use_gzip=True)
    ) if args.profile else contextlib.nullcontext() as prof:
        # build DDP/Pipeline/FSDP model
        if args.mode == "ddp":
            model = build_ddp_model(args)
        elif args.mode == "pdp":
            model = build_pdp_model(args)
        elif args.mode == "fsdp":
            model = build_fsdp_model(args)

    init_end_event.record()
    sync_all_device()
    dist.barrier()

    if rank == 0:
        print(f"Building model time: {init_start_event.elapsed_time(init_end_event) / 1000}sec")
        print(f"{model}")

    print_memory_summary("After model init", "cuda:0")

    # build dummy inputs
    if "GPT" in args.model:
        inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.block_size), device="cuda:0")
    else:
        raise ValueError("Inputs not implemented for non-GPT models")

    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    print_memory_summary("After optimizer", "cuda:0")

    # warmup
    for i in range(4):
        out = model(inputs)
        print_memory_summary(f"Step {i} After forward", "cuda:0")
        loss = out.sum() if isinstance(out, torch.Tensor) else out.local_value().sum()
        loss.backward()
        print_memory_summary(f"Step {i} After backward", "cuda:0")
        del loss
        del out
        print_memory_summary(f"Step {i} After del loss", "cuda:0")
        opt.step()
        print_memory_summary(f"Step {i} After optimizer", "cuda:0")
        opt.zero_grad()
        print_memory_summary(f"Step {i} After zero grad", "cuda:0")

    # make sure all pending warm up ops are done
    sync_all_device()
    dist.barrier()

    tik = time.time()

    now = datetime.now()
    n_iters = 4
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # record_shapes=True, # Causes seg fault in export_chrome_trace
        # with_stack=True, # Causes seg fault with EFA
        # with_flops=True, # Causes seg fault in export_chrome_trace
        record_shapes=False,
        with_stack=False,
        with_flops=False,
        on_trace_ready=my_tensorboard_trace_handler(f"tb/{now.strftime('%Y_%m_%d_%H_%M_%S')}", rank, use_gzip=True)
    ) if args.profile else contextlib.nullcontext() as prof:
        for i in range(n_iters):
            before_forward_event.record()           
            out = model(inputs)
            after_forward_event.record()
            loss = out.sum() if isinstance(out, torch.Tensor) else out.local_value().sum()
            loss.backward()
            after_backward_event.record()
            del loss
            del out
            gc.collect()
            opt.step()
            after_step_event.record()
            for param in model.parameters():
                param.grad = None
            after_zero_grad_event.record()
            torch.cuda.synchronize()
            fwd_list = []
            bwd_list = []
            opt_list = []
            total_latency = before_forward_event.elapsed_time(after_zero_grad_event) / 1000
            forward_time = before_forward_event.elapsed_time(after_forward_event) / 1000
            backward_time = after_forward_event.elapsed_time(after_backward_event) / 1000
            step_time = after_backward_event.elapsed_time(after_step_event) / 1000
            zero_grad_time = after_step_event.elapsed_time(after_zero_grad_event) / 1000
            fwd_list.append(forward_time * 100.0 / total_latency)
            bwd_list.append(backward_time * 100.0 / total_latency)
            opt_list.append((step_time + zero_grad_time) * 100.0 / total_latency)
            if rank == 0:
                print(f"train {i}th step, total_latency: {total_latency}sec, "
                      f"forward_time: {forward_time}sec, "
                      f"backward_time: {backward_time}sec, "
                      f"step_time: {step_time}sec, "
                      f"zero_grad_time: {zero_grad_time}sec, "
                      f"TFLOP/s/GPU: {FLOP/10**12/total_latency}")


    sync_all_device()
    tok = time.time()
    delays = [None for _ in range(ws)]
    torch.distributed.all_gather_object(delays, (tok-tik) / n_iters)
    tflops_gpu = FLOP / 10**12 * np.reciprocal(np.array(delays))

    if rank == 0:
        name = (
            f"{args.mode}_{args.model}_"
            f"ws{ws}_bs{args.batch_size}_"
            f"vs{args.vocab_size}_blk{args.block_size}_"
            f"{args.activation}_"
            f"fp{str(args.dtype)[-2:]}"
        )

        if args.mode == "pdp":
            name += f"_ck{args.chunks}_nd{args.ndevice_per_proc}"

        Path("delay").mkdir(parents=True, exist_ok=True)
        fout = open(f"delay/{name}.txt", "w")
        fout.write(f"delays = {sum(delays) / len(delays):.2f} ({stdev(delays):.2f})\n")
        fout.write(f"tflops/gpu = {sum(tflops_gpu) / len(tflops_gpu):.2f} ({stdev(tflops_gpu):.2f})\n")
        mem = max([torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
        fout.write(f"max mem = {mem // 1e9}GB\n")
        fout.write(f"max cpu mem percent = {psutil.virtual_memory().percent}\n")
        fout.write(f"forward percent = {sum(fwd_list) / len(fwd_list):.2f}, backward percent = {sum(bwd_list) / len(bwd_list):.2f}, optimizer percent = {sum(opt_list) / len(opt_list):.2f}")
        fout.close()

        if args.profile:
            Path("chrome").mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(f"chrome/{name}.json.gz")

    dist.barrier()


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
        print(f"PyTorch version {torch.__version__}")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if args.mode == "pdp":
        # use fake RPC gang for local pipeline
        options = dist.rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=8,
            rpc_timeout=120, # 20 second timeout
            init_method=f"tcp://localhost:{7777 + rank}",
        )
        dist.rpc.init_rpc(
            f"worker{rank}",
            rank=0,
            world_size=1,
            rpc_backend_options=options,
        )


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
