from models import (
    GPT,
    GPTSmallConfig,
    GPTLargeConfig,
    configure_optimizers,
    sequential_gpt
)
from trainer import TrainConfig

import os

import torch
from torch.distributed.pipeline.sync import Pipe

def test_gpt_small():
    config = TrainConfig()

    gpt = GPT(GPTSmallConfig(vocab_size=config.vocab_size, block_size=config.block_size)).cuda()
    opt = configure_optimizers(gpt, config)

    x = torch.randint(0, config.vocab_size, (config.batch_size, config.block_size)).cuda()
    gpt(x).sum().backward()
    opt.step()

    print("test GPT3-Small done")


def test_gpt_large():
    config = TrainConfig()

    gpt = GPT(GPTLargeConfig(vocab_size=config.vocab_size, block_size=config.block_size)).cuda()
    opt = configure_optimizers(gpt, config)

    x = torch.randint(0, config.vocab_size, (config.batch_size, config.block_size)).cuda()
    gpt(x).sum().backward()
    opt.step()

    print("test GPT3-Large done")


def _test_sequential_gpt(train_config, gpt_config):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "5678"
    torch.distributed.rpc.init_rpc("worker0", world_size=1, rank=0)

    train_config = TrainConfig()
    gpt = sequential_gpt(gpt_config, devices = ["cuda:0", "cuda:1"])
    opt = configure_optimizers(gpt, train_config)

    device_numel = [0, 0]
    for _, m in gpt.named_modules():
        for p in m.parameters():
            device_numel[p.device.index] += p.numel()

    assert max(device_numel) / min(device_numel) < 1.5

    pipe = Pipe(gpt, chunks=2)

    x = torch.randint(0, train_config.vocab_size, (train_config.batch_size, train_config.block_size)).cuda(0)
    pipe(x).local_value().sum().backward()
    opt.step()

    torch.distributed.rpc.shutdown()

    print("test pipelined GPT done")


def test_sequential_gpt_small():
    train_config = TrainConfig()
    gpt_config = GPTSmallConfig(vocab_size=train_config.vocab_size, block_size=train_config.block_size)
    _test_sequential_gpt(train_config, gpt_config)


def test_sequential_gpt_large():
    train_config = TrainConfig()
    gpt_config = GPTLargeConfig(vocab_size=train_config.vocab_size, block_size=train_config.block_size)
    _test_sequential_gpt(train_config, gpt_config)


test_gpt_small()
test_gpt_large()
test_sequential_gpt_small()
test_sequential_gpt_large()
