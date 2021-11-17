"""
This file was borrowed from https://github.com/karpathy/minGPT with modifications.

GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)

class GPTSmallConfig(GPTConfig):
    """ GPT3-small like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768

class GPTLargeConfig(GPTConfig):
    """ GPT3-large like network roughly 760M params """
    n_layer = 24
    n_head = 16
    n_embd = 1536

class GPTXLConfig(GPTConfig):
    """ GPT3-XL like network roughly 1.3B params """
    n_layer = 24
    n_head = 24
    n_embd = 2048

class GPTXXLConfig(GPTConfig):
    """ GPT3-XL like network roughly 2.7B params """
    n_layer = 32
    n_head = 32
    n_embd = 2560

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config, device=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd, device=device)
        self.query = nn.Linear(config.n_embd, config.n_embd, device=device)
        self.value = nn.Linear(config.n_embd, config.n_embd, device=device)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd, device=device)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # TODO: leave buffer on CPU for now, until we can do meta_tensor.to_empty()
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
                 .view(1, 1, config.block_size, config.block_size)
        )
        self.n_head = config.n_head

    def reset_parameters(self):
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class EmbeddingStem(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()

        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd, device=device)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd, device=device))
        self.drop = nn.Dropout(config.embd_pdrop)
        self.block_size = config.block_size

    def reset_parameters(self):
        self.tok_emb.reset_parameters()

    def forward(self, idx):
        b, t = idx.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."

        token_embeddings = self.tok_emb(idx) # each index maps to a (learnable) vector
        position_embeddings = self.pos_emb[:, :t, :] # each position maps to a (learnable) vector
        return self.drop(token_embeddings + position_embeddings)


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, device=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, device=device)
        self.ln2 = nn.LayerNorm(config.n_embd, device=device)
        self.attn = CausalSelfAttention(config, device=device)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, device=device),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, device=device),
            nn.Dropout(config.resid_pdrop),
        )

    def reset_parameters(self):
        self.attn.reset_parameters()
        for _, m in self.named_modules():
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, config, device=None):
        super().__init__()

        # input embedding stem
        self.emb_stem = EmbeddingStem(config, device=device)
        # transformer
        self.blocks = nn.Sequential(
            *[Block(config, device=device) for _ in range(config.n_layer)]
        )
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd, device=device)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False, device=device)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, idx):
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        return self.head(x)


def configure_optimizers(model, train_config):
    """
    This long function is unfortunately doing something very simple and is being very defensive:
    We are separating out all parameters of the model into two buckets: those that will experience
    weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
    We are then returning the PyTorch optimizer object.
    """

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            elif pn.endswith('pos_emb') and isinstance(m, EmbeddingStem):
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
    return optimizer


def sequential_gpt(config, devices):
    """
    Returns an ``nn.Sequential`` of GPT model balanced across the given devices.
    N.B.: this function does not dedup devices.
    """
    # put all layers into a list
    emb_stem = EmbeddingStem(config, device="meta")
    blocks = [Block(config, device="meta") for _ in range(config.n_layer)]
    ln_f = nn.LayerNorm(config.n_embd, device="meta")
    head = nn.Linear(config.n_embd, config.vocab_size, bias=False, device="meta")

    layers = [emb_stem, *blocks, ln_f, head]

    # partition layers into the given devices
    def numel(layer):
        return sum([p.numel() for p in layer.parameters()])

    total_numel = sum([numel(layer) for layer in layers])
    phase_numel = total_numel // len(devices)
    delim_numel = phase_numel
    accum_numel = 0

    # seal one pipeline phase when its numel is larger than phase_numel
    phases = [[]]
    for layer in layers:
        phases[-1].append(layer)
        accum_numel += numel(layer)
        if accum_numel > delim_numel:
            delim_numel += phase_numel
            phases.append([])

    # pack all remaining layers into the last phase
    while len(phases) > len(devices):
        phases[-2].extend(phases[-1])
        phases.pop()

    for i, phase in enumerate(phases):
        for layer in phase:
            layer.to_empty(device=torch.device(devices[i])).reset_parameters()

    # create nn.Sequential
    return nn.Sequential(*[nn.Sequential(*phase) for phase in phases])