from dataclasses import dataclass
from typing import Tuple

@dataclass
class TrainConfig:
    weight_decay : float = 0.01
    learning_rate : float  = 0.01
    betas : Tuple[float, float] = (0.9, 0.999)
    vocab_size : int = 3072
    block_size : int = 128
    batch_size : int = 10
