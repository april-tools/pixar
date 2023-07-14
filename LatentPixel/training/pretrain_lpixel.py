import os

import torch

from .train_config import ExpConfig

def init_exp(config: ExpConfig, rank: int, world_size: int, device: torch.device):
    pass
