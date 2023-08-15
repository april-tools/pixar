from __future__ import annotations
import os
from os import PathLike
from dataclasses import dataclass, asdict
import json

import torch
from torch import nn

from transformers.utils import logging

from ..text_graph import TGraph

logger = logging.get_logger(__name__)


@dataclass
class DiscriminatorConfig:

    # Model structure
    hidden_size: int = 768
    n_blocks: int = 2
    n_layer_per_bolck: int = 2
    drop_out: float = 0.2
    act: str = 'gelu'

    # Input data
    num_channel: int = 3
    patch_width: int = 16
    patch_height: int = 16

    def save(self, folder: str | PathLike) -> str:
        os.makedirs(folder, exist_ok=False)

        js = json.dumps(asdict(self), indent=2)
        path = os.path.join(folder, 'config.json')
        with open(path, 'w') as fout:
            fout.write(js)
        
        return js
    
    @classmethod
    def load(cls, folder: str | PathLike) -> DiscriminatorConfig:
        with open(os.path.join(folder, 'config.json'), 'r') as fin:
            conf = json.load(fin)
        
        return cls(**conf)


class DiscriminatorLayer(nn.Module):

    def __init__(self, config: DiscriminatorConfig) -> None:
        super().__init__()

        self.ln = nn.Linear(config.hidden_size, config.hidden_size)
        match config.act:
            case 'gelu':
                self.act = nn.GELU()
            case 'relu':
                self.act = nn.ReLU()
            case _:
                raise KeyError(f'Do not support activation type {config.act}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.ln(x))


class DiscriminatorBlock(nn.Module):

    def __init__(self, config: DiscriminatorConfig) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(config.hidden_size)
        self.layers = nn.Sequential(
            *[DiscriminatorLayer(config) for _ in range(config.n_layer_per_bolck)]
        )
        self.dropout = nn.Dropout(config.drop_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        y = self.norm(x)
        y = self.layers(y)
        y = self.dropout(y)

        # residual connection
        y = x + y

        return y


class Discriminator(nn.Module):

    def __init__(self, config: DiscriminatorConfig) -> None:
        super().__init__()
        self.config = config

        self.in_proj = nn.Conv2d(
            in_channels=config.num_channel,
            out_channels=config.hidden_size,
            kernel_size=[config.patch_height, config.patch_width],
            stride=config.patch_width
        )
        self.blocks = nn.Sequential(
            *[DiscriminatorBlock(config) for _ in range(config.n_blocks)]
        )
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward_loss(self, logits: torch.Tensor, mask: torch.Tensor, target: int) -> torch.Tensor:
        """
        logits: [n_batch, n_patch, 2]
        mask:   [n_batch, n_patch]
        target: 1 or 0
        """
        lossfn = nn.CrossEntropyLoss(reduction='none')

        logits = logits.flatten(0, 1)
        mask = mask.flatten()
        target = torch.ones_like(mask, dtype=torch.long) * target

        # Average the loss per patch
        return (lossfn(logits, target) * mask).sum() / mask.sum()

    def forward(self, input: TGraph, target: int) -> TGraph:
        """
        Target is 0 or 1, 1 means real, 0 means fake
        """
        images = input.unsquarelize().value

        inputs_embeds = self.in_proj(images)
        inputs_embeds = inputs_embeds.flatten(2).transpose(1, 2)
        logger.warning_once(f'Discriminator inputs_embeds shape {inputs_embeds.shape}')

        hidden = self.blocks(inputs_embeds)
        logger.warning_once(f'Discriminator inputs_embeds shape {hidden.shape}')

        logits = self.out_proj(hidden)
        loss = self.forward_loss(logits, input.text_mask, target)

        result = TGraph()
        result._value = logits
        result.loss = loss

        return result
    
    def save(self, folder: str | PathLike) -> None:
        self.config.save(folder)
        path = os.path.join(folder, 'pytorch_model.bin')
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, folder: str | PathLike | None = None, config: DiscriminatorConfig | None = None) -> Discriminator:
        conf = DiscriminatorConfig.load(folder) if config is None else config

        model = cls(conf)

        if folder is None:
            logger.warning(f'Initialize new discriminator parameters')
            return model
        
        state_path = os.path.join(folder, 'pytorch_model.bin')
        if os.path.exists(state_path):
            logger.warning(f'Loading discriminator state dict at {state_path}')
            model.load_state_dict(torch.load(state_path))
        else:
            logger.warning(f'Initialize new discriminator parameters')

        return model
