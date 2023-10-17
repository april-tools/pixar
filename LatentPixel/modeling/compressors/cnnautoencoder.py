from __future__ import annotations
from dataclasses import dataclass, asdict
import os
from os import PathLike
import json
import math

import torch
from torch import nn

from ...text_graph import TGraph
from .cnn_blocks import (
    CNNDecoder,
    CNNEncoder
)
from .compressor import Compressor, CompressorConfig


@dataclass
class CNNAutoencoderConfig(CompressorConfig):
    
    hidden_channels: int = 128
    num_res: int = 1
    dropout: float = 0.2
    norm_groups: int = 32


class CNNAutoencoder(Compressor):
    
    ckpt_name = 'CNNAutoencoder.pt'
    
    def init(self, config: CNNAutoencoderConfig) -> Compressor:
        self.config = config
        self.latent_channels = config.hidden_channels
        
        self.encoder = CNNEncoder(
            in_channels=config.num_channel,
            hidden_channels=config.hidden_channels,
            num_downsample=int(math.log2(config.compress_ratio)),
            num_res=config.num_res,
            hidden_dim=config.num_latent_channel,
            dropout=config.dropout,
            norm_groups=config.norm_groups
        )
        
        self.decoder = CNNDecoder(
            target_channels=config.num_channel,
            hidden_channels=config.hidden_channels,
            num_upsample=int(math.log2(config.compress_ratio)),
            num_res=config.num_res,
            hidden_dim=config.num_latent_channel,
            dropout=config.dropout,
            norm_groups=config.norm_groups
        )
        
        if self.config.binary:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.MSELoss()
        
        return self
    
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward(x)

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder.forward(z)
    
    def load(self, path: str | PathLike) -> Compressor:
        self.config = CNNAutoencoderConfig.load(path)
        self.init(self.config)
        
        statedict = torch.load(
            os.path.join(path, self.ckpt_name), 
            map_location='cpu'
        )
        self.load_state_dict(statedict)
        
        return self
    
    def save(self, path: str | PathLike) -> None:
        self.config.save(path)
        torch.save(self.state_dict(), os.path.join(path, self.ckpt_name))
        return
            
    def forward_loss(self, preds: TGraph, target: TGraph, hidden: TGraph) -> torch.Tensor:
        if self.config.binary:
            return self.loss_fn.forward(preds.value.view(-1), target.value.view(-1).float())
        
        return self.loss_fn.forward(preds.value, target=target.value)
