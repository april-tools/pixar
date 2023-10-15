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
from .compressor import Compressor


@dataclass
class CNNAutoencoderConfig:
    
    compress_ratio: int = 4
    in_channels: int = 3
    hidden_channels: int = 128
    num_res: int = 1
    hidden_dim: int = 4
    dropout: float = 0.2
    norm_groups: int = 32
    binary: bool = False
    
    def save(self, folder: str | PathLike) -> str:
        os.makedirs(folder, exist_ok=True)

        js = json.dumps(asdict(self), indent=2)
        path = os.path.join(folder, 'config.json')
        with open(path, 'w') as fout:
            fout.write(js)
        
        return js
    
    @classmethod
    def load(cls, folder: str | PathLike) -> CNNAutoencoderConfig:
        with open(os.path.join(folder, 'config.json'), 'r') as fin:
            conf = json.load(fin)
        
        return cls(**conf)


class CNNAutoencoder(Compressor):
    
    ckpt_name = 'CNNAutoencoder.pt'
    
    def init(self, config: CNNAutoencoderConfig) -> Compressor:
        self.config = config
        self.latent_channels = config.hidden_channels
        
        self.encoder = CNNEncoder(
            in_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_downsample=int(math.log2(config.compress_ratio)),
            num_res=config.num_res,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            norm_groups=config.norm_groups
        )
        
        self.decoder = CNNDecoder(
            target_channels=config.in_channels,
            hidden_channels=config.hidden_channels,
            num_upsample=int(math.log2(config.compress_ratio)),
            num_res=config.num_res,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            norm_groups=config.norm_groups
        )
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
        return nn.MSELoss().forward(preds.value, target=target.value)
