from __future__ import annotations
from dataclasses import dataclass, asdict
import os
from os import PathLike
import json
from typing import Any
import math

import torch
from torch import nn

from LatentPixel.modeling.latent_model import Compressor
from LatentPixel.text_graph import TGraph

from .cnn_blocks import (
    CNNDecoder,
    CNNEncoder
)
from .latent_model import Compressor


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
        return  self
    
    def load(self, path: str | PathLike) -> Compressor:
        self.config = CNNAutoencoderConfig.load(path)
        self.init(self.config)
        
        statedict = torch.load(os.path.join(path, self.ckpt_name))
        self.load_state_dict(statedict)
        
        return self
    
    def save(self, path: str | PathLike) -> None:
        self.config.save(path)
        torch.save(self.state_dict(), os.path.join(path, self.ckpt_name))
        return
    
    def encode(self, img: TGraph) -> TGraph:
        # clip the long img into patches
        x = img.value
        bs, c, h, w = x.shape
        lh = h // self.config.compress_ratio
        lw = w // self.config.compress_ratio
        lc = self.config.hidden_dim
        
        x = x.reshape(bs, c, h, -1, img.patch_len * img.patch_size)    # bs, c, h, ps, w
        x = x.permute(3, 0, 1, 2, 4)    # ps, bs, c, h, w
        x = x.flatten(0, 1) # bps, c, h, w
        
        # encode patches
        z = self.encoder.forward(x)

        # connect patches into long img
        z = z.unflatten(0, (-1, bs))    # ps, bs, lc, lh, lw
        z = z.permute(1, 2, 3, 0, 4)
        z = z.reshape([bs, lc, lh, lw])
        
        encoded = TGraph.from_tgraph(img)
        encoded._patch_size = lh
        encoded._value = z
        
        return encoded
    
    def decode(self, img: TGraph) -> TGraph:
        z = img.value
        bs, lc, lh, lw = z.shape
        h = lh * self.config.compress_ratio
        w = lw * self.config.compress_ratio
        c = self.config.in_channels
        
        z = z.reshape(bs, lc, lh, -1, img.patch_len * img.patch_size)
        z = z.permute(3, 0, 1, 2, 4)    # ps, bs, lc, lh, lw
        z = z.flatten(0, 1) # bps, lc, lh, lw
        
        # decode patches
        y = self.decoder.forward(z)
        
        # connect patches into long img
        y = y.unflatten(0, (-1, bs))    # ps, bs, c, h, w
        y = y.permute(1, 2, 3, 0, 4)
        y = y.reshape([bs, c, h, w])    # bs, c, h, w
        
        decoded = TGraph.from_tgraph(img)
        decoded._patch_size = h
        decoded._value = y
        
        return decoded
            
    def forward_loss(self, preds: TGraph, target: TGraph, hidden: TGraph) -> torch.Tensor:
        return nn.MSELoss().forward(preds.value, target=target.value)
    