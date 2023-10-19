from __future__ import annotations
from typing import Any
from os import PathLike

import torch

from diffusers import AutoencoderKL

from LatentPixel.text_graph import TGraph
from .compressor import Compressor


class SDAutoencoder(Compressor):
    # wrapper class for the Stable Diffusion AutoencoderKL

    def init(self, config: Any) -> Compressor:
        raise KeyError('Only support init SDAutoencoder from path')
    
    def load(self, path: str | PathLike) -> Compressor:
        model: AutoencoderKL = AutoencoderKL.from_pretrained(path)
        self.latent_channels = 4    # it's always 4
        self.config = model.config
        self.model = model
        self.encoder = model.encoder
        self.decoder = model.decoder
        return self
    
    def save(self, path: str | PathLike) -> None:
        print('No need to save SDautoencoder.')

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.encode(x).latent_dist.mode()

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.model.decode(z).sample
    
    def forward_loss(self, preds: TGraph, target: TGraph, hidden: TGraph) -> torch.Tensor:
        # We do not need to calculate loss for SDautoencoder
        return None
