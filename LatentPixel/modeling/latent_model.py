from __future__ import annotations
from typing import Any, Callable, Iterator
import os

import torch
from torch import nn
from torchvision.transforms import PILToTensor, Compose, ToPILImage, Normalize

from diffusers import AutoencoderKL

from ..text_graph import TGraph
from ..config import LATENT_DEFAULT_MEAN, LATENT_DEFAULT_STD

from .compressors import (
    Compressor,
    CNNAutoencoder,
    SDAutoencoder
)


class LatentModel(nn.Module):
    
    _latent_norm = Compose([Normalize(LATENT_DEFAULT_MEAN, LATENT_DEFAULT_STD)])
    _inv_latent_norm = Compose([Normalize(- LATENT_DEFAULT_MEAN / LATENT_DEFAULT_STD, 1 / LATENT_DEFAULT_STD)])

    def __init__(
            self, 
            compressor_path: str | os.PathLike | None = None,
            backbone_path: str | os.PathLike | None = None,
            compressor_name: str | None = None,
            num_channels: int | None = None,
            num_labels: int | None = None,
            patch_size: int = 16,
            patch_len: int = 1
            ) -> None:
        super().__init__()

        self.compressor_name = compressor_name
        self.num_channel = num_channels
        self.patch_size = patch_size
        self.patch_len = patch_len
        self.num_labels = num_labels
        self.latent_norm = True

        self.compressor = self.load_compressor(compressor_path, compressor_name)
        if self.compressor is None:
            self.num_latent_channel = num_channels
            self.latent_patch_size = patch_size
        else:
            self.num_latent_channel = self.compressor.config.num_latent_channel
            self.latent_patch_size = patch_size // self.compressor.config.compress_ratio
            
        self.backbone = self.load_backbone(
            path=backbone_path,
            num_latent_channel=self.num_latent_channel,
            latent_patch_size=self.latent_patch_size,
            patch_len=self.patch_len,
            num_labels=self.num_labels
        )        

    def forward(self, img: TGraph) -> TGraph:
        if self.compressor is None:
            return self.latent_forward(img)
        
        latent = self.encode(img)
        recon = self.latent_forward(latent)
        if self.has_decoder:
            return self.decode(recon)
        
        return recon

    @torch.no_grad()
    def encode(self, img: TGraph) -> TGraph:
        return self.compressor.encode(img)
    
    @torch.no_grad()
    def decode(self, img: TGraph) -> TGraph:
        return self.compressor.decode(img)

    def latent_forward(self, img: TGraph) -> TGraph:
        raise NotImplementedError('All child module should define this function within it')

    def load_backbone(
            self, 
            path: str | os.PathLike,
            num_latent_channel: int,
            latent_patch_size: int,
            patch_len: int,
            num_labels: int
        ) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')

    def save_backbone(self, path: str | os.PathLike) -> None:
        raise NotImplementedError('All child module should define this function within it')

    def load_compressor(self, path: str | os.PathLike, name: str) -> Compressor | None:
        compressor = None
        if path is None or len(path) == 0:
            print(f'Coder path is none, do not load compressor for this model')
            return compressor
        
        print(f'loading the compressor from {path}')
        match name.lower():
            case 'cnnautoencoder':
                compressor = CNNAutoencoder(path=path)
            case 'sdautoencoder':
                compressor = SDAutoencoder(path=path)
            case _:
                raise KeyError(f'Do not support {name} compressor!')

        return compressor
            
    def save_compressor(self, path: str | os.PathLike) -> None:
        if self.compressor is None:
            print('Abandon the coder saving since the decoder is deleted.')
            return
        elif self.compressor.decoder is None:
            print('Abandon the coder saving since the decoder is deleted.')
            return
        
        self.compressor.save(path)
    
    def init_connection_layers(self) -> None:
        raise NotImplementedError('All child module should define this function within it')

    def get_connection_layers(self) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')
    
    def get_connection_params(self) -> Iterator[nn.Parameter]:
        return self.get_connection_layers().parameters()

    def get_backbone_parameters(self) -> Iterator[nn.Parameter]:
        return self.backbone.parameters()

    def compile(self) -> None:
        raise NotImplementedError('All child module should define this function within it')
    
    def wrap(self, wrap_fn: Callable) -> None:
        raise NotImplementedError('All child module should define this function within it')
    
    def delete_unused_layers(self) -> None:
        raise NotImplementedError('All child module should define this function within it')
    
    def set_grad_for_stage(self, stage: int) -> None:
        if stage == 1:
            print(f"Set gradient config for stage {stage}")
            for param in self.parameters():
                param.requires_grad = False
            for param in self.get_connection_params():
                param.requires_grad = True
        elif stage == 2:
            print(f"Set gradient config for stage {stage}")
            if self.compressor is None:
                print(f'This model has no coder to set grad')
                return
            for param in self.compressor.parameters():
                param.requires_grad = False
                
    def autoregressive_generate(self, prompt: TGraph, gen_idx: int, num_new_patches: int) -> TGraph:
        raise NotImplementedError('All child module should define this function within it')

    @property
    def has_decoder(self) -> bool:
        if self.compressor is None:
            return False
        if self.compressor.decoder is None:
            return False
        
        return True
