from __future__ import annotations
from typing import Any, Callable, Iterator
import os

import torch
from torch import nn
from torchvision.transforms import PILToTensor, Compose, ToPILImage, Normalize

from diffusers import AutoencoderKL
from tqdm import tqdm

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
            patch_len: int = 1,
            binary: bool = False
            ) -> None:
        super().__init__()

        self.compressor_name = compressor_name
        self.num_channel = num_channels
        self.patch_size = patch_size
        self.patch_len = patch_len
        self.num_labels = num_labels
        self.latent_norm = True
        self.binary = binary

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
            num_labels=self.num_labels,
            binary=self.binary
        )

        problems, ok = self.check_structure()
        if not ok:
            for p in problems:
                print(p)
            raise KeyError('Wrong configuration')

        
    def check_structure(self) -> tuple(list[str], bool):
        msgs = []
        ok = True
        if self.compressor is not None:
            if self.binary != self.compressor.config.binary:
                msgs.append('Binary setting does not match the compressor!')
                ok = False
            if self.num_channel != self.compressor.config.num_channel:
                msgs.append('Input channel setting does not match the compressor!')
                ok = False
        if self.binary:
            if self.num_channel != 1:
                msgs.append('Number of input channel should be 1 when binary')
                ok = False
            if self.compressor is not None:
                if self.compressor.config.num_channel != 1:
                    msgs.append('Number of compressor\'s channel shold be 1 when binary')
                    ok = False
                
        return msgs, ok
        
    def forward(self, img: TGraph, temperature: float=1.0) -> TGraph:
        if img._binary and img._value.is_floating_point():
            img.binary()
        if self.compressor is None:
            recon = self.latent_forward(img)
            recon._labels = img._labels
            if self.binary and self.num_labels is None:
                recon._value /= temperature
                recon._value.sigmoid_()
                recon._binary = True
            return recon
        
        latent = self.encode(img)
        latent._labels = img._labels
        recon = self.latent_forward(latent)
        loss = recon.loss
        recon._labels = img._labels

        if self.has_decoder:
            recon = self.decode(recon)
            recon._labels = img._labels

            if self.binary and self.num_labels is None:
                recon._value /= temperature
                recon._value.sigmoid_()
                recon._binary = True

        recon.loss = loss
        
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
            num_labels: int,
            binary: bool
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

    def _generate(self, prompt: TGraph, binary_method: str='threshold', threshold: float=0.5, temperature: float=1.0) -> TGraph:
        gen = TGraph.from_tgraph(prompt)
        gen._value = prompt.value
        output = self.forward(prompt, temperature=temperature)
        for idx, num_text in enumerate(prompt.num_text_patches):
            bidx = (num_text - 2) * prompt.patch_len * prompt.patch_size            
            eidx = bidx + prompt.patch_len * prompt.patch_size
            patch = output._value[idx, :, :, bidx:eidx]
            gen._value[idx, :, :, eidx:eidx + prompt.patch_len * prompt.patch_size] = patch
            gen._num_text_patches[idx] += 1
            gen._attention_mask = None
        if gen._binary:
            gen.binary(binary_method, threshold)
        return gen
                
    def autoregressive_generate(self, prompt: TGraph, gen_idx: int, num_new_patches: int, binary_method: str='threshold', threshold: float=0.5, temperature: float=1.0) -> TGraph:
        for _ in range(num_new_patches):
            gen = self._generate(prompt, binary_method, threshold, temperature=temperature)
            prompt = gen
        return gen

    @property
    def has_decoder(self) -> bool:
        if self.compressor is None:
            return False
        if self.compressor.decoder is None:
            return False
        
        return True
