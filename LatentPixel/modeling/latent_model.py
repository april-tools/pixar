from typing import Any, Callable, Iterator
import os

import torch
from torch import nn

from ..text_graph import TGraph


class LatentModel(nn.Module):

    backbone: nn.Module = None
    coder: nn.Module = None
    img_size: tuple[int, int, int] = None
    latent_size: tuple[int, int, int] = None

    def __init__(
            self, 
            coder_path: str | os.PathLike | None = None,
            backbone_path: str | os.PathLike | None = None,
            img_size: tuple[int, int] | None = None,
            latent_size: tuple[int, int] | None = None
            ) -> None:
        super().__init__()

        self.img_size = img_size
        self.latent_size = latent_size

        self.load_coder(coder_path)
        self.load_backbone(backbone_path)

    def forward(self, img: TGraph) -> TGraph:
        if self.coder is None:
            return self.latent_forward(img)
        
        latent = self.encode(img)
        if self.has_decoder:
            return self.decode(latent)
        
        return latent

    def encode(self, img: TGraph) -> TGraph:
        raise NotImplementedError('All child module should define this function within it')

    def decode(self, img: TGraph) -> TGraph:
        raise NotImplementedError('All child module should define this function within it')

    def latent_forward(self, img: TGraph) -> TGraph:
        raise NotImplementedError('All child module should define this function within it')
    
    def reverse_diffuse(self, img: TGraph, num_steps: int) -> TGraph:
        raise NotImplementedError(f'Model {type(self)} do not implemented the diffusion function!')

    def load_backbone(self, path: str | os.PathLike) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')

    def save_backbone(self, path: str | os.PathLike) -> None:
        raise NotImplementedError('All child module should define this function within it')

    def load_coder(self, path: str | os.PathLike) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')

    def save_coder(self, path: str | os.PathLike) -> None:
        raise NotImplementedError('All child module should define this function within it')
    
    def init_connection_layers(self) -> None:
        raise NotImplementedError('All child module should define this function within it')

    def get_connection_layers(self) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')
    
    def get_connection_params(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError('All child module should define this function within it')

    def get_backbone_parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError('All child module should define this function within it')

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
            if self._coder is not None:
                print(f'This model has no coder to set grad')
                return
            for param in self.coder.parameters():
                param.requires_grad = False

    @property
    def has_decoder(self) -> bool:
        raise NotImplementedError('All child module should define this function within it')
