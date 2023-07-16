from typing import Any
import os

import torch
from torch import nn


class LatentModel(nn.Module):

    def get_connection_layers(self) -> nn.ModuleList:
        raise NotImplementedError('All child module should define this function within it')
    
    def save_model(self, path: str | os.PathLike, only_backbone: bool = False) -> None:
        raise NotImplementedError('All child module should define this function within it')
    
    def load_model(self, path: str | os.PathLike, no_decoder: bool = False) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')

    def save_backbone(self, path: str | os.PathLike) -> None:
        raise NotImplementedError('All child module should define this function within it')
    
    def load_backbone(self, path: str | os.PathLike, config: Any = None) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')

    def save_coder(self, path: str | os.PathLike) -> None:
        raise NotImplementedError('All child module should define this function within it')

    def load_coder(self, path: str | os.PathLike, config: Any = None) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')

    def compile(self) -> None:
        raise NotImplementedError('All child module should define this function within it')
