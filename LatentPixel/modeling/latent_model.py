from typing import Any, Callable
import os

import torch
from torch import nn


class LatentModel(nn.Module):

    _backbone: nn.Module = None
    _coder: nn.Module = None

    def load_backbone(self, path: str | os.PathLike, config: Any = None) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')

    def save_backbone(self, path: str | os.PathLike) -> None:
        raise NotImplementedError('All child module should define this function within it')

    def load_coder(self, path: str | os.PathLike, config: Any = None) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')

    def save_coder(self, path: str | os.PathLike) -> None:
        raise NotImplementedError('All child module should define this function within it')

    def get_connection_layers(self) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')
    
    def get_connection_params(self) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')

    def get_backbone_parameters(self) -> Any:
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
        else:
            raise NotImplementedError('All child module should define this function within it')

    
    # @property
    # def backbone(self) -> nn.Module:
    #     return self._backbone
    
    # @backbone.setter
    # def backbone(self, module: nn.Module) -> None:
    #     self._backbone = module

    # @property
    # def coder(self) -> nn.Module:
    #     return self._coder
    
    # @coder.setter
    # def coder(self, module: nn.Module) -> None:
    #     self._coder = module
