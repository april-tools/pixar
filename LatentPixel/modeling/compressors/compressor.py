from __future__ import annotations
from typing import Any

import os
import torch
from torch import nn
from ...text_graph import TGraph

class Compressor(nn.Module):
    
    def __init__(
            self, 
            path: str | os.PathLike | None = None,
            binary: bool = False,
            config: Any | None = None
            ) -> None:
        super().__init__()
        
        self.path = path
        self.binary = binary
        self.config = config
        self.encoder: nn.Module = None
        self.decoder: nn.Module = None
        self.latent_channels: int = 0
        
        if self.path is None:
            self.init(self.config)
        else:
            self.load(self.path)
            
    def init(self, config: Any) -> Compressor:
        raise NotImplementedError('All child module should define this function within it')
                
    def load(self, path: str | os.PathLike) -> Compressor:
        raise NotImplementedError('All child module should define this function within it')

    def save(self, path: str | os.PathLike) -> None:
        raise NotImplementedError('All child module should define this function within it')

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('All child module should define this function within it')

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('All child module should define this function within it')

    def encode(self, img: TGraph) -> TGraph:
        # clip the long img into patches
        x = img.value * 2 - 1  # map values from [0, 1] range to [-1, 1] range
        patches = self.patchify(x, img.patch_len)
        
        # encode patches
        z = self._encode(patches)

        # connect patches into long img
        z = self.unpatchify(z, x.shape[0])
        
        encoded = TGraph.from_tgraph(img)
        encoded._patch_size = z.shape[2]
        encoded._value = z
        
        return encoded

    def decode(self, img: TGraph) -> TGraph:
        z = img.value
        patches = self.patchify(z, img.patch_len)
        
        # decode patches
        y = self._decode(patches)
        
        # connect patches into long img
        y = self.unpatchify(y, z.shape[0])
        
        decoded = TGraph.from_tgraph(img)
        decoded._patch_size = y.shape[2]
        decoded._value = (y + 1) / 2    # map values from [-1, 1] range to [0, 1] range
        
        return decoded
    
    def forward_loss(self, preds: TGraph, target: TGraph, hidden: TGraph) -> torch.Tensor:
        raise NotImplementedError('All child module should define this function within it')

    def forward(self, img: TGraph) -> TGraph:
        hidden = self.encode(img)
        recon = self.decode(hidden)
        
        loss = self.forward_loss(recon, img, hidden)
        recon.loss = loss
        
        return recon
    
    @classmethod
    def patchify(cls, x: torch.Tensor, patch_len: int) -> torch.Tensor:
        # clip the long img into patches
        if x.dim() == 3:
            x = x.unsqueeze(0)

        bs, c, h, w = x.shape
        
        x = x.reshape(bs, c, h, -1, patch_len * h)    # bs, c, h, ps, w
        x = x.permute(3, 0, 1, 2, 4)    # ps, bs, c, h, w
        x = x.flatten(0, 1) # bps, c, h, w

        return x
    
    @classmethod
    def unpatchify(cls, z: torch.Tensor, batch_size: int) -> torch.Tensor:

        bps, c, h, patch_width = z.shape
        patch_num = bps // batch_size
        w = patch_width * patch_num
        

        # connect patches into long img
        z = z.unflatten(0, (patch_num, batch_size))    # ps, bs, lc, lh, lw
        z = z.permute(1, 2, 3, 0, 4)
        z = z.reshape([batch_size, c, h, w])
        
        return z
