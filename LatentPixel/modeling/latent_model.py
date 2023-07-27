from typing import Any, Callable, Iterator
import os

import torch
from torch import nn
from torchvision.transforms import PILToTensor, Compose, ToPILImage, Normalize

from diffusers import AutoencoderKL

from ..text_graph import TGraph
from ..config import LATENT_DEFAULT_MEAN, LATENT_DEFAULT_STD


class LatentModel(nn.Module):
    
    _latent_norm = Compose([Normalize(LATENT_DEFAULT_MEAN, LATENT_DEFAULT_STD)])
    _inv_latent_norm = Compose([Normalize(- LATENT_DEFAULT_MEAN / LATENT_DEFAULT_STD, 1 / LATENT_DEFAULT_STD)])

    def __init__(
            self, 
            coder_path: str | os.PathLike | None = None,
            backbone_path: str | os.PathLike | None = None,
            img_size: tuple[int, int, int] | None = None,
            latent_size: tuple[int, int, int] | None = None,
            num_labels: int | None = None
            ) -> None:
        super().__init__()

        self.img_size = img_size
        self.latent_size = latent_size
        self.num_labels = num_labels
        
        if self.img_size is None:
            self.img_size = self.latent_size
        if self.latent_size is None:
            self.latent_size = self.img_size

        self.coder = self.load_coder(coder_path)
        self.backbone = self.load_backbone(backbone_path)

    def forward(self, img: TGraph) -> TGraph:
        if self.coder is None:
            return self.latent_forward(img)
        
        latent = self.encode(img)
        recon = self.latent_forward(latent)
        if self.has_decoder:
            return self.decode(recon)
        
        return recon

    @torch.no_grad()
    def encode(self, img: TGraph) -> TGraph:
        pixel_values = img.unsquarelize().to_SD()
        with torch.no_grad():
            latent = self.coder.encode(pixel_values).latent_dist.mode()
        
        return TGraph.from_value(
            value=latent,
            patch_size=self.patch_size,
            attention_mask=img.attention_mask,
            patch_mask=img.patch_mask,
            num_text_patches=img.patch_mask,
            num_gen_patches=img.num_gen_patches
        )

    @torch.no_grad()
    def decode(self, img: TGraph) -> TGraph:
        decoded = self.coder.decode(img.value).sample
        result = TGraph.from_SD(
            img=decoded, 
            do_clip=True,
            attention_mask=img.attention_mask,
            patch_mask=img.patch_mask
        )
        result.patch_size = self.patch_size

        return result

    def latent_forward(self, img: TGraph) -> TGraph:
        raise NotImplementedError('All child module should define this function within it')
    
    def reverse_diffuse(self, img: TGraph, num_steps: int) -> TGraph:
        raise NotImplementedError(f'Model {type(self)} do not implemented the diffusion function!')

    def load_backbone(self, path: str | os.PathLike) -> nn.Module:
        raise NotImplementedError('All child module should define this function within it')

    def save_backbone(self, path: str | os.PathLike) -> None:
        raise NotImplementedError('All child module should define this function within it')

    def load_coder(self, path: str | os.PathLike) -> nn.Module:
        if path is None or len(path) == 0:
            self.coder = None
            self.latent_size = self.img_size
            print(f'Coder path is none, do not load coder for this model')
            return None
        
        print(f'loading the coder from {path}')
        coder: AutoencoderKL = AutoencoderKL.from_pretrained(path)
        assert self.latent_size[0] == coder.config['latent_channels']
        self.coder = coder
        return self.coder

    def save_coder(self, path: str | os.PathLike) -> None:
        if self.coder is None:
            print('Abandon the coder saving since the decoder is deleted.')
            return
        elif self.coder.decoder is None:
            print('Abandon the coder saving since the decoder is deleted.')
            return
        
        self.coder.save_pretrained(path)
    
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
            if self.coder is None:
                print(f'This model has no coder to set grad')
                return
            for param in self.coder.parameters():
                param.requires_grad = False

    @property
    def has_decoder(self) -> bool:
        if self.coder is None:
            return False
        if self.coder.decoder is None:
            return False
        
        return True
    
    @property
    def has_backbone(self) -> bool:
        if self.backbone is None:
            return False
        return True
    
    @property
    def patch_size(self) -> int:
        return self.img_size[1]

    @property
    def latent_patch_size(self) -> int:
        return self.latent_size[1]
    
    @property
    def num_channel(self) -> int:
        return self.img_size[0]
    
    @property
    def num_latent_channel(self) -> int:
        return self.latent_size[0]
    