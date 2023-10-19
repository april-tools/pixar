import os
from os import PathLike
from enum import Enum
from dataclasses import dataclass, asdict

import torch
from diffusers.models import AutoencoderKL

from pixel import PIXELForPreTraining

PIXEL_DEFAULT_IMAGE_MEAN = torch.tensor([0.5, 0.5, 0.5])
PIXEL_DEFAULT_IMAGE_STD = torch.tensor([0.5, 0.5, 0.5])
LATENT_DEFAULT_MEAN = torch.tensor([4.2295, 3.9307, -0.6823, -2.6319])
LATENT_DEFAULT_STD = torch.tensor([4.3281, 3.4913, 3.0322, 3.3330])


class ModelType(Enum):
    SD = AutoencoderKL  # represents the VQGAN from Stable Diffusion
    PIXEL = PIXELForPreTraining


@dataclass
class RenderConfig:
    dpi: int = 120
    font_size: int = 8
    pixels_per_patch: int = 16
    pad_size: int = 3
    patch_len: int = 1
    font_file: str = 'GoNotoCurrent.ttf'
    path: str = 'storage/pixel-base'
    rgb: bool = True
    mask_ratio: float = 0.25
    binary: bool = False
    max_seq_length: int = 529
    patch_len: int = 1

    def to_dict(self) -> dict:
        return asdict(self)
    
    # def __str__(self) -> str:
    #     return str(self.to_dict)
