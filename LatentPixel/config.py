import os
from os import PathLike
from enum import Enum
from dataclasses import dataclass, asdict

import torch
from diffusers.models import AutoencoderKL

from pixel import PIXELForPreTraining

PIXEL_DEFAULT_IMAGE_MEAN = torch.tensor([0.5, 0.5, 0.5])
PIXEL_DEFAULT_IMAGE_STD = torch.tensor([0.5, 0.5, 0.5])


class ModelType(Enum):
    SD = AutoencoderKL  # represents the VQGAN from Stable Diffusion
    PIXEL = PIXELForPreTraining

@dataclass
class RenderConfig:
    dpi: int = 120
    font_size: int = 8
    pixels_per_patch: int = 16
    pad_size: int = 3
    font_file: str = 'GoNotoCurrent.ttf'
    path: str = 'storage/pixel-base'
    rgb = True
    mask_ratio: float = 0.25

    def to_dict(self) -> dict:
        return asdict(self)
