import os
from os import PathLike
from enum import Enum
from dataclasses import dataclass

import torch
from diffusers.models import AutoencoderKL

from pixel import PIXELForPreTraining

PIXEL_DEFAULT_IMAGE_MEAN = torch.tensor([0.5, 0.5, 0.5])
PIXEL_DEFAULT_IMAGE_STD = torch.tensor([0.5, 0.5, 0.5])


class ModelType(Enum):
    SD = AutoencoderKL  # represents the VQGAN from Stable Diffusion
    PIXEL = PIXELForPreTraining


