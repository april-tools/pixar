from enum import Enum

import torch
from torchvision.transforms import PILToTensor, Compose, ToPILImage, Normalize

import numpy as np

from diffusers.models import AutoencoderKL

from PIL.Image import Image

from ..config import (
    PIXEL_DEFAULT_IMAGE_STD,
    PIXEL_DEFAULT_IMAGE_MEAN
)


class Coder(Enum):
    SD = AutoencoderKL  # represents the VQGAN from Stable Diffusion
    

class Graph:
    
    # The default image format, [n, c * h * w], varies in [0, 1]
    _value: torch.Tensor
    _pil2val = Compose([PILToTensor()])
    _val2pil = Compose([ToPILImage])
    _pix_normalizer = Compose([Normalize(PIXEL_DEFAULT_IMAGE_MEAN, PIXEL_DEFAULT_IMAGE_STD)])

    def _from_PIL(self, img: Image) -> torch.Tensor:
        img = img.convert('RGB')
        return self._pil2val(img) / 255

    def from_PIL(self, img: Image | list[Image]) -> None:
        if isinstance(img, list):
            imgs = [self._from_PIL(i).unsqueeze(0) for i in img]
            self._value = torch.cat(imgs, dim=0)
            return
        
        self._value = self._from_PIL(img)

    def _to_PIL(self, value: torch.Tensor) -> Image:
        return self._val2pil(self._value).convert('RGB')

    def to_PIL(self) -> Image | list[Image]:
        if self._value.dim() == 3:
            return self._to_PIL(self._value)
        
        return [self._to_PIL(img) for img in self._value]
    
    def from_numpy(self, img: np.ndarray) -> None:
        # H * W * C -> C * H * W
        timg = torch.Tensor(img)
        if timg.dim() == 3:
            self._value = timg.permute([2, 0, 1])
        elif timg.dim() == 4:
            self._value = timg.permute([0, 3, 1, 2])
    
    def to_numpy(self) -> np.ndarray:
        # C * H * W -> H * W * C
        if self._value.dim() == 3:
            return self._value.permute([1, 2, 0]).cpu().detach().numpy()
        
        return self._value.permute([0, 2, 3, 1]).cpu().detach().numpy()
    
    def from_SD(self, img: torch.Tensor) -> None:
        self._value = (img / 2 + 0.5).clamp(0, 1)

    def to_SD(self) -> torch.Tensor:
        return self._value * 2 - 1
    
    def from_pixel(self, img: torch.Tensor) -> None:
        ...

    def to_pixel(self) -> torch.Tensor:
        return self._pix_normalizer(self._value)
    

