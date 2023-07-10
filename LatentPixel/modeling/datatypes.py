from __future__ import annotations
import os
from os import PathLike
from enum import Enum
from math import sqrt

import torch
from torchvision.transforms import PILToTensor, Compose, ToPILImage, Normalize

import numpy as np

from diffusers.models import AutoencoderKL

import PIL
from PIL.Image import Image

from ..utils import (
    render_text,
    get_attention_mask,
    get_num_patches,
    get_patch_size
)
from ..config import (
    PIXEL_DEFAULT_IMAGE_STD,
    PIXEL_DEFAULT_IMAGE_MEAN
)

from pixel import Encoding


class Coder(Enum):
    SD = AutoencoderKL  # represents the VQGAN from Stable Diffusion
    

class Graph:
    
    # The default image format, [n, c * h * w], varies in [0, 1]
    _value: torch.Tensor | None = None
    _patch_size: int | None = None
    _attention_mask: torch.Tensor | None = None
    num_text_patches: int | list[int] | None = None

    # useful transforms
    _pil2val = Compose([PILToTensor()])
    _val2pil = Compose([ToPILImage(mode='RGB')])
    _pix_normalizer = Compose([Normalize(PIXEL_DEFAULT_IMAGE_MEAN, PIXEL_DEFAULT_IMAGE_STD)])
    _inv_pix_normalizer = Compose([Normalize(- PIXEL_DEFAULT_IMAGE_MEAN / PIXEL_DEFAULT_IMAGE_STD, 1 / PIXEL_DEFAULT_IMAGE_STD)])

    def _from_PIL(self, img: Image) -> torch.Tensor:
        img = img.convert('RGB') if img.mode != 'RGB' else img
        return self._pil2val(img) / 255

    @classmethod
    def from_PIL(cls, img: Image | list[Image]) -> Graph:
        graph = cls()
        if isinstance(img, list):
            imgs = [graph._from_PIL(i).unsqueeze(0) for i in img]
            graph._value = torch.cat(imgs, dim=0)
        else:
            print(type(img))
            graph._value = graph._from_PIL(img)
        return graph

    def _to_PIL(self, value: torch.Tensor) -> Image:
        return self._val2pil(value)

    def to_PIL(self) -> Image | list[Image]:
        if self._value.dim() == 3:
            return self._to_PIL(self._value)
        
        return [self._to_PIL(img) for img in self._value]
    
    @classmethod
    def from_numpy(cls, img: np.ndarray) -> Graph:
        # H * W * C -> C * H * W
        graph = cls()
        timg = torch.tensor(img, dtype=torch.float)
        if timg.dim() == 3:
            graph._value = timg.permute([2, 0, 1])
        elif timg.dim() == 4:
            graph._value = timg.permute([0, 3, 1, 2])
        return graph
    
    def to_numpy(self) -> np.ndarray:
        # C * H * W -> H * W * C
        if self._value.dim() == 3:
            return self._value.permute([1, 2, 0]).cpu().detach().numpy()
        
        return self._value.permute([0, 2, 3, 1]).cpu().detach().numpy()
    
    @classmethod
    def from_SD(cls, img: torch.Tensor, do_clip: bool = False) -> None:
        graph = cls()
        graph._value = (img / 2 + 0.5)
        if do_clip:
            graph._value = graph._value.clamp(0, 1)

        return graph

    def to_SD(self) -> torch.Tensor:
        return self._value * 2 - 1
    
    @classmethod
    def from_pixel(cls, img: torch.Tensor, do_clip: bool = False) -> None:
        graph = cls()
        graph._value = cls._inv_pix_normalizer(img)
        if do_clip:
            graph._value = graph._value.clamp(0, 1)

        return graph

    def to_pixel(self) -> torch.Tensor:
        return self._pix_normalizer(self._value)
    
    @classmethod
    def from_file(cls, path: str | PathLike) -> Graph:
        image = PIL.Image.open(path, 'r')
        return cls.from_PIL(image)
    
    def _to_file(self, path: str | PathLike, value: torch.Tensor) -> None:
        img = value.clamp(0, 1)
        img = self._val2pil(value)
        img.save(path, 'PNG')

    def to_file(self, path: str | PathLike) -> None:
        if self._value.dim() == 3:
            return self._to_file(path, self._value)
        
        os.makedirs(path, exist_ok=False)
        for idx, value in enumerate(self._value):
            file_path = os.path.join(path, f'{idx}.png')
            self._to_file(file_path, value)

    @classmethod
    def from_text(cls, text: str | list[str]) -> Graph:
        encods = render_text(text)
        graph = cls()
        if isinstance(encods, Encoding):
            graph._value = torch.tensor(encods.pixel_values / 255, dtype=torch.float).permute(2, 0, 1)
            graph.num_text_patches = encods.num_text_patches
            return graph
        
        imgs = [torch.tensor(encod.pixel_values / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0) for encod in encods]
        graph._value = torch.cat(imgs, dim=0).contiguous()

        nums = [encod.num_text_patches for encod in encods]
        graph.num_text_patches = nums

        return graph
    
    @property
    def is_squre(self) -> bool:
        sp = self._value.shape
        return sp[-1] == sp[-2]
    
    def squarelize(self) -> Graph:
        if self.is_squre:
            return self
        
        self._patch_size = self._value.shape[-2]
        num_rows = int(sqrt(self._value.shape[-1] // self._patch_size))
        rows = torch.tensor_split(self._value, num_rows, dim=-1)
        self._value = torch.cat(rows, dim=-2)

        return self
    
    def unsquarelize(self, patch_size: int | None = None) -> Graph:
        if not self.is_squre:
            return self
        
        if patch_size:
            self._patch_size = patch_size

        if self._patch_size is None:
            self._patch_size = get_patch_size()

        num_rows = self._value.shape[-2] // self._patch_size
        rows = torch.tensor_split(self._value, num_rows, dim=-2)
        self._value = torch.cat(rows, dim=-1)

        return self
    
    def get_attention_mask(self) -> torch.Tensor:
        if self._attention_mask:
            return self._attention_mask
        
        if self._value.dim() == 3:
            self._attention_mask = get_attention_mask(self.num_text_patches, get_num_patches())
            return self._attention_mask
        
        masks = [get_attention_mask(num, get_num_patches()).unsqueeze(0) for num in self.num_text_patches]
        self._attention_mask = torch.cat(masks, dim=0)
        return self._attention_mask
    