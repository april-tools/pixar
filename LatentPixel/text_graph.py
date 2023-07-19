from __future__ import annotations
from typing import Any

import os
from os import PathLike
from math import sqrt

import torch
from torchvision.transforms import PILToTensor, Compose, ToPILImage, Normalize

import numpy as np

import PIL
from PIL.Image import Image

from pixel import PangoCairoTextRenderer

from .utils import (
    init_render,
    render_text,
    get_attention_mask,
    get_num_patches,
    get_patch_size,
    unpatchify,
    get_span_mask,
    mask2img,
    cons_mask,
    gen_circle_mask,
    color_image,
    shrink_mask
)
from .config import (
    PIXEL_DEFAULT_IMAGE_STD,
    PIXEL_DEFAULT_IMAGE_MEAN,
    RenderConfig
)

from pixel import Encoding, PIXELForPreTrainingOutput
    

class TGraph:
    
    # The default image format, [n, c, h, w] or [c, h, w], varies in [0, 1]
    _value: torch.Tensor | None = None
    _patch_size: int | None = None
    _attention_mask: torch.Tensor | None = None
    _patch_mask: torch.Tensor | None = None
    _num_text_patches: int | list[int] | None = None
    _num_gen_patches: int | list[int] | None = None
    loss: torch.Tensor | None = None

    # useful transforms
    _pil2val = Compose([PILToTensor()])
    _val2pil = Compose([ToPILImage(mode='RGB')])
    _pix_normalizer = Compose([Normalize(PIXEL_DEFAULT_IMAGE_MEAN, PIXEL_DEFAULT_IMAGE_STD)])
    _inv_pix_normalizer = Compose([Normalize(- PIXEL_DEFAULT_IMAGE_MEAN / PIXEL_DEFAULT_IMAGE_STD, 1 / PIXEL_DEFAULT_IMAGE_STD)])

    @staticmethod
    def init_render(
        dpi: int = 120,
        font_size: int = 8,
        pixels_per_patch: int = 16,
        pad_size: int = 3,
        font_file: str = 'GoNotoCurrent.ttf',
        path: str = 'storage/pixel-base',
        rgb: bool = True,
        mask_ratio: float = 0.25,
        num_workers: int = 1
    ) -> PangoCairoTextRenderer:
        config = RenderConfig(
            dpi=dpi,
            font_size=font_size,
            pixels_per_patch=pixels_per_patch,
            pad_size=pad_size,
            font_file=font_file,
            path=path,
            rgb=rgb,
            mask_ratio=mask_ratio
        )
        return init_render(config, num_worker=num_workers)

    @property
    def patch_size(self) -> int:
        if self._patch_size:
            return self._patch_size
        self._patch_size = get_patch_size() # use the default patch_size from render
        return self._patch_size
    
    @patch_size.setter
    def patch_size(self, size: int) -> None:
        self._patch_size = size

    @property
    @torch.no_grad()
    def attention_mask(self) -> torch.Tensor:
        if self._attention_mask is not None:
            return self._attention_mask
        
        if isinstance(self.num_text_patches, int):
            self._attention_mask = get_attention_mask(self.num_text_patches, get_num_patches())
            return self._attention_mask
        
        masks = [get_attention_mask(num, get_num_patches()).unsqueeze(0) for num in self.num_text_patches]
        self._attention_mask = torch.cat(masks, dim=0)
        return self._attention_mask
    
    @attention_mask.setter
    def attention_mask(self, mask: torch.Tensor) -> None:
        self._attention_mask = mask

    @property
    @torch.no_grad()
    def num_text_patches(self) -> int | list[int]:
        if self._num_text_patches:
            return self._num_text_patches
        
        if self._attention_mask is not None:
            self._num_text_patches = (self._attention_mask.sum(-1) - 1).tolist()
        
        return self._num_text_patches
    
    @num_text_patches.setter
    def num_text_patches(self, num: int | list[int]) -> None:
        self._num_text_patches = num

    @torch.no_grad()
    def clamp(self) -> TGraph:
        self._value = self._value.clamp(0, 1)

    @property
    def is_squre(self) -> bool:
        sp = self._value.shape
        return sp[-1] == sp[-2]
    
    @property
    def gen_begin_idx(self) -> int | list[int]:
        if isinstance(self.num_text_patches, list):
            idx = [n - g for n, g in zip(self.num_text_patches, self.num_gen_patches)]
        else:
            idx = self.num_text_patches - self.num_gen_patches
        return idx
    
    def init_patch_mask(self, mode: str, num: int = 1, idx: int = -1) -> torch.Tensor:
        match mode:
            case 'span':
                self._patch_mask = get_span_mask(self.num_text_patches)
            case 'end':
                mask = torch.zeros_like(self.attention_mask)
                self._patch_mask = cons_mask(mask, self.num_text_patches, num)
            case 'at_gen':
                mask = torch.zeros_like(self.attention_mask)
                ini_idx = self.gen_begin_idx
                if isinstance(ini_idx, list):
                    begin_idx = [b + idx for b in ini_idx]
                else:
                    begin_idx = ini_idx + idx
                self._patch_mask = cons_mask(mask, begin_idx, num)
            case 'at':
                mask = torch.zeros_like(self.attention_mask)
                mask[..., idx : idx + num] = 1
                self._patch_mask = mask
            case _:
                raise KeyError(f'Unsupported patch mask mode {mode}')
        
        return self._patch_mask
    
    @property
    def patch_mask(self):
        if self._patch_mask is not None:
            return self._patch_mask
        
        return self.init_patch_mask('span')
    
    @patch_mask.setter
    def patch_mask(self, mask: torch.Tensor) -> None:
        self._patch_mask = mask

    def shrink_patch_mask(self) -> torch.Tensor:
        self.patch_mask = shrink_mask(self.patch_mask)

    @property
    def num_gen_patches(self) -> int | list[int]:
        if self._num_gen_patches:
            return self._num_gen_patches 
        
        if isinstance(self.num_text_patches, list):
            self._num_gen_patches = [0 for _ in self.num_text_patches]
        else:
            self._num_gen_patches = 0
        
        return self._num_gen_patches
    
    @num_gen_patches.setter
    def num_gen_patches(self, num: int | list[int]) -> None:
        self._num_gen_patches = num

    def add_gen(self, num_generated: int) -> TGraph:
        o_gen_patches = self.num_gen_patches

        if isinstance(self.num_text_patches, list):
            max_gen_num = [get_num_patches() - 1 - n for n in self.num_text_patches]
            gen_num = [min(m, num_generated) for m in max_gen_num]  # The actual generated number of patches
            generated_num = [o + g for o, g in zip(o_gen_patches, gen_num)]
            self.num_gen_patches = generated_num    # update the number of generated patches
            num_text_patches = [n + g for n, g in zip(self.num_text_patches, gen_num)]  
            self.num_text_patches = num_text_patches # update the number of text patches
        else:
            max_gen_num = get_num_patches() - 1 - self.num_text_patches
            gen_num = min(max_gen_num, num_generated)   # The actual generated number of patches
            generated_num = o_gen_patches + gen_num
            self.num_gen_patches = generated_num    # update the number of generated patches
            self.num_text_patches = self.num_text_patches + gen_num
        
        self.attention_mask = None  # reset the attention mask
        
        return self

    @classmethod
    def from_value(
        cls, 
        value: torch.Tensor, 
        patch_size: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        patch_mask: torch.Tensor | None = None,
        num_text_patches: int | list[int] | None = None,
        num_gen_patches: int | list[int] | None = None,
        loss: torch.Tensor | None = None
        ) -> TGraph:

        graph = TGraph()
        graph._value = value
        graph._patch_size = patch_size
        graph._attention_mask = attention_mask
        graph._patch_mask = patch_mask
        graph._num_text_patches = num_text_patches
        graph._num_gen_patches = num_gen_patches
        graph.loss = loss

        return graph
    
    # paint the masked patchs with color or noise
    @torch.no_grad()
    def paint_mask(self, color: str | tuple[float, float, float] = 'green', alpha: float = 0.1) -> TGraph:
        self.unsquarelize()
                
        height = self.patch_size
        width = height * get_num_patches()

        mask = mask2img(self.attention_mask * self.patch_mask, self.patch_size)
        backcolor = color_image(color, height, width)

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        masked_value = (self._value * mask) * (1 - alpha) + backcolor * mask * alpha
        un_masked = self._value * (1 - mask)
        self._value = masked_value + un_masked

        return self
    
    @torch.no_grad()
    def circle_mask(self, color: str | tuple[float, float, float] = 'green', alpha: float = 0.1, width: int = 2) -> TGraph:
        self.unsquarelize()
        height = self.patch_size
        w = height * get_num_patches()

        mask = self.attention_mask * self.patch_mask
        mask = mask2img(mask, self.patch_size)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        grid_mask = gen_circle_mask(height, get_num_patches(), width) * mask
        color = color_image(color, height, w)
        grid = grid_mask * color
        
        grid_value = (self._value * (1 - alpha) + grid * alpha) * grid_mask
        self._value = self._value * (1 - grid_mask) + grid_value

        return self

    def _from_PIL(self, img: Image) -> torch.Tensor:
        img = img.convert('RGB') if img.mode != 'RGB' else img
        return self._pil2val(img) / 255

    @classmethod
    def from_PIL(cls, img: Image | list[Image]) -> TGraph:
        graph = cls()
        if isinstance(img, list):
            imgs = [graph._from_PIL(i).unsqueeze(0) for i in img]
            graph._value = torch.cat(imgs, dim=0)
        else:
            print(type(img))
            graph._value = graph._from_PIL(img)
        return graph

    def _to_PIL(self, value: torch.Tensor) -> Image:
        return self._val2pil(value.clamp(0, 1))

    def to_PIL(self) -> Image | list[Image]:
        if self._value.dim() == 3:
            return self._to_PIL(self._value)
        
        return [self._to_PIL(img) for img in self._value]
    
    @classmethod
    def from_numpy(cls, img: np.ndarray) -> TGraph:
        # H * W * C -> C * H * W
        TGraph = cls()
        timg = torch.tensor(img, dtype=torch.float)
        if timg.dim() == 3:
            TGraph._value = timg.permute([2, 0, 1])
        elif timg.dim() == 4:
            TGraph._value = timg.permute([0, 3, 1, 2])
        return TGraph
    
    def to_numpy(self) -> np.ndarray:
        # C * H * W -> H * W * C
        if self._value.dim() == 3:
            return self._value.permute([1, 2, 0]).cpu().detach().numpy()
        
        return self._value.permute([0, 2, 3, 1]).cpu().detach().numpy()
    
    @classmethod
    def from_SD(cls, img: torch.Tensor, do_clip: bool = False, attention_mask: torch.Tensor | None = None, patch_mask: torch.Tensor | None = None) -> TGraph:
        graph = cls()
        graph._value = (img / 2 + 0.5)
        if do_clip:
            graph._value = graph._value.clamp_(0, 1)

        graph.attention_mask = attention_mask
        graph.patch_mask = patch_mask

        return graph

    def to_SD(self) -> torch.Tensor:
        return self._value * 2 - 1
    
    def to_device(self, device: Any) -> TGraph:
        self._value = self._value.to(device)
        return self
    
    @classmethod
    def from_pixel_img(cls, img: torch.Tensor, do_clip: bool = False, attention_mask: torch.Tensor | None = None, patch_mask: torch.Tensor | None = None) -> None:
        graph = cls()
        graph._value = cls._inv_pix_normalizer(img)

        if do_clip:
            graph._value = graph._value.clamp_(0, 1)

        graph.attention_mask = attention_mask
        graph.patch_mask = patch_mask

        return graph

    @classmethod
    def from_pixel_logits(cls, logits: torch.Tensor, do_clamp: bool = False, patch_size: int = None) -> TGraph:
        graph = cls()
        graph._value = cls._inv_pix_normalizer(unpatchify(logits, patch_size))
        if do_clamp:
            graph._value = graph._value.clamp_(0, 1)
        return graph.unsquarelize()

    @classmethod
    def from_pixel(cls, output: PIXELForPreTrainingOutput, do_clamp: bool = False) -> TGraph:
        graph = cls.from_pixel_logits(output.logits, do_clamp=do_clamp)
        graph.attention_mask = output.attention_mask
        graph.patch_mask = output.mask
        return graph

    def to_pixel(self) -> torch.Tensor:
        return self._pix_normalizer(self._value)
    
    @classmethod
    def from_file(cls, path: str | PathLike) -> TGraph:
        image = PIL.Image.open(path, 'r')
        return cls.from_PIL(image)
    
    def _to_file(self, path: str | PathLike, value: torch.Tensor) -> None:
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
    def from_text(cls, text: str | list[str], **kwargs) -> TGraph:
        encods = render_text(text, **kwargs)
        TGraph = cls()
        if isinstance(encods, Encoding):
            TGraph._value = torch.tensor(encods.pixel_values / 255, dtype=torch.float).permute(2, 0, 1)
            TGraph.num_text_patches = encods.num_text_patches
            return TGraph
        
        imgs = [torch.tensor(encod.pixel_values / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0) for encod in encods]
        TGraph._value = torch.cat(imgs, dim=0).contiguous()

        nums = [encod.num_text_patches for encod in encods]
        TGraph.num_text_patches = nums

        return TGraph
    
    def squarelize(self) -> TGraph:
        if self.is_squre:
            return self
        
        self._patch_size = self._value.shape[-2]
        num_rows = int(sqrt(self._value.shape[-1] // self._patch_size))
        rows = torch.tensor_split(self._value, num_rows, dim=-1)
        self._value = torch.cat(rows, dim=-2)

        return self
    
    def unsquarelize(self, patch_size: int | None = None) -> TGraph:
        if not self.is_squre:
            return self
        
        if patch_size:
            self._patch_size = patch_size

        if self._patch_size is None:
            self._patch_size = self.patch_size

        num_rows = self._value.shape[-2] // self._patch_size
        rows = torch.tensor_split(self._value, num_rows, dim=-2)
        self._value = torch.cat(rows, dim=-1)
    
        return self
    
    @classmethod
    def reconstruct(cls, origin: TGraph, generated: TGraph, do_clamp: bool = False) -> TGraph:
        recon = cls()
        origin.unsquarelize()
        generated.unsquarelize()

        mask = origin.attention_mask * generated.patch_mask
        mask = mask2img(mask, origin.patch_size)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        mask = mask == 1

        recon._value = (
            origin._value * (~mask).int() + generated._value * mask.int()
        )
        if do_clamp:
            recon._value = recon._value.clamp_(0, 1)

        recon.attention_mask = origin.attention_mask
        recon.patch_mask = generated.patch_mask
        recon.num_text_patches = origin.num_text_patches

        return recon
    