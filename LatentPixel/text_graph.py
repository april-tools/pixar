from __future__ import annotations
from typing import Any

import os
from os import PathLike
from math import sqrt, ceil
from functools import cache

import torch
from torchvision.transforms import PILToTensor, Compose, ToPILImage, Normalize
from transformers import logging
import numpy as np
import pytesseract

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
    shrink_mask,
    copy_list,
    rand_mask
)
from .config import (
    PIXEL_DEFAULT_IMAGE_STD,
    PIXEL_DEFAULT_IMAGE_MEAN,
    RenderConfig
)

from pixel import Encoding, PIXELForPreTrainingOutput

logger = logging.get_logger(__name__)

@cache
def square_number(num: int) -> tuple[int, int]:
    upper = int(sqrt(num)) + 1
    n1, n2 = -99999, 99999
    for i in range(1, upper):
        j = num // i
        if i * j == num:
            if j - i < n2 - n1:
                n1, n2 = i, j
                
    return n1, n2


class TGraph:
    
    # The default image format, [n, c, h, w] or [c, h, w], varies in [0, 1]
    _value: torch.Tensor | None = None
    _patch_size: int | None = None
    _patch_len: int = 1
    text: str | list[str] | None = None
    _num_text_patches: int | list[int] | None = None

    _attention_mask: torch.Tensor | None = None
    _text_mask: torch.Tensor | None = None
    _patch_mask: torch.Tensor | None = None
    
    _num_gen_patches: int | list[int] | None = None
    _half: bool = False
    _predicts: torch.Tensor | None = None
    _labels: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    device: Any = None

    # useful transforms
    _pil2val = Compose([PILToTensor()])
    _val2pil = Compose([ToPILImage(mode='RGB')])
    _pix_normalizer = Compose([Normalize(PIXEL_DEFAULT_IMAGE_MEAN, PIXEL_DEFAULT_IMAGE_STD)])
    _inv_pix_normalizer = Compose([Normalize(- PIXEL_DEFAULT_IMAGE_MEAN / PIXEL_DEFAULT_IMAGE_STD, 1 / PIXEL_DEFAULT_IMAGE_STD)])

    @classmethod
    def from_tgraph(cls, graph: TGraph, with_value: bool = False) -> TGraph:
        new = TGraph()
        new.device = graph.device
        new._half = graph._half
        if with_value:
            new._value = graph.value
        new._patch_size = graph.patch_size
        new._patch_len = graph.patch_len
        new.text = graph.text
        new._num_text_patches = graph.num_text_patches
        new._attention_mask = graph.attention_mask
        new._text_mask = graph._text_mask
        new._patch_mask = graph.patch_mask
        new._num_gen_patches = graph.num_gen_patches
        new._predicts = graph.predcits
        new._labels = graph.labels
        
        return new
    
    def detach_(self) -> TGraph:
        self._value = self._value.detach_()
        if self._attention_mask is not None:
            self._attention_mask = self._attention_mask.detach_()
        if self._text_mask is not None:
            self._text_mask = self._text_mask.detach_()
        if self._patch_mask is not None:
            self._patch_mask = self._patch_mask.detach_()
        if self._predicts is not None:
            self._predicts = self._predicts.detach_()
        if self._labels is not None:
            self._labels = self._labels.detach_()
        if self.loss is not None:
            self.loss = self.loss.detach_()
        
        return self
    
    def __getitem__(self, idx: int | slice) -> TGraph:
        result = TGraph()
        
        if isinstance(idx, slice):
            bidx = idx.start * self.patch_size * self.patch_len
            eidx = self.patch_len * self.patch_size * idx.stop
        else:
            bidx = idx * self.patch_size * self.patch_len
            eidx = bidx + self.patch_len * self.patch_size
        
        result._value = self.value[..., :, bidx:eidx]
        result._patch_size = self._patch_size
        result._patch_len = self._patch_len
        return result

    @staticmethod
    def init_render(
        dpi: int = 120,
        font_size: int = 8,
        pixels_per_patch: int = 16,
        pad_size: int = 3,
        font_file: str = 'GoNotoCurrent.ttf',
        path: str = 'storage/pixel-base',
        rgb: bool = True,
        binary: bool = False,
        max_seq_length: int = 529,
        mask_ratio: float = 0.25,
        patch_len: int = 1,
        num_workers: int = 1
    ) -> None:
        TGraph._patch_len = patch_len
        config = RenderConfig(
            dpi=dpi,
            font_size=font_size,
            pixels_per_patch=pixels_per_patch,
            pad_size=pad_size,
            font_file=font_file,
            path=path,
            rgb=rgb,
            mask_ratio=mask_ratio,
            max_seq_length=max_seq_length,
            binary=binary
        )
        init_render(config, num_worker=num_workers)
        return
    
    @property
    def patch_len(self) -> int:
        return self._patch_len
    
    @patch_len.setter
    def patch_len(self, length: int) -> None:
        if get_num_patches() % length != 0:
            raise ValueError(f'The sequence length {get_num_patches()} is not devidible by the patch length {length}.')
        
        self._patch_len = length
    
    @property
    def predcits(self) -> torch.Tensor:
        global logger
        if self.labels is None:
            return None
        label_type = self.labels.dtype
        if label_type == torch.float or label_type == torch.float64:
            logger.warning_once('Detect float labels, do not call argmax while predict')
            return self.value
        if self._predicts is None:
            logger.warning_once('Detect integer labels, call argmax to predict labels')
            self._predicts = self.value.argmax(-1)
        return self.process(self._predicts)
    
    @property
    def value(self) -> torch.Tensor:
        return self.process(self._value)
    
    @property
    def labels(self) -> torch.Tensor:
        return self.process(self._labels)
    
    @labels.setter
    def labels(self, labels: torch.Tensor) -> None:
        self._labels = labels
    
    def process(self, value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        if self.device is not None:
            value = value.to(self.device)
        if self._half:
            value = value.half()
        else:
            value = value.float()
        return value

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
    def text_mask(self) -> torch.Tensor:
        if self._text_mask is not None:
            return self.process(self._text_mask)
        
        mask = torch.clone(self.attention_mask)
        assert mask.dim() == 2
        inds = (mask.sum(-1) - 2).type(torch.long)
        num = mask.shape[0]
        mask.T[inds, list(range(num))] = 0
        self._text_mask = mask

        return self.process(mask)

    def _get_attention_mask(self) -> torch.Tensor:
        if self._attention_mask is not None:
            return self._attention_mask
        
        if isinstance(self.num_text_patches, int):
            self._attention_mask = get_attention_mask(self.num_text_patches, self.patch_len, get_num_patches())
            return self._attention_mask
        
        masks = [get_attention_mask(num, self.patch_len, get_num_patches()).unsqueeze(0) for num in self.num_text_patches]
        self._attention_mask = torch.cat(masks, dim=0)
        return self._attention_mask
    
    @property
    def attention_mask(self) -> torch.Tensor:
        return self.process(self._get_attention_mask())
    
    @attention_mask.setter
    def attention_mask(self, mask: torch.Tensor) -> None:
        if mask is None:
            self._attention_mask = None
            return
        self._attention_mask = torch.clone(mask)

    @property
    def num_text_patches(self) -> int | list[int]:
        if self._num_text_patches is not None:
            return self._num_text_patches
        
        if self._attention_mask is not None:
            self._num_text_patches = (self._attention_mask.sum(-1) - 1).tolist()
        
        return self._num_text_patches
    
    @num_text_patches.setter
    def num_text_patches(self, num: int | list[int]) -> None:
        if num is None:
            self.num = None
            return
        if isinstance(num, torch.Tensor):
            num = num.tolist()
            self._num_text_patches = num
            return
        self._num_text_patches = copy_list(num)

    def clamp(self) -> TGraph:
        self._value = self._value.clamp(0, 1)
        return self

    # @property
    # def is_squre(self) -> bool:
    #     sp = self._value.shape
    #     return sp[-1] == sp[-2]
    
    @property
    def gen_begin_idx(self) -> int | list[int]:
        if isinstance(self.num_text_patches, list):
            idx = [n - g for n, g in zip(self.num_text_patches, self.num_gen_patches)]
        else:
            idx = self.num_text_patches - self.num_gen_patches
        return idx
    
    def init_patch_mask(self, mode: str, num: int = 1, idx: int = -1, ratio: float = 0.25) -> torch.Tensor:
        match mode:
            case 'rand':
                self._patch_mask = rand_mask(self.attention_mask.shape, ratio)
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
    
    def _get_patch_mask(self) -> torch.Tensor:
        if self._patch_mask is not None:
            return self._patch_mask
        
        return self.init_patch_mask('span')
    
    @property
    def patch_mask(self):
        return self.process(self._get_patch_mask())
    
    @patch_mask.setter
    def patch_mask(self, mask: torch.Tensor) -> None:
        if mask is None:
            self._patch_mask = None
            return
        self._patch_mask = torch.clone(mask)

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
        if num is None:
            self._num_gen_patches = None
            return
        self._num_gen_patches = copy_list(num)

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
        graph.attention_mask = attention_mask
        graph.patch_mask = patch_mask
        graph.num_text_patches = num_text_patches
        graph.num_gen_patches = num_gen_patches
        graph.loss = loss

        return graph
    
    # paint the masked patchs with color or noise
    @torch.no_grad()
    def paint_mask(self, color: str | tuple[float, float, float] = 'green', alpha: float = 0.1, mask: torch.Tensor = None) -> TGraph:                
        height = self.patch_size
        width = height * get_num_patches()

        if mask is None:
            mask = mask2img(self.attention_mask * self.patch_mask, self.patch_size)
        else:
            mask = mask2img(mask, self.patch_size)
        backcolor = color_image(color, height, width)

        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        masked_value = (self._value * mask) * (1 - alpha) + backcolor * mask * alpha
        un_masked = self._value * (1 - mask)
        self._value = masked_value + un_masked

        return self
    
    @torch.no_grad()
    def circle_mask(self, color: str | tuple[float, float, float] = 'green', alpha: float = 0.1, width: int = 2, mask: torch.Tensor = None) -> TGraph:
        height = self.patch_size
        w = height * get_num_patches()

        if mask is None:
            mask = self.attention_mask * self.patch_mask
        mask = mask2img(mask, self.patch_size)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        grid_mask = gen_circle_mask(height, get_num_patches(), width) * mask
        color = color_image(color, height, w)
        grid = grid_mask * color
        
        grid_value = (self.value * (1 - alpha) + grid * alpha) * grid_mask
        self._value = self.value * (1 - grid_mask) + grid_value

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

    def to_PIL(self, square: bool=True) -> Image | list[Image]:
        if square:
            value = self._squarelize(self._value)
        else:
            value = self.value
        
        if value.dim() == 3:
            return self._to_PIL(value)
        
        return [self._to_PIL(img) for img in value]
    
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
        return self.value * 2 - 1
    
    def set_device(self, device: Any) -> TGraph:
        self.device = device
        return self
    
    @classmethod
    def from_pixel_img(cls, img: torch.Tensor, do_clip: bool = False, attention_mask: torch.Tensor | None = None, patch_mask: torch.Tensor | None = None) -> TGraph:
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
        return graph

    @classmethod
    def from_pixel(cls, output: PIXELForPreTrainingOutput, do_clamp: bool = False, patch_size: int = 16) -> TGraph:
        graph = cls.from_pixel_logits(output.logits, do_clamp=do_clamp, patch_size=patch_size)
        graph.attention_mask = output.attention_mask
        graph.patch_mask = output.mask
        graph.loss = output.loss
        return graph

    def to_pixel(self) -> torch.Tensor:
        return self._pix_normalizer(self.value)
    
    @classmethod
    def from_file(cls, path: str | PathLike) -> TGraph:
        image = PIL.Image.open(path, 'r')
        return cls.from_PIL(image)
    
    def _to_file(self, path: str | PathLike, value: torch.Tensor) -> None:
        img = self._val2pil(value.clamp(0, 1))
        img.save(path, 'PNG')

    def to_file(self, path: str | PathLike, square: bool=True) -> None:
        if square:
            value = self._squarelize(self.value)
        else:
            value = self.value
                    
        if value.dim() == 3:
            return self._to_file(path, self._value)
        
        os.makedirs(path, exist_ok=True)
        for idx, val in enumerate(value):
            file_path = os.path.join(path, f'{idx}.png')
            self._to_file(file_path, val)

    @classmethod
    def from_text(cls, text: str | list[str], **kwargs) -> TGraph:
        encods = render_text(text, **kwargs)
        graph = cls()
        if isinstance(encods, Encoding):
            graph._value = torch.tensor(encods.pixel_values / 255, dtype=torch.float).permute(2, 0, 1)
            graph.num_text_patches = ceil((encods.num_text_patches + 1) / cls._patch_len) # Add 1 for [SEP] token (black patch)
            graph.text = text
            return graph
        
        imgs = [torch.tensor(encod.pixel_values / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0) for encod in encods]
        graph._value = torch.cat(imgs, dim=0).contiguous()

        nums = [ceil((encod.num_text_patches + 1) / cls._patch_len) for encod in encods]
        graph.num_text_patches = nums
        graph.text = text

        return graph
    
    def _squarelize(self, value: torch.Tensor) -> torch.Tensor:
        np = self._value.shape[-1] // self.patch_size
        nrows, _ = square_number(np)
        
        rows = torch.tensor_split(value, nrows, dim=-1)
        square = torch.cat(rows, dim=-2).contiguous()
        
        return square
    
    def show(self, square: bool = True) -> Image | list[Image]:
        if square:
            value = self._squarelize(self.value)
        else:
            value = self.value
        
        if value.dim() == 3:
            return self._val2pil(value)
        
        ims = []
        for val in value:
            ims.append(self._val2pil(val))
            
        return ims
    
    # def squarelize(self) -> TGraph:
    #     if self.is_squre:
    #         return self
        
    #     self._patch_size = self._value.shape[-2]
    #     num_rows = int(sqrt(self._value.shape[-1] // self._patch_size))
    #     rows = torch.tensor_split(self._value, num_rows, dim=-1)
    #     self._value = torch.cat(rows, dim=-2)

    #     return self
    
    # def unsquarelize(self, patch_size: int | None = None) -> TGraph:
    #     if not self.is_squre:
    #         return self
        
    #     if patch_size:
    #         self._patch_size = patch_size

    #     if self._patch_size is None:
    #         self._patch_size = self.patch_size

    #     num_rows = self._value.shape[-2] // self._patch_size
    #     rows = torch.tensor_split(self._value, num_rows, dim=-2)
    #     self._value = torch.cat(rows, dim=-1)
    
    #     return self
    
    @classmethod
    def reconstruct(cls, origin: TGraph, generated: TGraph, do_clamp: bool = False) -> TGraph:
        recon = cls()

        mask = origin.attention_mask * generated.patch_mask
        mask = mask2img(mask, origin.patch_size)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        mask = mask == 1

        recon._value = (
            origin.value * (~mask).int() + generated.value * mask.int()
        )
        if do_clamp:
            recon._value = recon.value.clamp_(0, 1)

        recon.attention_mask = origin.attention_mask
        recon.patch_mask = generated.patch_mask
        recon.num_text_patches = origin.num_text_patches
        recon.patch_size = origin.patch_size

        return recon
    
    @torch.no_grad()
    def ocr(self) -> str | list[str]:
        imgs = self.to_PIL()
        if isinstance(imgs, list):
            self.text = [pytesseract.image_to_string(img) for img in imgs]
        else:
            self.text = pytesseract.image_to_string(imgs)

        return self.text

    def half(self) -> TGraph:
        self._half = True
        return self

    def float(self) -> TGraph:
        self._half = False
        return self
