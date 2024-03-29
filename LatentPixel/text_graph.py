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
from .ocr import ocr

from pixel import Encoding, PIXELForPreTrainingOutput

logger = logging.get_logger(__name__)

BINARY_THRESHOLD = 0.5

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
    _binary: bool = False

    _attention_mask: torch.Tensor | None = None
    _text_mask: torch.Tensor | None = None
    _patch_mask: torch.Tensor | None = None
    
    _num_gen_patches: int | list[int] | None = None
    _half: bool = False
    _predicts: torch.Tensor | None = None
    _labels: torch.Tensor | None = None
    _contoru: torch.Tensor = ...
    loss: torch.Tensor | None = None
    device: Any = None

    # useful transforms
    _pil2val = Compose([PILToTensor()])
    _val2pil = Compose([ToPILImage(mode='RGB')])
    _val2pil_binary = Compose([ToPILImage(mode=None)])
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
        new._labels = graph._labels
        
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
        
        result._value = self._value[..., :, bidx:eidx]
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
        TGraph._binary = binary
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
            binary=binary,
            patch_len=patch_len
        )
        return init_render(config, num_worker=num_workers)
    
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
            return self._value
        if self._predicts is None:
            logger.warning_once('Detect integer labels, call argmax to predict labels')
            self._predicts = self._value.argmax(-1)
        if self.device is not None:
            return self._predicts.to(self.device)
        return self._predicts
    
    @property
    def value(self) -> torch.Tensor:
        return self.process(self._value)
    
    @property
    def labels(self) -> torch.Tensor:
        if self.device is not None:
            return self._labels.to(self.device)
        else:
            return self._labels
    
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
        inds = (mask.sum(-1) - 1).type(torch.int)
        num = mask.shape[0]
        mask[torch.arange(0, num), inds] = 0
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
            if isinstance(self._num_text_patches, int):
                return self._num_text_patches
            return [num for num in self._num_text_patches]
        
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
        # return self.process(self._get_patch_mask())
        return None
    
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

    def _to_PIL(self, value: torch.Tensor, span: bool=False, span_ratio: float=1.2, binary_method: str='threshold', threshold: float = BINARY_THRESHOLD) -> Image:
        if span:
            value = self._span(value, span_ratio)
        if self._binary:
            match binary_method:
                case 'threshold':
                    value = value > threshold
                case 'bernoulli':
                    assert value.max() <= 1 and value.min() >= 0
                    value = torch.bernoulli(value)
                case 'gray_scale':
                    ...
                case _:
                    raise KeyError(f"Not support {binary_method}, please use 'threshold', 'bernoulli', or 'gray_scale' instead.")
            return self._val2pil_binary(value.float().clamp(0, 1))
        return self._val2pil(value.clamp(0, 1))

    def to_PIL(self, square: bool=True, contour: float = 0, span: bool=False, span_ratio: float=1.2, binary_method: str='threshold', threshold: float=BINARY_THRESHOLD) -> Image | list[Image]:
        value = self._value
        if contour > 0:
            if self._contoru is ...:
                self._contoru = self._gen_contour()
            value = value.repeat([1, 3, 1, 1])
            value = value.float() + self._contoru * contour
            value[:, 1, :, :] += self._contoru * contour

        if square:
            value = self._squarelize(value)
        
        if value.dim() == 3:
            return self._to_PIL(value, span=span, span_ratio=span_ratio, binary_method=binary_method, threshold=threshold)
        
        return [self._to_PIL(img, span=span, span_ratio=span_ratio, binary_method=binary_method, threshold=threshold) for img in value]
    
    def _gen_contour(self) -> torch.Tensor:
        contour = torch.zeros(self.patch_size, get_num_patches() * self.patch_size, dtype=torch.float)
        width = self.patch_len * self.patch_size
        print(width)
        h = self._value.shape[-2]
        w = self._value.shape[-1]
        for i in range(h):
            for j in range(w):
                if i % h == 0:
                    contour[i, j] = 1
                elif j % width == 0:
                    contour[i, j] = 1
        return contour
    
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
    
    def _to_file(self, path: str | PathLike, value: torch.Tensor, binary_method: str='threshold', threshold:float=BINARY_THRESHOLD) -> None:
        img = self._to_PIL(value, binary_method=binary_method, threshold=threshold)
        img.save(path, 'PNG')

    def to_file(self, path: str | PathLike, square: bool=True, contour: float=0.0, binary_method: str='threshold', threshold:float=BINARY_THRESHOLD) -> None:
        value = self.value
        if contour > 0:
            if self._contoru is ...:
                self._contoru = self._gen_contour()
            value = value.repeat([1, 3, 1, 1])
            value = value.float() + self._contoru * contour
            value[:, 1, :, :] += self._contoru * contour
            
        if square:
            value = self._squarelize(value)
        else:
            value = self.value
                    
        if value.dim() == 3:
            return self._to_file(path, value, binary_method=binary_method, threshold=threshold)
        
        os.makedirs(path, exist_ok=True)
        for idx, val in enumerate(value):
            file_path = os.path.join(path, f'{idx}.png')
            self._to_file(file_path, val, binary_method=binary_method, threshold=threshold)
            
    def binary(self, method: str='threshold', threshold=BINARY_THRESHOLD) -> None:
        if self._value.dim() == 3:  # c, h, w
            val = self._value.mean(dim=0, keepdim=True)
        elif self._value.dim() == 4:    # bs, c, h, w
            val = self._value.mean(dim=1, keepdim=True)
        
        match method:
            case 'threshold':
                val = (val > threshold).long()
            case 'bernoulli':
                val = torch.bernoulli(val)
            case _:
                ...
        self._value = val

    @classmethod
    def from_text(cls, text: str | list[str], **kwargs) -> TGraph:
        encods = render_text(text, **kwargs)
        graph = cls()
        graph.patch_len = cls._patch_len
        graph._binary = cls._binary
        if isinstance(encods, Encoding):
            graph._value = torch.tensor(encods.pixel_values / 255, dtype=torch.float).permute(2, 0, 1)
            graph.binary() if graph._binary else ...
            graph.num_text_patches = ceil((encods.num_text_patches + 1) / cls._patch_len) # Add 1 for [SEP] token (black patch)
            graph.text = text
            return graph
        
        imgs = [torch.tensor(encod.pixel_values / 255, dtype=torch.float).permute(2, 0, 1).unsqueeze(0) for encod in encods]
        graph._value = torch.cat(imgs, dim=0).contiguous()
        graph.binary() if graph._binary else ...

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
    def _span(self, value: torch.Tensor, span_ratio: float) -> torch.Tensor:
        '''
        Span the image into a large image, to have better ocr effect
        image values are within the range [0, 1]
        '''
        assert span_ratio > 1, f'span_ratio must larger than 1, <{span_ratio}> found'
        if value.dim() == 4:
            b, c, h, w = value.shape
            sp = int(max(w, h) * span_ratio)
            canvas = torch.ones([b, c, sp, sp], dtype=value.dtype)
        elif value.dim() == 3:
            c, h, w = value.shape
            b = None
            sp = int(max(w, h) * span_ratio)
            canvas = torch.ones([c, sp, sp], dtype=value.dtype)


        bh = sp // 2 - h // 2
        eh = bh + h
        bw = sp // 2 - w // 2
        ew = bw + w
        canvas[..., :, bh:eh, bw:ew] = value
        return canvas
    
    @torch.no_grad()
    def ocr(self, square: bool=False, method: str = 'PaddleOCR', span: bool=False, span_ratio: float=1.2, scale: float=3.0, sample_method: str='threshold', threshold: float=BINARY_THRESHOLD) -> str | list[str]:
        """Ocr the image

        Args:
            square (bool, optional): Whether to reshape the image to a square. Defaults to False.
            method (str, optional): "PaddleOCR" or "tesseract". Defaults to 'PaddleOCR'.
            span (bool, optional): Whether to span the image into a square iamge by filling white space. Defaults to False.
            span_ratio (float, optional): How large we enlarge the original image. Defaults to 1.2.
            scale (float, optional): Resize the image by the scale. Defaults to 3.0.

        Returns:
            str | list[str]: Recognized texts
        """
        imgs = self.to_PIL(square=square, span=span, span_ratio=span_ratio, binary_method=sample_method, threshold=threshold)
        return ocr(imgs, method=method, scale=scale)

    def half(self) -> TGraph:
        self._half = True
        return self

    def float(self) -> TGraph:
        self._half = False
        return self

    @property
    def patch_width(self) -> int:
        return self.patch_size * self.patch_len

    @torch.no_grad()
    def _shift_text(self, sidx: int, shift: int) -> TGraph:
        """
        Shift text patches forward (positive) or backward (negative) with specified number of pixels
        """
        widx = (self.num_text_patches[sidx] - 1) * self.patch_width - 1    # index of the last pixel of text patches
        text_pixels = self._value[sidx, :, :, max(0, -shift):min(widx+1, widx+1-shift)].clone()
        self._value[sidx, :, :, 0:widx+1] = 1
        self._value[sidx, :, :, max(shift, 0):min(widx+1, widx+1+shift)] = text_pixels
        return self
    
    @torch.no_grad()
    def count_white_space(self, sidx: int) -> int:
        """
        Count how many white pixels at the end of the text before the black square
        """
        num = 0
        widx = (self.num_text_patches[sidx] - 1) * self.patch_width - 1    # index of the last pixel of text patches
        while widx >= 0:
            if self._value[sidx, :, :, widx].float().mean().item() >= 0.999999:
                num += 1
                widx -= 1
            else:
                break
        return num

    @torch.no_grad()
    def _spacing_text(self, space_len: int) -> TGraph:
        for idx in range(self._value.shape[0]):
            if self.num_text_patches[idx] <= 1:
                continue
            space = self.count_white_space(idx)
            self._shift_text(idx, space-space_len)
        return self

    def bernoulli(self) -> TGraph:
        assert self._value.max() <= 1 and self._value.min() >= 0
        self._value = torch.bernoulli(self._value)
        return self
