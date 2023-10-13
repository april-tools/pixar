from typing import Any, Callable
import os
from os import PathLike
from functools import wraps
import atexit
from time import strftime
import json
import random
import time
import math
import argparse

import torch
import torch.multiprocessing as mp
import numpy as np

from pixel import PangoCairoTextRenderer
from pixel import Encoding
from pixel import SpanMaskingGenerator

from .config import RenderConfig

render: PangoCairoTextRenderer = None
render_config = dict()
_render_processes: list[mp.Process] = []
_txt_queue = mp.Queue()
_img_queue = mp.Queue()
_timestamp: str = None
render_fn = None
span_masking_generator: SpanMaskingGenerator = None
binary = False

def seed_everyting(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def _rend_once(text: str | tuple[str, str]) -> Encoding:
    global binary, render
    result = render(text)
    if not render.rgb:
        result.pixel_values = np.tile(np.expand_dims(result.pixel_values, axis=-1), [1, 1, 3])
    if binary:
        result.pixel_values = (result.pixel_values > 255 / 2).astype(int) * 255

    return result

def _stand_alone_render(text: str | list[str]) -> Encoding | list[Encoding]:
    if isinstance(text, str) or isinstance(text, tuple):
        return _rend_once(text)
    
    return [_rend_once(txt) for txt in text]

def _render_process() -> None:
    global _txt_queue, _img_queue, render
    while True:
        txt = _txt_queue.get(timeout=600)
        img = _rend_once(text=txt)
        _img_queue.put(img, timeout=30)

def _parallel_rend(text: str | list[str]) -> Encoding | list[Encoding]:
    if isinstance(text, str) or isinstance(text, tuple):
        _txt_queue.put(text)
        return _img_queue.get(block=True, timeout=30)
    else:
        for txt in text:
            _txt_queue.put(txt)
        result = []
        for _ in range(len(text)):
            result.append(_img_queue.get(block=True, timeout=30))
    return result

def _clean_up() -> None:
    global _render_processes
    for process in _render_processes:
        print(f'Terminating {len(_render_process)} render processes...')
        process.terminate()
        process.join()
    print('All render process terminated.')

def set_render(render_to_set: Any) -> None:
    global render
    render = render_to_set

def init_render(
        config: RenderConfig,
        num_worker: int = 1,
        ) -> PangoCairoTextRenderer:
    global render, render_config, _render_processes, render_fn, span_masking_generator, binary
    with open(os.path.join(config.path, 'text_renderer_config.json'), 'r') as fin:
        render_config = json.load(fin)

    render_config['dpi'] = config.dpi
    render_config['font_size'] = config.font_size
    render_config['pixels_per_patch'] = config.pixels_per_patch
    render_config['font_file'] = os.path.join(config.path, config.font_file)
    render_config['pad_size'] = config.pad_size
    render_config['rgb'] = config.rgb
    render_config['max_seq_length'] = config.max_seq_length
    binary = config.binary

    render = PangoCairoTextRenderer(**render_config)
    span_masking_generator = SpanMaskingGenerator(
        num_patches=render.max_seq_length,
        num_masking_patches=math.ceil(config.mask_ratio * render.max_seq_length),
        max_span_length=6,
        spacing="span",
        cumulative_span_weights=[0.2, 0.4, 0.6, 0.8, 0.9, 1.0],
    )

    if num_worker > 1:
        for idx in range(num_worker):
            p = mp.Process(target=_render_process)
            p.start()
            _render_processes.append(p)
            print(f'Render process {idx} started.')
        render_fn = _parallel_rend
    else:
        render_fn = _stand_alone_render
    
    return render

def get_render() -> PangoCairoTextRenderer:
    if render is None:
        raise Exception('The text render has not been initialized!')
    return render
    
def render_text(text: str | list[str], **kwargs) -> Encoding | list[Encoding]:
    global render_fn
    return render_fn(text)

def get_span_mask(num_text_patches: int | list[int]) -> torch.Tensor:
    if isinstance(num_text_patches, list):
        return torch.cat([get_span_mask(num).unsqueeze(0) for num in num_text_patches], dim=0)
    return torch.tensor(span_masking_generator(num_text_patches + 1))
    
def init_timestamp() -> str:
    global _timestamp
    _timestamp = strftime('%Y%m%d-%H%M%S')

def timestamp() -> str:
    global _timestamp
    if _timestamp:
        return _timestamp
    _timestamp = strftime('%Y%m%d-%H%M%S')
    return _timestamp

def get_attention_mask(num_text_patches: int, seq_length: int | None = None):
    """
    Creates an attention mask of size [1, seq_length]
    The mask is 1 where there is text or a [SEP] black patch and 0 everywhere else
    """
    if seq_length is None:
        seq_length = get_num_patches()

    n = min(num_text_patches + 1, seq_length)  # Add 1 for [SEP] token (black patch)
    zeros = torch.zeros(seq_length)
    ones = torch.ones(n)
    zeros[:n] = ones
    return zeros

def get_patch_size() -> int:
    if render is None:
        raise Exception('The text render has not been initialized!')
    
    return render.max_pixels_len // render.max_seq_length

def get_num_patches() -> int:
    return render.max_seq_length

def patchify(imgs: torch.Tensor) -> torch.Tensor:
    """
    imgs: (N, 3, H, W) x: (N, L, patch_size**2 *3)
    """
    p = get_patch_size()
    assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

    h = imgs.shape[2] // p
    w = imgs.shape[3] // p
    x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    x = torch.einsum("nchpwq->nhwpqc", x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))

    return x

def unpatchify(x: torch.Tensor, p: int = None) -> torch.Tensor:
    """
    x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W)
    """
    if p is None:
        p = get_patch_size()
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

def timeit(fn: Callable) -> Callable:
    @wraps(fn)
    def inner_fn(*args, **kwargs) -> Any:
        tic = time.time()
        result = fn(*args, **kwargs)
        seconds = time.time() - tic
        print(f'{str(fn)} used {seconds} seconds.')
        return result
    return inner_fn

def mask2img(mask: torch.Tensor | list[int] | list[list[int]], patch_size: int) -> torch.Tensor:
    if isinstance(mask, list):
        mask = torch.tensor(mask)
    if mask.dim() == 2:
        num_mask = mask.shape[0]
    else:
        assert mask.dim() == 1
        num_mask = 1

    olen = mask.shape[-1]
    return mask \
        .reshape(-1) \
        .repeat(patch_size**2)  \
        .reshape(-1, olen * num_mask) \
        .transpose(-1, -2) \
        .reshape(-1, olen * patch_size, patch_size) \
        .transpose(-1, -2) \
        .squeeze() \
        .contiguous()

def _cons_mask(mask: torch.Tensor, begin_idx: int, num: int) -> torch.Tensor:
    max_len = mask.shape[-1]
    for idx in range(begin_idx, begin_idx + num):
        if idx >= max_len:
            break
        mask[idx] = 1
    return mask

def cons_mask(mask: torch.Tensor, begin_idx: int | list[int], num: int) -> torch.Tensor:
    if isinstance(begin_idx, int):
        return _cons_mask(mask, begin_idx, num)
    
    for m, b_idx in zip(mask, begin_idx):
        _cons_mask(m, b_idx, num)
    
    return mask

def _gen_circle_mask(patch_size: int, width: int = 1) -> torch.Tensor:
    mask = torch.zeros(patch_size, patch_size)
    for row_idx in range(patch_size):
        for column_idx in range(patch_size):
            if row_idx < width or row_idx >= patch_size - width or column_idx < width or column_idx >= patch_size - width:
                mask[row_idx, column_idx] = 1
    return mask

def gen_circle_mask(patch_size: int, num_patches: int, width: int = 1) -> torch.Tensor:
    pattern = _gen_circle_mask(patch_size, width)
    mask = pattern.reshape(-1).repeat(num_patches).reshape(patch_size * num_patches, patch_size).transpose(-1, -2).contiguous()
    return mask

def color_image(color: str | tuple[float, float, float], height: int, width: int) -> torch.Tensor:
    if isinstance(color, str):
        match color:
            case 'white':
                color = (1, 1, 1)
            case 'black':
                color = (0, 0, 0)
            case 'green':
                color = (0, 1, 0)
            case 'red':
                color = (1, 0, 0)
            case 'blue':
                color = (0, 0, 1)
            case 'orange':
                color = (1, 0.647, 0)
            case 'yellow':
                color = (1, 1, 0)
            case 'brown':
                color = (150/255, 75/255, 0)
            case 'noise':
                color = None
            case _:
                raise KeyError(f'Unsupported color {color}')
    if color:
        backcolor = torch.tensor(color, dtype=torch.float).repeat(height, width, 1).permute(2, 0, 1).squeeze()
    else:
        backcolor = torch.randn(3, height, width) + 0.5

    return backcolor

def _shrink_mask(mask: torch.Tensor) -> torch.Tensor:
    pre = 1
    for idx in range(len(mask) - 1):
        cur = mask[idx].item()
        nex = mask[idx + 1]
        if pre == 0 or nex == 0:
            mask[idx] = 0
        pre = cur
    if mask[-1] == 1 and cur == 0:
        mask[-1] = 0
    return mask

def shrink_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.dim() == 1:
        return _shrink_mask(mask)
    for idx in range(len(mask)):
        mask[idx] = _shrink_mask(mask[idx])

    return mask

def _copy_list(origin: list[int]) -> list[int]:
    if isinstance(origin, list):
        return [o for o in origin]
    else:
        return origin

def copy_list(origin: list) -> list:
    if not isinstance(origin, list):
        return origin
    if isinstance(origin[0], list):
        return [_copy_list(o) for o in origin]
    return _copy_list(origin)

def rand_mask(shape: torch.Size, ratio: float):
    return (torch.rand(shape) < ratio).int()

def str2bool(txt: str | list[str] | Any) -> bool | list[bool] | Any:
    if isinstance(txt, list):
        return [str2bool(t) for t in txt]
    
    if isinstance(txt, str):
        if txt.lower().strip() == 'true':
            return True
        if txt.lower().strip() == 'false':
            return False
    
    return txt

def params2dict(params: dict) -> dict:
    parser = argparse.ArgumentParser()

    # add args to the argparse
    for k, v in params.items():
        if k[0] == '_':
            continue
        if isinstance(v, list):
            parser.add_argument(f'--{k}', type=type(v[0]) if not isinstance(v[0], bool) else str, default=v, required=False, nargs='+')
            continue
        parser.add_argument(f'--{k}', type=type(v) if not isinstance(v, bool) else str, default=v, required=False)

    # update the parsed arguments
    args = vars(parser.parse_args())
    params.update(args)
    for k, v in params.items():
        params[k] = str2bool(v)
    return params

atexit.register(_clean_up)
