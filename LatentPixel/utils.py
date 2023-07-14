from typing import Any
import os
from os import PathLike
import atexit
from time import strftime
import json

import torch
import torch.multiprocessing as mp

from pixel import PangoCairoTextRenderer
from pixel import Encoding

render: PangoCairoTextRenderer = None
render_config = dict()
_render_processes: list[mp.Process] = []
_txt_queue = mp.Queue()
_img_queue = mp.Queue()
_timestamp: str = None
render_fn = None

def _stand_alone_render(text: str | list[str]) -> Encoding | list[Encoding]:
    if isinstance(text, str):
        return render(text)
    
    return [render(txt) for txt in text]

def _render_process() -> None:
    global _txt_queue, _img_queue, render
    while True:
        txt = _txt_queue.get(timeout=600)
        img = render(text=txt)
        _img_queue.put(img, timeout=30)

def _parallel_rend(text: str | list[str]) -> Encoding | list[Encoding]:
    if isinstance(text, str):
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

{
  "background_color": "white",
  "dpi": 120,
  "font_color": "black",
  "font_file": "GoNotoCurrent.ttf",
  "font_size": 8,
  "max_seq_length": 529,
  "pad_size": 3,
  "pixels_per_patch": 16,
  "text_renderer_type": "TextRenderer"
}

def init_render(
        path: str | PathLike, 
        dpi: int = 120, 
        font_size: int = 8,
        pixels_per_patch: int = 16,
        pad_size: int = 3,
        num_worker: int = 1,
        ) -> PangoCairoTextRenderer:
    global render, render_config, _render_processes, render_fn
    with open(os.path.join(path, 'text_renderer_config.json'), 'r') as fin:
        render_config = json.load(fin)

    render_config['dpi'] = dpi
    render_config['font_size'] = font_size
    render_config['pixels_per_patch'] = pixels_per_patch
    render_config['font_file'] = os.path.join(path, 'GoNotoCurrent.ttf')
    render_config['pad_size'] = pad_size
    render_config['rgb'] = True

    render = PangoCairoTextRenderer(**render_config)

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

atexit.register(_clean_up)
