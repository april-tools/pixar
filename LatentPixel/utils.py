from typing import Any
from os import PathLike
from time import strftime

import torch

from pixel import PangoCairoTextRenderer
from pixel import Encoding

render: PangoCairoTextRenderer = None
_timestamp: str = None

def set_render(render_to_set: Any) -> None:
    global render
    render = render_to_set

def init_render(path: str | PathLike, **kwargs) -> PangoCairoTextRenderer:
    global render
    render = PangoCairoTextRenderer.from_pretrained(path, rgb=True, **kwargs)
    return render

def get_render() -> PangoCairoTextRenderer:
    if render is None:
        raise Exception('The text render has not been initialized!')
    return render
    
def render_text(text: str | list[str], **kwargs) -> Encoding | list[Encoding]:
    if isinstance(text, str):
        return render(text=text, **kwargs)
    
    return [render(txt, **kwargs) for txt in text]

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

def unpatchify(x: torch.Tensor) -> torch.Tensor:
    """
    x: (N, L, patch_size**2 *3) imgs: (N, 3, H, W)
    """
    p = get_patch_size()
    h = w = int(x.shape[1] ** 0.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum("nhwpqc->nchpwq", x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs