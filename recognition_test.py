import os
from os import PathLike
from dataclasses import dataclass, field, asdict
from LatentPixel import (
    get_pixel_pretrain_dataloader,
    EditDistance,
    params2dict,
    timeit,
    TGraph,
    RenderConfig
)

from diffusers import AutoencoderKL
from pandas import DataFrame as df
from tqdm import tqdm
import torch

@dataclass
class RecgnitionConfig:

    render_path: str | os.PathLike = 'storage/pixel-base'
    font: list[str] = field(default_factory=lambda: ['GoNotoCurrent.ttf'])
    dpi: list[int] = field(default_factory=lambda: [120])
    pixels_per_patch: list[int] = field(default_factory=lambda: [16])
    num_patches: int = 529
    font_size: list[int] = field(default_factory=lambda: [8])
    rgb: list[bool] = field(default_factory=lambda: [True])
    binary: list[bool] = field(default_factory=lambda: [False])
    coder_path: str = ''
    dataset_path: list[str] = field(default_factory=lambda: ['storage/enwiki', 'storage/bookcorpus'])
    seed: int = 42

def single_test(records: dict, render_path: str | os.PathLike, font: str, dpi: int, patch_size: int, font_size: int, dataset_path: list[str], rgb: bool, binary: bool, seed: int, coder: AutoencoderKL | None = None) -> dict:
    rconfig = RenderConfig(
        dpi=dpi,
        font_size=font_size,
        pixels_per_patch=patch_size,
        font_file=font,
        path=render_path,
        rgb=rgb,
        binary=binary
    )
    dataloader = get_pixel_pretrain_dataloader(
        paths=dataset_path,
        batch_size=8,
        num_workers=4,
        seed=seed,
        mask_ratio=0.25,    # no use
        mask_type='span',   # no use
        render_config=rconfig,
        min_len=200,
        max_len=600,
        streaming=True
    )
    edits = EditDistance()
    it = iter(dataloader)
    for _ in tqdm(range(10)):
        batch = next(it)
        batch: TGraph
        golden = batch.text
        if coder is not None:
            coder: AutoencoderKL
            batch.set_device('cuda')
            with torch.no_grad():
                result = coder.forward(batch.to_SD().half()).sample
                result = TGraph.from_SD(result.to('cpu').float(), True)
            recon = result.ocr()
        else:
            recon = batch.ocr()
        print(golden[0])
        print(recon[0])
        edits.accumulate(golden, recon)

    records['font'] += [font]
    records['dpi'] += [dpi]
    records['patch_size'] += [patch_size]
    records['font_size'] += [font_size]
    records['rgb'] += [rgb]
    records['binary'] += [binary]
    records['coder'] += [type(coder) if coder is not None else None]
    records['num_char'] += [edits.num_char]
    records['sum_dist'] += [edits.sum_dist]
    records['dist_ratio'] += [edits.average_dist()]

    return records


if __name__ == '__main__':
    config = RecgnitionConfig(**params2dict(asdict(RecgnitionConfig())))
    print(config)

    if len(config.coder_path) > 0:
        print('Load the coder')
        coder: AutoencoderKL = AutoencoderKL.from_pretrained(config.coder_path)
        coder.half()
        coder.to('cuda')
        coder.eval()
    else:
        print('No coder loaded')
        coder = None
    
    records = {
        'font': [],
        'dpi': [],
        'patch_size': [],
        'font_size': [],
        'rgb': [],
        'binary': [],
        'coder': [],
        'num_char': [],
        'sum_dist': [],
        'dist_ratio': []
    }
    l = len(config.dpi)
    assert l == len(config.pixels_per_patch)
    assert l == len(config.font_size)
    assert l == len(config.rgb)
    assert l == len(config.binary)
    assert l == len(config.font)


    for dpi, patch_size, font_size, rgb, binary, font in zip(config.dpi, config.pixels_per_patch, config.font_size, config.rgb, config.binary, config.font):
        records = single_test(
            records=records,
            render_path=config.render_path,
            font=font,
            dpi=dpi,
            patch_size=patch_size,
            font_size=font_size,
            dataset_path=config.dataset_path,
            rgb=rgb,
            seed=config.seed,
            coder=coder,
            binary=binary
        )

    result = df(data=records)
    result.to_csv('recon.csv', sep ='\t')
