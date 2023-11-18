'''
evaluate the accuracy of reconstructed iamages by CNNAutoencoder.
'''

import os

import torch
from torch.utils.data import DataLoader
from LatentPixel import (
    get_pixel_pretrain_dataloader,
    params2dict,
    CNNAutoencoder,
    RenderConfig,
    TGraph
)
from tqdm import tqdm

params = {
    'model_paths': ['none'],
    'data_paths': ['storage/bookcorpus', 'storage/enwiki'],
    'dpi': 80,
    'font_size': 8,
    'pixels_per_patch': 8,
    'rgb': False,
    'binary': True,
    'font_file': 'PixeloidSans-mLxMm.ttf',
    'patch_len': 5,
    'max_seq_len': 2000,
    'num_batch': 50
}

def get_dataloader(params: dict) -> DataLoader:
    rconf = RenderConfig(
        dpi=params['dpi'],
        font_size=params['font_size'],
        pixels_per_patch=params['pixels_per_patch'],
        patch_len=params['patch_len'],
        font_file=params['font_file'],
        rgb=params['rgb'],
        binary=params['binary'],
        max_seq_length=params['max_seq_len']
    )

    return get_pixel_pretrain_dataloader(
        paths=params['data_paths'],
        batch_size=16,
        num_workers=8,
        seed=42,
        mask_ratio=0.5,
        mask_type='span',
        render_config=rconf,
        min_len=400,
        max_len=900,
        streaming=True,
        rank=0,
        world_size=1
    )

def test(params: dict, model_path: str | os.PathLike) -> float:
    print('Begin to load the dataset')
    loader = iter(get_dataloader(params))
    print('data loaded!')
    model = CNNAutoencoder(model_path)
    model.eval()
    model.cuda()
    num_correct = 0
    num_total = 0

    for i in tqdm(range(params['num_batch'])):
        img: TGraph = next(loader)
        img.set_device('cuda')
        r = model.forward(img)

        comp = r.value == img.value
        num_correct += comp.sum()
        num_total += comp.numel()

    return num_correct / num_total

if __name__ == '__main__':
    params = params2dict(params)
    print(params)
    for path in params['model_paths']:
        acc = test(params, path)
        print(f'Model: {path} \n Acc: {acc}')
