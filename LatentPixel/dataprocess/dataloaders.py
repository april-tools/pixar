import os
from functools import partial
import random

import torch
from torch.utils.data import DataLoader
from ..text_graph import TGraph
from ..config import ModelType, RenderConfig
from ..utils import seed_everyting, timeit
from tqdm import tqdm
import numpy as np

from datasets import load_dataset, interleave_datasets, load_from_disk
from datasets.distributed import split_dataset_by_node

def dataloader_init_fn(worker_id, seed: int, render_config: RenderConfig) -> None:
    seed_everyting(seed)
    os.system("taskset -p 0xffffffffff %d" % os.getpid())
    from ..utils import render, init_render
    if render is None:
        print(f'initialize the render with parameters {render_config.to_dict()}')
        init_render(render_config)

def render_batched_text(batch: list[dict[str, str]], mask_ratio: float, mask_type: str) -> torch.Tensor:
    sents = []
    for sent in batch:
        sents.append(sent['text'])
    img = TGraph.from_text(sents)
    img.init_patch_mask(mask_type, ratio=mask_ratio)
    img.patch_mask
    img.attention_mask

    return img
        
def collate_text(batch: dict, min_len: int, max_len: int | None = None) -> list[str]:
    texts: list[str] = batch['text']
    splitted = []
    for text in texts:
        splitted += text.strip().split('\n')

    results = []
    block = []
    length = 0
    for sent in splitted:
        length += len(sent)
        block += [sent]
        if length >= min_len:
            length = 0
            results.append(' '.join(block))
            block = []
    
    if len(block) > 0:
        results.append(' '.join(block))

    if max_len is not None:
        chunked = [sent[:max_len] for sent in results]
    else:
        chunked = results

    return {'text': chunked}

def get_pixel_pretrain_dataloader(
        paths: list[str | os.PathLike],
        batch_size: int, 
        num_workers: int, 
        seed: int,
        mask_ratio: float,
        mask_type: str,
        render_config: RenderConfig = None,
        n_skip: int = 0,
        min_len: int = 400,
        max_len: int | None = None,
        streaming: bool = True,
        rank: int = None, 
        world_size: int = None,
        pin_memory: bool = False,
        pin_memory_device: str = 'cuda'
        ) -> DataLoader:
    paths.sort()
    datasets = [load_from_disk(path) for path in paths]
    print(f'Datasets loaded from {paths}')

    dataset_sizes = [ds._info.splits.total_num_examples for ds in datasets]
    combined_size = sum(dataset_sizes)
    dataset_sampling_probs = [d_size / combined_size for d_size in dataset_sizes]

    if streaming:
        print('Convert the dataset into a streaming dataset')
        datasets = [dataset.to_iterable_dataset(num_shards=128) for dataset in datasets]

    if rank is not None:
        print('Split dataset for parallel training')
        datasets = [split_dataset_by_node(dataset, rank=rank, world_size=world_size) for dataset in datasets]

    print('Begin to interleave datasets')
    dataset = interleave_datasets(datasets, probabilities=dataset_sampling_probs, seed=seed, stopping_strategy='all_exhausted')
    print('max_len', max_len)
    dataset = dataset.map(
        partial(collate_text, min_len=min_len, max_len=max_len), 
        batch_size=1000, 
        drop_last_batch=True, 
        batched = True,
        remove_columns=[name for name in dataset.column_names if name != 'text'],
    )

    if n_skip > 0:
        dataset = dataset.skip(n_skip)

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=4,
        collate_fn=partial(render_batched_text, mask_ratio=mask_ratio, mask_type=mask_type),
        worker_init_fn=partial(dataloader_init_fn, seed=seed, render_config=render_config),
        drop_last=True
    )

    return loader
