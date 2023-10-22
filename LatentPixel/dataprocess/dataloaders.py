import os
from functools import partial
from collections import defaultdict
from dataclasses import asdict
import random
from typing import Iterator, Any

import torch
from torch.utils.data import DataLoader, Dataset
from torch import distributed
from ..text_graph import TGraph
from ..config import ModelType, RenderConfig, PretrainDatasetConfig
from ..utils import seed_everyting, timeit
from ..training.train_config import GLUE_META, ExpConfig
from ..metrics import Metric
from .preprocess import preprocess_pretrain_data
from tqdm import tqdm
import numpy as np

from datasets import load_dataset, interleave_datasets, load_from_disk
from datasets.distributed import split_dataset_by_node


class SkipDataset(Dataset):
    def __init__(self, dataset: Dataset, n_skip: int = 0) -> None:
        super().__init__()
        self.dataset = dataset
        self.n_skip = n_skip
    
    def __len__(self) -> int:
        return max(0, len(self.dataset) - self.n_skip)
    
    def __getitem__(self, index) -> Any:
        return self.dataset[index + self.n_skip]

def dataloader_init_fn(worker_id, seed: int, render_config: RenderConfig) -> None:
    seed_everyting(seed)
    os.system("taskset -p 0xffffffffff %d" % os.getpid())
    print(f'initialize the render with parameters {render_config.to_dict()}')
    render_params = render_config.to_dict()
    TGraph.init_render(**render_params)
    print(f'patch_len initialized as {TGraph._patch_len}')

def render_batched_text(batch: list[dict[str, str]], mask_ratio: float, mask_type: str) -> torch.Tensor:
    sents = []
    for sent in batch:
        sents.append(sent['text'])
    img = TGraph.from_text(sents)
    img.init_patch_mask(mask_type, ratio=mask_ratio)
    img.patch_mask
    img.attention_mask

    return img
        
def collate_text(batch: dict, max_len: int | None = None) -> list[str]:
    texts: list[str] = batch['text']
    docs = []
    for txt in texts:
        docs.append(txt.split('\n'))
    
    samples = []
    for doc in docs:
        sents = []
        length = 0
        idx = 0
        while idx < len(doc):
            sent = doc[idx]
            l = len(sent)
            if l + length < max_len:
                length += l
                sents.append(sent.strip())
                idx += 1
            elif l >= max_len:
                samples.append(' '.join(sents))
                samples.append(sent)
                idx += 1
                length = 0
                sents = []
            else:
                samples.append(' '.join(sents))
                length = 0
                sents = []

    return {'text': samples}

def collate_glue(batch: dict) -> TGraph:
    merged = defaultdict(list)
    for sample in batch:
        for k, v in sample.items():
            merged[k].append(v)
    batch = merged
    label = batch['label']
    if len(batch) == 4:
        keys = list(batch.keys())
        text = list(zip(batch[keys[0]], batch[keys[1]]))
    elif len(batch) == 3:
        key = list(batch.keys())[0]
        text = batch[key]
    else:
        raise ValueError(f'GLUE dataset should be 3 or 4 fields, but got {len(batch)} fields')
    
    tgraph = TGraph.from_text(text)
    tgraph.labels = torch.tensor(label)
    assert isinstance(tgraph.labels, torch.LongTensor) or isinstance(tgraph.labels, torch.DoubleTensor) or isinstance(tgraph.labels, torch.FloatTensor)
    tgraph.attention_mask   # init the attention mask
    
    return tgraph

def get_pretrain_dataloader(
    data_conf: PretrainDatasetConfig,
    render_conf: RenderConfig,
    cache_path: str | os.PathLike,
    batch_size: int,
    n_skip: int,
    num_workers: int,
    rank: int,
    world_size: int,
    mask_ratio: float = 0.25,
    mask_type: str = 'rand'
) -> DataLoader:
    # Check whether this dataset is cached
    cache_folder = os.path.join(cache_path, data_conf.signature())
    if os.path.isdir(cache_folder):
        print(f'Find cached path at {cache_folder}')
        dataset = load_from_disk(os.path.join(cache_folder, 'data'))
    else:
        print(f'No cached data found, begin to preprocess on rank {0}')
        if rank == 0:
            dataset = preprocess_pretrain_data(data_conf)
            dataset.save_to_disk(os.path.join(cache_folder, 'data'), num_shards=data_conf.num_shards)
            data_conf.save(cache_folder)
            print(f'Preprocessed dataset saved to {cache_folder}')

        if world_size > 1:
            distributed.barrier()
        dataset = load_from_disk(os.path.join(cache_folder, 'data'))
        print(f'Dataset loaded on rank {rank}')
        
    dataset = split_dataset_by_node(
        dataset=dataset, 
        rank=rank,
        world_size=world_size
    )

    num_samples = len(dataset)

    if n_skip > 0:
        print(f'Skip first {n_skip} samples to continue training')
        dataset = SkipDataset(dataset, (n_skip // world_size) % num_samples)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=4,
        collate_fn=partial(render_batched_text, mask_ratio=mask_ratio, mask_type=mask_type),
        worker_init_fn=partial(dataloader_init_fn, seed=data_conf.seed, render_config=render_conf),
        drop_last=True
    )
    TGraph.init_render(**asdict(render_conf))
    return loader

def get_pixel_pretrain_dataloader(
        paths: list[str | os.PathLike],
        batch_size: int, 
        num_workers: int, 
        seed: int,
        mask_ratio: float,
        mask_type: str,
        render_config: RenderConfig = None,
        n_skip: int = 0,
        max_len: int = 800,
        streaming: bool = True,
        rank: int = None, 
        world_size: int = None,
        pin_memory: bool = False,
        pin_memory_device: str = 'cuda'
        ) -> DataLoader:
    paths.sort()
    if len(paths) > 1: # for interlevaing datasets, many datasets into one
        datasets = [load_from_disk(path) for path in paths]
        print(f'Datasets loaded from {paths}')

        dataset_sizes = [ds._info.splits.total_num_examples for ds in datasets]
        combined_size = sum(dataset_sizes)
        dataset_sampling_probs = [d_size / combined_size for d_size in dataset_sizes]
        
        if rank is not None:
            print(f'Split dataset for parallel training for rank {rank}/{world_size}')
            datasets = [split_dataset_by_node(dataset, rank=rank, world_size=world_size) for dataset in datasets]

        if streaming:
            print('Convert the dataset into a streaming dataset')
            datasets = [dataset.to_iterable_dataset(num_shards=128) for dataset in datasets]

        print('Begin to interleave datasets')
        dataset = interleave_datasets(datasets, probabilities=dataset_sampling_probs, seed=seed, stopping_strategy='all_exhausted')
    else:
        path = paths[0]
        dataset = load_from_disk(path)        
        if streaming:
            print('Convert the dataset into a streaming dataset')
            dataset = dataset.to_iterable_dataset(num_shards=128)
        if rank is not None:
            print('Split dataset for parallel training')
            dataset = split_dataset_by_node(dataset, rank=rank, world_size=world_size)
            
    print('max_len', max_len)
    dataset = dataset.map(
        partial(collate_text, max_len=max_len), 
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
    dataloader_init_fn(0, seed, render_config)
    return loader


def get_glue_dataset(
        task: str,
        config: ExpConfig | None = None,
        sub_size: int | None = None,
        mp_workers: int | None = None,
        seed: int | None = None,
        render_config: RenderConfig | None = None,
        rank: int = 0,
        world_size: int = 1
    ) -> tuple[DataLoader, list[DataLoader], list[Metric], int]: # return the train dataloader, validation dataloaders, the metrics, and the number of labels
    if config is not None:
        sub_size = config.sub_size
        mp_workers = config.mp_workers
        seed = config.seed
        render_config = config.render_config
        rank = config.rank
        world_size = config.world_size
    
    metrics = dict(GLUE_META)[task][0]
    num_labels = dict(GLUE_META)[task][1]
    
    print(f'Begin to load data for <{task}> task at rank {rank}')
    train_data = load_dataset('glue', task, split='train')
    print(f'Train dataset for {task} loaded')
    if task == 'mnli':
        val_datas = [load_dataset('glue', 'mnli', split='validation_matched'), load_dataset('glue', 'mnli', split='validation_mismatched')]
    else:
        val_datas = [load_dataset('glue', task, split='validation')]
        
    print(f'Validation dataset for {task} loaded')
    print(f'{len(val_datas)} validation sets in total')

    train_data = split_dataset_by_node(train_data, rank=rank, world_size=world_size)
    print(f'Train dataset splitted')

    val_datas = [split_dataset_by_node(data, rank=rank, world_size=world_size) for data in val_datas]
    print(f'Validation dataset splitted')
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=sub_size,
        shuffle=True,
        num_workers=mp_workers,
        prefetch_factor=4,
        collate_fn=collate_glue,
        worker_init_fn=partial(dataloader_init_fn, seed=seed, render_config=render_config),
        drop_last=False
    )
    print(f'Train dataloader prepared')

    
    val_loaders = [
        DataLoader(
            dataset=data,
            batch_size=sub_size,
            shuffle=True,
            num_workers=mp_workers,
            prefetch_factor=4,
            collate_fn=collate_glue,
            worker_init_fn=partial(dataloader_init_fn, seed=seed, render_config=render_config),
            drop_last=False
        ) for data in val_datas
    ]
    print(f'Validation dataloaders prepared')
    print(f'{len(metrics)} metrics will be used')
    print([met().metric_name() for met in metrics])

    
    return train_loader, val_loaders, metrics, num_labels
