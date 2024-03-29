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
from nltk import sent_tokenize

from datasets import load_dataset, interleave_datasets, load_from_disk
from datasets.distributed import split_dataset_by_node


class FakeDataset(Dataset):
    
    def __init__(self, num: int=1024) -> None:
        super().__init__()
        self.num = num
        
    def __len__(self) -> int:
        return self.num
    
    def __getitem__(self, index) -> Any:
        return 'This is a sentence'


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
    os.system("taskset -p 0xfffffffffffffffffff %d > /dev/null" % os.getpid())
    render_params = render_config.to_dict()
    TGraph.init_render(**render_params)

def render_batched_text(batch: list[dict[str, str]], mask_ratio: float, mask_type: str) -> torch.Tensor:
    sents = []
    for sent in batch:
        sents.append(sent['text'])
    img = TGraph.from_text(sents)
    # img.init_patch_mask(mask_type, ratio=mask_ratio)
    # img.patch_mask
    img.attention_mask

    return img

def prepare_rendered_text(batch: list[dict[str, str]], mask_ratio: float, mask_type: str) -> torch.Tensor:
    imgs = []
    num_text_patches = []
    texts = []
    for sample in batch:
        imgs.append(sample['image'])
        num_text_patches.append(sample['num_text_patches'])
        texts.append(sample['text'])
    
    imgs = torch.tensor(imgs, dtype=torch.long)
    
    result = TGraph()
    result._value = imgs
    result.patch_len = TGraph._patch_len
    result._binary = TGraph._binary
    result.num_text_patches = num_text_patches
    result.text = texts

    return result


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

def collate_glue(batch: dict, detoken: bool) -> TGraph:
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
    
    if detoken:
        detokned = []
        for sample in text:
            if isinstance(sample, str):
                detokned.append(detokenize(sample))
            elif isinstance(sample, tuple):
                detokned.append((detokenize(sample[0]), detokenize(sample[1])))
            else:
                raise ValueError(f'GLUE sentence in wrong format!')
        text = detokned
    
    tgraph = TGraph.from_text(text)
    tgraph._labels = torch.tensor(label)
    assert isinstance(tgraph._labels, torch.LongTensor) or isinstance(tgraph._labels, torch.DoubleTensor) or isinstance(tgraph._labels, torch.FloatTensor)
    tgraph.attention_mask   # init the attention mask
    
    return tgraph

def detokenize(sent: str) -> str:
    sent = sent.replace("''", '"').replace("' '", '"')
    words = sent.split(' ')
    in_quota = False
    sent = ''
    idx = 0
    no_space = False
    while idx < len(words):
        w = words[idx]
        if w in (',', '.', "'", ';', '%', ')', '>', '?', "'m", "'s", '”', ':') or w[0] == "'":
            sent += w
        elif w == '"':
            if in_quota:
                in_quota = False
                sent += w
            else:
                in_quota = True
                no_space = True
                sent += ' ' + w
        elif w in ('$', '<', '“', '('):
            sent += ' ' + w
            no_space = True
        else:
            if no_space:
                sent += w
                no_space = False
            else:
                sent += ' ' + w
        idx += 1  
    return sent.strip()

def get_fake_dataloader(
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
    
    dataset = FakeDataset()
    
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=32 if num_workers > 0 else None,
        collate_fn=partial(render_batched_text, mask_ratio=mask_ratio, mask_type=mask_type),
        worker_init_fn=partial(dataloader_init_fn, seed=data_conf.seed, render_config=render_conf),
        drop_last=True
    )
    TGraph.init_render(**asdict(render_conf))
    return loader
    

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
    if data_conf.dataset_paths[0] == 'c4':
        print("Using C4 dataset!")
        return get_c4_dataloader(
            data_conf=data_conf,
            render_conf=render_conf,
            cache_path=cache_path,
            batch_size=batch_size,
            n_skip=n_skip,
            num_workers=num_workers,
            rank=rank,
            world_size=world_size,
            mask_ratio=mask_ratio,
            mask_type=mask_type
        )
        
    if len(data_conf.dataset_paths) > 1:
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
    else:
        dataset = load_from_disk(os.path.join(data_conf.dataset_paths[0], 'data'))
        print(f'Dataset {data_conf.dataset_paths[0]} loaded on rank {rank}')
        
    dataset = split_dataset_by_node(
        dataset=dataset, 
        rank=rank,
        world_size=world_size
    )

    num_samples = len(dataset)

    if data_conf.prerendered:
        collate_fn = prepare_rendered_text
    else:
        collate_fn = render_batched_text

    if n_skip > 0:
        print(f'Skip first {n_skip} samples to continue training')
        dataset = SkipDataset(dataset, (n_skip // world_size) % num_samples)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=32 if num_workers > 0 else None,
        collate_fn=partial(collate_fn, mask_ratio=mask_ratio, mask_type=mask_type),
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
        prefetch_factor=32 if num_workers > 0 else None,
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
    detoken = False
    
    train_data = load_dataset('glue', task, split='train')
    if task == 'mnli':
        val_datas = [load_dataset('glue', 'mnli', split='validation_matched'), load_dataset('glue', 'mnli', split='validation_mismatched')]
    else:
        val_datas = [load_dataset('glue', task, split='validation')]
        
    if task in ('mrpc',):
        detoken = True
        

    train_data = split_dataset_by_node(train_data, rank=rank, world_size=world_size)

    val_datas = [split_dataset_by_node(data, rank=rank, world_size=world_size) for data in val_datas]
    
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=sub_size,
        shuffle=True,
        num_workers=mp_workers,
        prefetch_factor=32,
        collate_fn=partial(collate_glue, detoken=detoken),
        worker_init_fn=partial(dataloader_init_fn, seed=seed, render_config=render_config),
        drop_last=False
    )

    
    val_loaders = [
        DataLoader(
            dataset=data,
            batch_size=sub_size,
            shuffle=True,
            num_workers=mp_workers,
            prefetch_factor=32,
            collate_fn=partial(collate_glue, detoken=detoken),
            worker_init_fn=partial(dataloader_init_fn, seed=seed, render_config=render_config),
            drop_last=False
        ) for data in val_datas
    ]

    
    return train_loader, val_loaders, metrics, num_labels

def render_batched_c4(batch: list[dict[str, str]]) -> torch.Tensor:
    sents = []
    for sent in batch:
        sents.append(sent['text'])
    img = TGraph.from_text(sents)
    img.attention_mask

    return img

def _sent_split(sent: str, max_len: int, min_len: int) -> list[str]:
    sents = sent_tokenize(sent)
    
    samples = []
    
    sample = []
    num = 0
    for sent in sents:
        l = len(sent)
        nnum = num + l
        if nnum >= max_len:
            if num >= min_len:
                samples.append(' '.join(sample))
            sample = [sent]
            num = l
        else:
            sample.append(sent)
            num = nnum
            
    if len(sample) > 0:
        sent = ' '.join(sample)
        if len(sent) > min_len:
            samples.append(' '.join(sample))
        
    return samples

def sent_split(batch: list[str], max_len: int, min_len: int) -> list[str]:
    batch = batch['text']
    samples = []
    for sent in batch:
        samples.extend(_sent_split(sent, max_len, min_len))
    return {'text': samples}

def get_c4_dataloader(
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
    TGraph.init_render(**asdict(render_conf))
    c4 = load_dataset('c4', 'en', split='train', streaming=True)
    if data_conf.shuffle:
        c4 = c4.shuffle(seed=data_conf.seed)
    c4 = split_dataset_by_node(c4, rank, world_size)
    c4 = c4.map(
        partial(sent_split, max_len=data_conf.max_len, min_len=data_conf.min_len),
        batch_size=2,
        drop_last_batch=False,
        batched=True,
        remove_columns=[name for name in c4.column_names if name != 'text'],
    )

    loader = DataLoader(
        c4, 
        batch_size=batch_size, 
        collate_fn=render_batched_c4, 
        drop_last=False, 
        pin_memory=True, 
        num_workers=num_workers, 
        prefetch_factor=32 if num_workers > 0 else None
    )
    
    return loader
