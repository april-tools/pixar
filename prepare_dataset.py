import torch
from LatentPixel.dataprocess.preprocess import preprocess_pretrain_data
from LatentPixel.config import PretrainDatasetConfig
from LatentPixel import TGraph, DEFAULT_BINARY_RENDERING
from datasets import load_dataset, Dataset
import pandas as pd
import pyarrow as pa

# books = load_dataset('lucadiliello/bookcorpusopen', split='train')
# books.save_to_disk('storage/bookcorpusopen', num_shards=256)    # shard the dataset for parallism

# wiki = load_dataset('wikipedia', '20220301.en', split='train')
# wiki.save_to_disk('storage/enwiki/', num_shards=256)

# del books
# del wiki

TGraph.init_render(**DEFAULT_BINARY_RENDERING)


dataset = preprocess_pretrain_data(
    PretrainDatasetConfig(
        dataset_paths=['storage/bookcorpusopen/', 'storage/enwiki/'],
        max_len=1180,
        min_len=100,
        seed=42,
        shuffle=True,
        num_shards=256
    )
)

# dataset = Dataset(pa.Table.from_pandas(pd.DataFrame([{'text': 'this is a sentence'},{'text': 'this is another sentence'}] * 10000)))


def add_image(sample: dict) -> dict:
    img = TGraph.from_text(sample['text'])
    sample['image'] = img._value.to(torch.uint8)
    sample['num_text_patches'] = img.num_text_patches
    return sample

dataset_with_im = dataset.map(add_image, num_proc=4)


dataset_with_im.save_to_disk('storage/booksAndWiki2', num_shards=256)
