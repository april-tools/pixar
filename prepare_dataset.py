from LatentPixel.dataprocess.preprocess import preprocess_pretrain_data
from LatentPixel.config import PretrainDatasetConfig
from datasets import load_dataset

books = load_dataset('lucadiliello/bookcorpusopen', split='train')
books.save_to_disk('storage/bookcorpusopen', num_shards=256)    # shard the dataset for parallism

wiki = load_dataset('wikipedia', '20220301.en', split='train')
wiki.save_to_disk('storage/enwiki/', num_shards=256)

del books
del wiki

dataset = preprocess_pretrain_data(
    PretrainDatasetConfig(
        dataset_paths=['storage/bookcorpusopen/', 'storage/enwiki/'],
        max_len=3500,
        seed=42,
        shuffle=True,
        num_shards=256
    )
)

dataset.save_to_disk('storage/booksAndWiki/data', num_shards=256)
