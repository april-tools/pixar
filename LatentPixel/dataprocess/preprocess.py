from functools import partial
from datasets import interleave_datasets, Dataset, load_from_disk
from ..config import PretrainDatasetConfig
from nltk import sent_tokenize

def collate_text(batch: dict, max_len: int | None = None) -> list[str]:
    texts: list[str] = batch['text']
    docs = []
    for txt in texts:
        docs.append(sent_tokenize(txt))
    
    samples = []
    for doc in docs:
        sents = []
        length = 0
        idx = 0
        while idx < len(doc):
            sent = doc[idx]
            l = len(sent)
            if l + length < max_len:
                length += l + 1 # including the space
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

def preprocess_pretrain_data(conf: PretrainDatasetConfig) -> Dataset:
    datasets = [load_from_disk(path) for path in conf.dataset_paths]
    print(f'Datasets loaded from {conf.dataset_paths}')


    dataset_sizes = [len(ds) for ds in datasets]
    combined_size = sum(dataset_sizes)
    print(f'Dataset sizes: {dataset_sizes}, {combined_size} in total.')

    dataset_sampling_probs = [d_size / combined_size for d_size in dataset_sizes]
    
    if len(datasets) > 1:
        print('begin to interleave datasets')
        dataset = interleave_datasets(
            datasets, 
            probabilities=dataset_sampling_probs,
            stopping_strategy='all_exhausted'
        )
    else:
        dataset = datasets[0]
        
    print(f'{len(dataset)} samples after interleave.')
    
    print(f'Split long sentences within max length {conf.max_len}')
    dataset = dataset.map(
        partial(collate_text, max_len=conf.max_len),
        batch_size=1000,
        drop_last_batch=False,
        batched=True,
        remove_columns=[name for name in dataset.column_names if name != 'text'],
        num_proc=8
    )
    
    if conf.shuffle:
        print(f'Shuffle the dataset with seed {conf.seed}')
        dataset = dataset.shuffle(seed=conf.seed)
        
    print(f'{len(dataset)} samples after splitting.')
    
    return dataset
