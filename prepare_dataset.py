from os.path import join
from datasets import load_dataset, load_from_disk
from LatentPixel import params2dict

params = {
    'hf_paths': ["lucadiliello/bookcorpusopen"],
    'num_shards': 256,
    'ds_names': ['fbookcorpusopen'],
    'split_names': ['train'],
    'local_path': '/work/sc118/sc118/shared'
}

def process(hf_path: str, path: str, split_name: str, num_shards: int) -> None:
    ds = load_dataset(hf_path)
    ds = ds[split_name]
    ds.save_to_disk(path, num_shards=num_shards)

if __name__ == '__main__':
    params = params2dict(params)
    for hf_path, ds_name, split_name in zip(params['hf_paths'], params['ds_names'], params['split_names']):
        process(hf_path, join(params['local_path'], ds_name), split_name, params['num_shards'])
