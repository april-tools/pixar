from LatentPixel.utils import timeit
from datasets import load_dataset, load_from_disk
from LatentPixel import LatentLlama
from LatentPixel.modeling.llama import LlamaForPatchCausalInference
from LatentPixel import TGraph, DEFAULT_BINARY_RENDERING
from tqdm import tqdm
import numpy as np
import os

@timeit
def main(model, data):
    model.eval()
    model.cuda()
    for txt in tqdm(data['text'][:2000]):
        words = txt.split()
        prompt = process_lambda(' '.join(words[:-1]))
        prompt = TGraph.from_text([prompt])
        prompt = prompt._spacing_text(3)
        prompt.set_device('cuda')
        target = words[-1]
        gen = model.autoregressive_generate(prompt, None, 4)


def process_lambda(raw: str) -> str:
    raw = raw.replace('`` ', '"').replace(" ''", '"').replace('``', '"').replace("''", '"')
    result = ''
    for idx in range(len(raw) - 1):
        c = raw[idx]
        cn = raw[idx + 1]
        if cn in ',.?><{}()\'' and c == ' ':
            continue
        if (raw[idx+1:idx+4] == "n't" or raw[idx+1:idx+4] == "'re" or raw[idx+1:idx+3] == "'d" or raw[idx+1:idx+3] == "'s" or raw[idx+1:idx+3] == "'m") and raw[idx] == ' ':
            continue
        if raw[idx+1] == "'" and raw[idx+2] in 'abcdefghijklmnopqrstuvwxyz' and raw[idx] == ' ':
            continue
        result += raw[idx]
    result += raw[-1]
    return result

if __name__ == "__main__":
    #os.environ["HF_HOME"] =  "/exports/eddie/scratch/s2302935/"
    data = load_from_disk("storage/lambada")
    
    TGraph.init_render(**DEFAULT_BINARY_RENDERING)
    # Flash Attention
    print("Flash Attention Speed")
    model = LatentLlama(
    backbone_path='storage/llama/',
    num_channels=1,
    patch_len=2,
    patch_size=8,
    binary=True,)
    main(model, data)
    
    # No Flash
    setattr(model.backbone.config, 'flash', False)
    config = model.backbone.config
    model.backbone = LlamaForPatchCausalInference.from_pretrained('storage/llama/', config=config, ignore_mismatched_sizes=True)
    main(model, data)
