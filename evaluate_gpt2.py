#%%
from transformers import pipeline, set_seed
from datasets import load_dataset
from tqdm import tqdm
import re
import subprocess
from LatentPixel import confuse
import pandas as pd
import torch
from LatentPixel.utils import seed_everyting, gen_babi, process_lambda

seed_everyting(42)
torch.no_grad()

generator = pipeline('text-generation', model='gpt2', device='cuda')
results = pd.DataFrame(columns=['task', 'ratio', 'acc'])
confu_ratios = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

#%% evaluate on babi
num_sample_babi = 5000

data = gen_babi(1, num_sample_babi, '|', '')

for index, ratio in enumerate(confu_ratios):
    correct = 0
    for prompt, target in tqdm(data):
        try:
            prompt = confuse(prompt, ratio)
            gen = generator(prompt, max_length=5, return_full_text=False)
            gen = gen[0]['generated_text'].strip().split()[0]
            if gen.lower() == target.lower():
                correct += 1
        except:
            pass

    results.loc[index] = ['babi', ratio, correct / num_sample_babi]
    results.to_csv('gpt2_confuse_eval.csv')
    
#%% evaluate on lambada
data = load_dataset('lambada', split='test')

for ratio in confu_ratios:
    index += 1
    correct = 0
    num_sample = len(data)
    for sample in tqdm(data):
        txt = process_lambda(sample['text'])
        
        prompt = ' '.join(txt.split()[:-1])
        target = txt.split()[-1]
        
        prompt = confuse(prompt, ratio)
        
        try:
            gen = generator(prompt, max_length=5, return_full_text=False, do_sample=False, min_length=5)
        except:
            gen = ''

        try:
            gen = gen[0]['generated_text'].strip().split()[0]
            if gen.lower() == target.lower():
                correct += 1
        except IndexError:
            pass
        
    results.loc[index] = ['lambada', ratio, correct / num_sample]

    results.to_csv('gpt2_confuse_eval.csv')
