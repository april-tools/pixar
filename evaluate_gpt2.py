from transformers import pipeline, set_seed
from datasets import load_dataset
from tqdm import tqdm
import re
import subprocess
from LatentPixel import confuse
import pandas as pd
import torch

torch.no_grad()

num_sample = 10000

generator = pipeline('text-generation', model='gpt2', device='cuda')
results = pd.DataFrame(columns=['ratio', 'acc'])


def gen_babi(task_id: int, num: int, prompt: str, after_prompt: str) -> tuple[str, str]:
    pat = re.compile("(^[0-9]*) ([^\t]*)(?:\t(.*)\t)*")

    raw = subprocess.run(['babi-tasks', f'{task_id}', f'{num}'], stdout=subprocess.PIPE).stdout.decode()
    lines = list(map(lambda x: x.groups(), map(pat.match, raw.splitlines())))
    # return lines
    results = []
    cur_sample = []
    for line in lines:
        if line[0] == '1':
            results.append(cur_sample) if len(cur_sample) > 0 else ...
            cur_sample = []
        cur_sample.append(line[1])
        if line[2] is not None:
            cur_sample.append(prompt)
            cur_sample.append(line[2] + after_prompt)
    results.append(cur_sample)
    data = []
    for sample in results:
        prompt = ' '.join(sample[:-1])
        target = sample[-1]
        data.append((prompt, target))
    return data

data = gen_babi(1, num_sample, '|', '')

# correct = 0
# for prompt, target in tqdm(data):
#     gen = generator(prompt, max_length=2, return_full_text=False)
#     gen = gen[0]['generated_text'].strip().split()[0]
#     if gen.lower() == target.lower():
#         correct += 1

# print(f'ACC of GPT2 model on {num_sample} samples:{correct / num_sample}')

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

data = load_dataset('lambada', split='test')

index = 5
for ratio in [0.4, 0.5]:
    correct = 0
    num_sample = len(data)
    idx = 0
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
        idx += 1
        
    print(f'ACC of GPT2 model on LAMBADA test set with {ratio} confusing:{correct / num_sample}')
    results.loc[index] = [ratio, correct / num_sample]

    results.to_csv('gpt2_conf_result.csv')
    index += 1
