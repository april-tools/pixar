import os
from typing import Any
from functools import partial
from contextlib import ExitStack
import json
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
# scale gradient to avoid gradient underflow while doing mixed-precision training
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy
)
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy
)

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from pixel import PIXELLayer, PIXELEmbeddings

import wandb

from .train_config import ExpConfig
from ..modeling import LatentModel
from ..utils import timeit

_progress_bar: tqdm = None
_config: ExpConfig = None

def init_progress_bar(config: ExpConfig) -> tqdm:
    global _progress_bar, _config
    if _config == None:
        _config = config
    if config.rank == 0:
        _progress_bar = tqdm(
            total=config.total_steps + 1 - config.current_step,
            initial=config.current_step,
            dynamic_ncols=True
        )

def output(*args, **kwargs) -> None:
    global _config
    if _config.rank == 0:
        print(*args, **kwargs)

def init_dist_environ(config: ExpConfig):
    global _config
    if _config is None:
        _config = config
    if config.distributed:
        dist.init_process_group(backend='gloo' if config.on_cpu else 'nccl')
        keys = ['RANK', 'GROUP_RANK', 'LOCAL_RANK', 'LOCAL_WORLD_SIZE', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT']
        infos = [f'{key}:{os.environ[key]}' for key in keys]
        output(' '.join(infos))
    else:
        output('Stand-alone training.')

def dist_sync(config: ExpConfig) -> None:
    dist.barrier() if config.distributed else ...

def mix_precision_stack(stack: ExitStack, config: ExpConfig) -> list:
    device_type = 'cpu' if config.on_cpu else 'cuda'
    ctxs = []
    if config.mix_precision == 'fp16':
        ctxs.append(stack.enter_context(torch.autocast(device_type=device_type, dtype=torch.float16)))
    elif config.mix_precision == 'bf16':
        ctxs.append(stack.enter_context(torch.autocast(device_type=device_type, dtype=torch.bfloat16)))
    return ctxs

def eva_stack(stack: ExitStack, config: ExpConfig, model: Any) -> list:
    ctxs = [stack.enter_context(torch.no_grad())] + mix_precision_stack(stack, config)
    return ctxs

def train_gacc_stack(stack: ExitStack, config: ExpConfig, model: Any) -> list:
    ctxs = mix_precision_stack(stack, config)
    if isinstance(model, DDP) or isinstance(model, FSDP):
        ctxs += [stack.enter_context(model.no_sync())]
    return ctxs

def train_stack(stack: ExitStack, config: ExpConfig, model: Any) -> list:
    ctxs = mix_precision_stack(stack, config)
    return ctxs

def fsdp_wrap(model: nn.Module, config: ExpConfig) -> Any:
    output(f'Wrap the {type(model)} with FSDP')
    match config.shard_strategy:
        case 'full':
            ss = ShardingStrategy.FULL_SHARD
        case 'grad':
            ss = ShardingStrategy.SHARD_GRAD_OP
        case _:
            raise KeyError(f'Unsupported sharding strategy {config.shard_strategy}')
    
    match config.backward_prefetch:
        case 'pre':
            bp = BackwardPrefetch.BACKWARD_PRE
        case 'post':
            bp = BackwardPrefetch.BACKWARD_POST
        case _:
            raise KeyError(f'Unsupport backward prefetch strategy {config.backward_prefetch}')

    match config.mix_precision:
        case 'no':
            mps = None
        case 'fp16':
            mps = MixedPrecision(
                param_dtype=torch.float16,
                # Gradient communication precision.
                reduce_dtype=torch.float16,
                # Buffer precision.
                buffer_dtype=torch.float16,
            )
        case 'bf16':
            mps = MixedPrecision(
                param_dtype=torch.bfloat16,
                # Gradient communication precision.
                reduce_dtype=torch.bfloat16,
                # Buffer precision.
                buffer_dtype=torch.bfloat16,
            )

    if config.offload_to_cpu:
        output(f'Enable the CPU offloading of {type(model)}')

    return FSDP(
        module=model,
        sharding_strategy=ss,
        cpu_offload=CPUOffload(True if config.offload_to_cpu else False),
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={
                LlamaDecoderLayer, GPT2Block, PIXELLayer, PIXELEmbeddings
            },),
        backward_prefetch=bp,
        mixed_precision=mps,
        device_id=config.device_id,
        use_orig_params=True if config.torch_compile else False
    )

def ddp_wrap(model: nn.Module, config: ExpConfig) -> Any:
    output(f'Wrap the {type(model)} with DDP')
    model.to(config.device_id)
    return DDP(model, device_ids=[config.device_id])

@timeit
def wrap_model(model: LatentModel, config: ExpConfig) -> LatentModel:
    if config.half_precision:
        output(f'Set {type(model)} to pure fp16')
        model = model.half()

    if config.on_cpu:
        pass
    elif config.shard_strategy != 'no':
        model = fsdp_wrap(model, config) 
    else:
        model = ddp_wrap(model, config)
    
    if config.torch_compile:
        output(f'Compile the f{type(model)}')
        model.compile()

    return model

# average through distributed processors
@torch.no_grad()
def distributed_average(numbers: list | torch.Tensor | float, device: torch.device) -> list | torch.Tensor | float:
    if isinstance(numbers, torch.Tensor):
        return_type = 'tensor'
    elif isinstance(numbers, list):
        return_type = 'list'
    else:
        return_type = 'number'

    if not isinstance(numbers, torch.Tensor):
        numbers = torch.tensor(numbers, device=device)

    dist.reduce(numbers, dst=0, op=dist.ReduceOp.SUM)
    numbers /= dist.get_world_size()
    
    if return_type == 'list':
        numbers = numbers.tolist()
    elif return_type == 'number':
        numbers = numbers.item()
    
    return numbers

def backward(loss: torch.Tensor, optim_parts: dict) -> None:
    scaler = optim_parts['scaler']
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

def step(config: ExpConfig, optim_parts: dict, model: nn.Module | FSDP) -> None:
    optim = optim_parts['optim']
    scaler = optim_parts['scaler']
    scheduler = optim_parts['scheduler']

    if scaler:
        scaler.unscale_(optim)

    # gradient clip
    if isinstance(model, FSDP):
        model.clip_grad_norm_(config.clip)
    else:
        clip_grad_norm_(model.parameters(), config.clip)
    
    # update parameters
    if scaler:
        scaler.step(optim)
        scaler.update()
    else:
        optim.step()
    scheduler.step()
    optim.zero_grad()

    global _progress_bar
    if _progress_bar:
        try:
            _progress_bar.update()
        except Exception:
            output(f'Progress bar finished')

def log(metric: dict) -> None:
    global _config
    if _config.rank == 0:
        wandb.log(metric, step=_config.current_step)