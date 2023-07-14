import os
from typing import Any
from functools import partial
from contextlib import ExitStack
import json

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

from .train_config import ExpConfig

def dist_sync(config) -> None:
    dist.barrier() if config.num_gpu > 1 else ...

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
    print('fsdp wrapping')
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
        print('offload shards to cpu memory')

    return FSDP(
        module=model,
        sharding_strategy=ss,
        cpu_offload=CPUOffload(True if config.offload_to_cpu else False),
        auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls={
                LlamaDecoderLayer, GPT2Block, PIXELLayer, PIXELEmbeddings
            },),
        backward_prefetch=bp,
        mixed_precision=mps,
        device_id=dist.get_rank(),
        use_orig_params=True if config.torch_compile else False
    )

def ddp_wrap(model: nn.Module, config: ExpConfig) -> Any:
    print('ddp wrapping')
    model.to(dist.get_rank())
    return DDP(model, device_ids=[dist.get_rank()])

def wrap_model(model: nn.Module, config: ExpConfig) -> Any:
    if config.half_precision:
        print('half the model')
        model = model.half()

    if config.on_cpu:
        pass
    elif config.shard_strategy != 'no':
        model = fsdp_wrap(model, config) 
    else:
        model = ddp_wrap(model, config)
    
    if config.torch_compile:
        print('Compile the model')
        model = torch.compile(model=model, mode='max-autotune', dynamic=config.dynamic_shape)

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
