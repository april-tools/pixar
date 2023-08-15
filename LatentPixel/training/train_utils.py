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
from torch.optim import Optimizer, AdamW, SGD
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
from transformers.optimization import get_cosine_schedule_with_warmup

from pixel import PIXELLayer, PIXELEmbeddings

import wandb

from .train_config import ExpConfig
from ..modeling import (
    LatentModel, 
    LPixelForMLM, 
    LatentGPT2, 
    LPixelForClassification,
    Discriminator,
    DiscriminatorConfig
)
from ..utils import timeit, init_render
from ..text_graph import TGraph

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
        infos = [f'{key}:{os.environ[key]}' for key in keys] + [f'DEVICE_ID:{config.device_id}']
        torch.cuda.set_device(config.device_id)
        print(' '.join(infos))
    else:
        print('Stand-alone training.')

def dist_sync(config: ExpConfig) -> None:
    dist.barrier() if config.distributed else print('no sync')

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
    scaler: ShardedGradScaler = optim_parts['scaler']
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

def step(config: ExpConfig, optim_parts: dict, model: nn.Module | FSDP, update_progress_bar: bool = True) -> None:
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
    if _progress_bar and update_progress_bar:
        try:
            _progress_bar.update()
        except Exception:
            output(f'Progress bar finished')

def log(metric: dict) -> None:
    global _config
    if _config.rank == 0:
        wandb.log(metric, step=_config.current_step)
        
@timeit
def prepare_model_for_glue(config: ExpConfig) -> tuple[LatentModel, dict]:
    # initialize the model according to the config
    match config.model:
        case 'LPixelForClassification':
            model = LPixelForClassification(
                coder_path=config.coder_path,
                backbone_path=config.backbone_path,
                img_size=config.image_size,
                latent_size=config.latent_size,
                num_labels=config.num_labels
            )
            model.delete_unused_layers()
        case _:
            raise NotImplementedError(f'GLUE evaluation on {config.model} has not been implemented')
        
    if config.latent_norm:
        print('Enable the latent norm')
        model.latent_norm = True
    else:
        print('Disable the latent norm')
        model.latent_norm = False
        
    if config.gradient_checkpointing:
        output('Enable gradient checkpointing')
        model.backbone.gradient_checkpointing_enable()
        
    if config.half_coder and model.coder is not None:
        output('Half the coder')
        model.coder.half()
        if not config.on_cpu:
            model.coder.to(config.device_id)
            
    # init optimizer
    match config.optim.lower():
        case 'adamw':
            if config.stage == 1:
                output('Load connection parameters to the AdamW optimizer.')
            else:
                output('Load backbone parameters to the AdamW optimizer.')
            optim = AdamW(
                params=model.get_backbone_parameters(),
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.decay,
            )
        case 'sgd':
            if config.stage == 1:
                output('Load connection parameters to the SGD optimizer.')
            else:
                output('Load backbone parameters to the SGD optimizer.')
            optim = SGD(
                params=model.get_backbone_parameters(),
                lr=config.lr,
                momentum=config.momentum
            )
        case _:
            raise KeyError(f'Invalid optim type {config.optim}')
        
    model.backbone = wrap_model(model.backbone, config)
    
    # init scheduler
    if config.scheduler.lower() == 'cosineannealinglr':
        if config.warm_up_step <= 0:
            print(f'initialize the CosineAnnealingLR with no warm up')
            scheduler = CosineAnnealingLR(
                optimizer=optim,
                T_max=config.total_steps,
                eta_min=0.1 * config.lr # from llama paper
            ) 
        else:
            print(f'initialize the CosineAnnealingLR with {config.warm_up_step} warm up steps')
            scheduler = get_cosine_schedule_with_warmup(
                optimizer=optim,
                num_warmup_steps=config.warm_up_step,
                num_training_steps=config.total_steps,
                num_cycles=0.45
            )
    else:
        raise KeyError(f'Invalid scheduler type {config.scheduler}')
    
    # init the scheduler for mixedprecision training
    scaler = None
    if config.mix_precision == 'fp16':
        print('init the gradscaler for mixed-precision training')
        scaler = ShardedGradScaler()    # This works for both DDP and FSDP

    optim_parts = {
        'optim': optim,
        'scheduler': scheduler,
        'scaler': scaler
    }

    init_progress_bar(config)
    init_render(config.render_config)

    return model, optim_parts

@timeit
def prepare_model(config: ExpConfig) -> tuple[LatentModel, dict]:
    # Select the correct model to load according to the config
    match config.model:
        case 'LPixelForMLM':
            output(f'Latent image size: {config.latent_size}')
            output(f'Latent patch size: {config.latent_patch_size}')
            model = LPixelForMLM(
                coder_path=config.coder_path,
                backbone_path=config.backbone_path,
                img_size=config.image_size,
                latent_size=config.latent_size
            )
            if config.stage == 1:
                model.init_connection_layers()
                model.delete_unused_layers()
            elif config.stage == 2:
                model.delete_unused_layers()
        case 'LatentGPT2':
            output(f'Latent image size: {config.latent_size}')
            output(f'Latent patch size: {config.latent_patch_size}')
            output('init latent gpt2 model')
            model = LatentGPT2(
                coder_path=config.coder_path,
                backbone_path=config.backbone_path,
                img_size=config.image_size,
                latent_size=config.latent_size
            )
            if config.stage == 1:
                model.init_connection_layers()
                model.delete_unused_layers()
            elif config.stage == 2:
                model.delete_unused_layers()
        case _:
            raise NotImplementedError(f'Unrecognizable model type {config.model}')
        
    if config.latent_norm:
        print('Enable the latent norm')
        model.latent_norm = True
    else:
        print('Disable the latent norm')
        model.latent_norm = False
        
    model.set_grad_for_stage(config.stage)
    
    if config.gradient_checkpointing:
        output('Enable gradient checkpointing')
        model.backbone.gradient_checkpointing_enable()

    if config.half_coder and model.coder is not None:
        output('Half the coder')
        model.coder.half()
        if not config.on_cpu:
            model.coder.to(config.device_id)


    if config.stage == 1:
        output('Load connection parameters for the optimizer.')
        params = model.get_connection_params()
    elif config.stage == 2:
        output('Load backbone parameters for the optimizer.')
        params = model.get_backbone_parameters()
    else:
        raise KeyError(f'Unsupport pretraining stage {config.stage}')

    # init optimizer
    match config.optim.lower():
        case 'adamw':
            output('Init AdamW optimizer')
            optim = AdamW(
                params=params,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.decay,
            )
        case 'sgd':
            output('Init SGD optimizer')
            optim = SGD(
                params=params,
                lr=config.lr,
                momentum=config.momentum
            )
        case _:
            raise KeyError(f'Invalid optim type {config.optim}')

    model.backbone = wrap_model(model.backbone, config)

    # init scheduler
    if config.scheduler.lower() == 'cosineannealinglr':
        scheduler = CosineAnnealingLR(
            optimizer=optim,
            T_max=config.total_steps,
            eta_min=0.1 * config.lr # from llama paper
        )
    else:
        raise KeyError(f'Invalid scheduler type {config.scheduler}')
    
    # init the scheduler for mixedprecision training
    scaler = None
    if config.mix_precision == 'fp16':
        print('init the gradscaler for mixed-precision training')
        scaler = ShardedGradScaler()    # This works for both DDP and FSDP

    optim_parts = {
        'optim': optim,
        'scheduler': scheduler,
        'scaler': scaler
    }

    init_progress_bar(config)
    init_render(config.render_config)

    return model, optim_parts

def prepare_gan(config: ExpConfig) -> tuple[LatentModel, dict, Discriminator, dict]:
    # Select the correct model to load according to the config
    match config.model:
        case 'LPixelForMLM':
            output(f'Latent image size: {config.latent_size}')
            output(f'Latent patch size: {config.latent_patch_size}')
            model = LPixelForMLM(
                coder_path=config.coder_path,
                backbone_path=config.backbone_path,
                img_size=config.image_size,
                latent_size=config.latent_size
            )
            if config.stage == 1:
                model.init_connection_layers()
                model.delete_unused_layers()
            elif config.stage == 2:
                model.delete_unused_layers()
        case 'LatentGPT2':
            output(f'Latent image size: {config.latent_size}')
            output(f'Latent patch size: {config.latent_patch_size}')
            output('init latent gpt2 model')
            model = LatentGPT2(
                coder_path=config.coder_path,
                backbone_path=config.backbone_path,
                img_size=config.image_size,
                latent_size=config.latent_size
            )
            if config.stage == 1:
                model.init_connection_layers()
                model.delete_unused_layers()
            elif config.stage == 2:
                model.delete_unused_layers()
        case _:
            raise NotImplementedError(f'Unrecognizable model type {config.model}')
        
    # Load the discriminator from path
    disc_config = DiscriminatorConfig.load(config.discriminator_path)
    disc_config.num_channel = config.latent_size[0]
    disc_config.patch_width = config.latent_size[1]
    disc_config.patch_height = config.latent_size[1]
    disc = Discriminator.load(folder=config.discriminator_path, config=disc_config)
    
    if config.latent_norm:
        print('Enable the latent norm')
        model.latent_norm = True
    else:
        print('Disable the latent norm')
        model.latent_norm = False
        
    model.set_grad_for_stage(config.stage)
    
    if config.gradient_checkpointing:
        output('Enable gradient checkpointing')
        model.backbone.gradient_checkpointing_enable()

    if config.half_coder and model.coder is not None:
        output('Half the coder')
        model.coder.half()
        if not config.on_cpu:
            model.coder.to(config.device_id)


    if config.stage == 1:
        output('Load connection parameters for the optimizer.')
        params = model.get_connection_params()
    elif config.stage == 2:
        output('Load backbone parameters for the optimizer.')
        params = model.get_backbone_parameters()
    else:
        raise KeyError(f'Unsupport pretraining stage {config.stage}')

    # init optimizer
    match config.optim.lower():
        case 'adamw':
            output('Init AdamW optimizer')
            optim = AdamW(
                params=params,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.decay,
            )
            disc_optim = AdamW(
                params=disc.parameters(),
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.decay,
            )
        case 'sgd':
            output('Init SGD optimizer')
            optim = SGD(
                params=params,
                lr=config.lr,
                momentum=config.momentum
            )
            disc_optim = SGD(
                params=disc.parameters(),
                lr=config.lr,
                momentum=config.momentum
            )
        case _:
            raise KeyError(f'Invalid optim type {config.optim}')

    model.backbone = wrap_model(model.backbone, config)
    disc = wrap_model(disc, config)

    # init scheduler
    if config.scheduler.lower() == 'cosineannealinglr':
        scheduler = CosineAnnealingLR(
            optimizer=optim,
            T_max=config.total_steps,
            eta_min=0.1 * config.lr # from llama paper
        )
        disc_scheduler = CosineAnnealingLR(
            optimizer=disc_optim,
            T_max=config.total_steps,
            eta_min=0.1 * config.lr # from llama paper
        )
    else:
        raise KeyError(f'Invalid scheduler type {config.scheduler}')
    
    # init the scheduler for mixedprecision training
    scaler = None
    if config.mix_precision == 'fp16':
        print('init the gradscaler for mixed-precision training')
        scaler = ShardedGradScaler()    # This works for both DDP and FSDP
        disc_scaler = ShardedGradScaler()

    optim_parts = {
        'optim': optim,
        'scheduler': scheduler,
        'scaler': scaler
    }
    disc_optim_parts = {
        'optim': disc_optim,
        'scheduler': disc_scheduler,
        'scaler': disc_scaler
    }

    init_progress_bar(config)
    init_render(config.render_config)

    return model, optim_parts, disc, disc_optim_parts

def save_exp(model: LatentModel, config: ExpConfig, name: str, discriminator: Discriminator | None = None) -> None:
    output(f'Saving the {name} model at step {config.current_step}')
    if config.rank == 0:
        model.save_backbone(config.backbone_ckpt_path(name))
        model.save_coder(config.coder_ckpt_path(name))
        if discriminator is not None:
            output(f'Saving the {name} discriminator at step {config.current_step}')
            if isinstance(discriminator, DDP):
                discriminator.module.save(config.discriminator_ckpt_path(name))
            else:
                discriminator.save(config.discriminator_ckpt_path(name))
        try:
            config.save(name)
        except:
            output(f'failed to save the config')
            output(config)
        
class InfLoader:
    
    def __init__(self, dataloader: DataLoader, config: ExpConfig | None = None) -> None:
        self.loader = dataloader
        self.config = config
        self.it = iter(self.loader)
        
    def next(self) -> TGraph:
        try:
            return next(self.it)
        except Exception:
            self.it = iter(self.loader)
            if self.config is not None:
                self.config.epoch += 1
                output(f'Begin epoch {self.config.epoch}')
            return next(self.it)
