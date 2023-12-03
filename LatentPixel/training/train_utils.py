import os
from typing import Any, Dict, Optional
from functools import partial
from contextlib import ExitStack
import json
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer, AdamW, SGD
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR, LinearLR, SequentialLR
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

from adamr import AdamR

import wandb

from .train_config import ExpConfig
from ..modeling import (
    LatentModel, 
    LPixelForMLM, 
    LatentGPT2, 
    LPixelForClassification,
    Discriminator,
    DiscriminatorConfig,
    CNNAutoencoder,
    Compressor,
    LatentLlama,
    LatentLlamaForSequenceClassification
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
            total=config.total_steps,
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

    dist.all_reduce(numbers, op=dist.ReduceOp.SUM)
    numbers /= dist.get_world_size()
    
    if return_type == 'list':
        numbers = numbers.tolist()
    elif return_type == 'number':
        numbers = numbers.item()
    
    return numbers

def backward(loss: torch.Tensor, optim_parts: dict, config: ExpConfig | None = None, retain_graph: bool=None) -> None:
    scaler: ShardedGradScaler = optim_parts['scaler']
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward(retain_graph=retain_graph)

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
        case 'LatentLlamaForSequenceClassification':
            output('init LatentLlamaForSequenceClassification model')
            model = LatentLlamaForSequenceClassification(
                compressor_path=config.compressor_path,
                backbone_path=config.backbone_path,
                compressor_name=config.compressor_name,
                num_channels=config.num_channel,
                num_labels=config.num_labels,
                patch_size=config.pixels_per_patch,
                patch_len=config.patch_len,
                binary=config.binary
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
        
    if config.half_coder and model.compressor is not None:
        output('Half the coder')
        model.compressor.half()
        if not config.on_cpu:
            model.compressor.to(config.device_id)

    params = model.get_backbone_parameters()
            
    # init optimizer
    optim = get_optimizer(params, config)
        
    model.backbone = wrap_model(model.backbone, config)
    
    # init scheduler
    scheduler = get_LrScheduler(optim, config)

    optim_path = config.load_optim_path()
    if optim_path:
        optim.load_state_dict(torch.load(optim_path, map_location=f'cuda:{config.device_id}'))
        print(f'Load the optimizer states from {optim_path}')

    scheduler_path = config.load_scheduler_path()
    if scheduler_path:
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=f'cuda:{config.device_id}'))
    
    # init the scaler for mixedprecision training
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
def prepare_model(config: ExpConfig) -> tuple[LatentModel | Compressor, dict]:
    # Select the correct model to load according to the config
    match config.model:
        case 'LPixelForMLM':
            output(f'Latent image size: {config.latent_size}')
            output(f'Latent patch size: {config.latent_patch_size}')
            model: LPixelForMLM = LPixelForMLM(
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
            output('init latent gpt2 model')
            model: LatentGPT2 = LatentGPT2(
                compressor_path=config.compressor_path,
                backbone_path=config.backbone_path,
                compressor_name=config.compressor_name,
                num_channels=config.num_channel,
                patch_size=config.pixels_per_patch,
                patch_len=config.patch_len,
                binary=config.binary
            )
            model.delete_unused_layers()
        case 'LatentLlama':
            output('init LatentLlama model')
            model: LatentLlama = LatentLlama(
                compressor_path=config.compressor_path,
                backbone_path=config.backbone_path,
                compressor_name=config.compressor_name,
                num_channels=config.num_channel,
                patch_size=config.pixels_per_patch,
                patch_len=config.patch_len,
                binary=config.binary
            )
            model.delete_unused_layers()
        case 'CNNAutoencoder':
            output('Initializing the CNNAutoencoder')
            output(f'Pathch length:{config.patch_len}')
            model = CNNAutoencoder(path=config.compressor_path)
            output(str(model.config))
        case _:
            raise NotImplementedError(f'Unrecognizable model type {config.model}')
    
    if not isinstance(model, Compressor):
        model: LatentModel
        if config.latent_norm:
            print('Enable the latent norm')
            model.latent_norm = True
        else:
            print('Disable the latent norm')
            model.latent_norm = False
            
        if config.gradient_checkpointing:
            output('Enable gradient checkpointing')
            model.backbone.gradient_checkpointing_enable()

        if config.half_coder and model.compressor is not None:
            output('Half the compressor')
            model.compressor.half()
        if not config.on_cpu and model.compressor is not None:
            model.compressor.to(config.device_id)

        params = model.get_backbone_parameters()
    else:
        model: Compressor
        params = model.parameters()
        
    # Load the discriminator from path
    if config.discriminator_path is not None and len(config.discriminator_path) > 0:
        disc_config = DiscriminatorConfig.load(config.discriminator_path)
        disc_config.num_channel = config.num_channel
        disc_config.patch_len = config.patch_len
        disc_config.pixel_per_patch = config.pixels_per_patch
        disc = Discriminator.load(folder=config.discriminator_path, config=disc_config)
    else:
        disc = None

    # init optimizer
    optim = get_optimizer(params, config)
    disc_optim = get_optimizer(disc.parameters(), config) if disc is not None else None
        
    if not isinstance(model, Compressor):
        model.backbone = wrap_model(model.backbone, config)
    else:
        model = wrap_model(model, config)
    disc = wrap_model(disc, config) if disc is not None else None

    # init scheduler
    scheduler = get_LrScheduler(optim, config)
    disc_scheduler = get_LrScheduler(disc_optim, config, True) if disc is not None else None

    optim_path = config.load_optim_path()
    if optim_path:
        optim.load_state_dict(torch.load(optim_path, map_location=f'cuda:{config.device_id}'))
        print(f'Load the optimizer states from {optim_path}')

    scheduler_path = config.load_scheduler_path()
    if scheduler_path:
        scheduler.load_state_dict(torch.load(scheduler_path, map_location=f'cuda:{config.device_id}'))
    
    # init the scaler for mixedprecision training
    scaler = None
    disc_scaler = None
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

    # init optimizers
    optim = get_optimizer(params, config)
    disc_optim = get_optimizer(disc.parameters(), config)

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

def save_exp(model: LatentModel | Compressor, config: ExpConfig, name: str, discriminator: Discriminator | None = None, optim_parts: dict | None = None) -> None:
    if config.rank != 0:
        return
    output(f'Saving the {name} model at step {config.current_step}')
    if not isinstance(model, LatentModel):
        model.module.save(config.coder_ckpt_path(name))
        try:
            config.save(name)
        except:
            output(f'failed to save the config')
            output(config)
        return
    
    model.save_backbone(config.backbone_ckpt_path(name))
    model.save_compressor(config.coder_ckpt_path(name))
    if discriminator is not None:
        output(f'Saving the {name} discriminator at step {config.current_step}')
        if isinstance(discriminator, DDP):
            discriminator.module.save(config.discriminator_ckpt_path(name))
        else:
            discriminator.save(config.discriminator_ckpt_path(name))

    config.save(name)

    # optim_parts = {
    #     'optim': optim,
    #     'scheduler': scheduler,
    #     'scaler': scaler
    # }
    if optim_parts:
        optim: Optimizer = optim_parts['optim']
        scheduler: LRScheduler = optim_parts['scheduler']
        torch.save(optim.state_dict(), config.optim_path(name))
        torch.save(scheduler.state_dict(), config.scheduler_path(name))

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


def is_one_more_batch(dataloader: DataLoader, config: ExpConfig) -> bool:
    '''
    Check whether other distributed process's dataloader has 1 more batch
    '''
    if config.num_gpu < 2:
        return False
    my_num = torch.tensor([len(dataloader)], dtype=torch.long, device=config.device_id)
    num_list = [torch.clone(my_num) for _ in range(config.world_size)]
    dist.all_gather(num_list, my_num)
    
    max_n = torch.cat(num_list).max().item()
    my_num = my_num.item()
    if max_n > my_num:
        print(f'RANK {config.rank} has {my_num} batches, wait others with {max_n} batches.')
        return True
    return False

def get_CosineAnnealingWithLrWarmUpLR(optimizer: Optimizer, warmup_steps:int, min_lr: float, total_steps: int, verbose: bool = ...) -> SequentialLR:
    lr1 = LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps, verbose=verbose)
    lr2 = CosineAnnealingLR(optimizer, T_max=total_steps-warmup_steps , eta_min=min_lr, verbose=verbose)
    return SequentialLR(optimizer, [lr1, lr2], milestones=[warmup_steps], verbose=verbose)

def get_ConstantWithLrWarmUpLR(optimizer: Optimizer, warmup_steps: int, total_steps: int, verbose: bool = ...) -> SequentialLR:
    lr1 = LinearLR(optimizer, start_factor=1e-5, end_factor=1.0, total_iters=warmup_steps, verbose=verbose)
    lr2 = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=total_steps-warmup_steps, verbose=verbose)
    return SequentialLR(optimizer, [lr1, lr2], milestones=[warmup_steps], verbose=verbose)

def get_LrScheduler(optimizer: Optimizer, conf: ExpConfig, is_gan: bool=False) -> LRScheduler:
    verbose = True if conf.rank == 0 else False
    if is_gan:
        warm_up_step = conf.gan_lr_warm_up_steps
        lr = conf.gan_lr
        total_steps = conf.gan_total_steps
    else:
        warm_up_step = conf.warm_up_step
        lr = conf.lr
        total_steps = conf.total_steps
        
    match conf.scheduler:
        case 'CosineAnnealingLR':
            if warm_up_step > 0:
                scheduler = get_CosineAnnealingWithLrWarmUpLR(optimizer, warmup_steps=warm_up_step, min_lr=0.1*lr, total_steps=total_steps, verbose=verbose)
            else:
                scheduler = CosineAnnealingLR(optimizer, T_max=conf.total_steps, eta_min=0.1*conf.lr, verbose=verbose)
        case 'ConstantLR':
            if warm_up_step > 0:
                scheduler = get_ConstantWithLrWarmUpLR(optimizer, warm_up_step, total_steps, verbose=verbose)
            else:
                scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=1.0, total_iters=total_steps, verbose=verbose)
        case '':
            scheduler = None
        case _:
            raise NotImplementedError(f'{conf.scheduler} not implemented yet!')
    
    return scheduler

def get_optimizer(params: torch.ParameterDict, config: ExpConfig) -> Optimizer:
    '''
    Initialize the optimizer
    '''
    match config.optim.lower():
        case 'adamw':
            output('Init AdamW optimizer')
            optim = AdamW(
                params=params,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.decay,
                eps=1e-8
            )
        case 'sgd':
            output('Init SGD optimizer')
            optim = SGD(
                params=params,
                lr=config.lr,
                momentum=config.momentum
            )
        case 'adamr':
            output('Init AdamR optimizer')
            optim = AdamR(
                params=params,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_recovery=config.decay,
                eps=1e-8
            )
        case _:
            raise KeyError(f'Invalid optim type {config.optim}')
        
    return optim
