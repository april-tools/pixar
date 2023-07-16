import os
from contextlib import ExitStack

import torch
from torch.optim import(
    AdamW,
    SGD
)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

from .train_config import (
    ExpConfig,
    init_wandb,
    get_config
)
from ..text_graph import TGraph
from ..utils import init_render, timeit
from ..dataprocess import get_pixel_pretrain_dataloader
from ..modeling import (
    LatentModel,
    LPixelForPreTraining
)
from ..config import (
    ModelType
)
from .train_utils import (
    wrap_model,
    init_progress_bar,
    output,
    train_gacc_stack,
    backward,
    train_stack,
    step,
    distributed_average,
    log,
    dist_sync
)

import wandb

@timeit
def init_exp(config: ExpConfig) -> tuple[LatentModel, DataLoader, DataLoader, dict]:
    '''
    Initialize models, wraps, optimizers, and dataloaders
    '''
    match config.model:
        case 'LPixelForPreTraining':
            output(f'Latent image size: {config.latent_size}')
            output(f'Latent patch size: {config.latent_patch_size}')
            model = LPixelForPreTraining(
                coder_type=ModelType.SD,
                mask_ratio=config.mask_ratio,
                image_size=config.latent_size,
                patch_size=config.latent_patch_size,
                pixel_path=config.backbone_path,
                coder_path=config.coder_path,
                ckpt_path=config.init_path,
                keep_decoder=False
            )
        case _:
            raise NotImplementedError(f'Unrecognizable model type {config.model}')
    
    if config.gradient_checkpointing:
        print('Enable gradient checkpointing')
        model.gradient_checkpointing_enable()

    if config.half_coder:
        print('Half the coder')
        model.coder.half()

    model = wrap_model(model, config)

    # init optimizer
    match config.optim.lower():
        case 'adamw':
            optim = AdamW(
                params=model.parameters() if config.stage == 1 else None,
                lr=config.lr,
                betas=(config.beta1, config.beta2),
                weight_decay=config.decay,
            )
        case 'sgd':
            optim = SGD(
                params=model.parameters() if config.stage == 1 else None,
                lr=config.lr,
                momentum=config.momentum
            )
        case _:
            raise KeyError(f'Invalid optim type {config.optim}')
        
        
    # init scheduler
    if config.scheduler.lower() == 'cosineannealinglr':
        scheduler = CosineAnnealingLR(
            optimizer=optim,
            T_max=config.total_steps,
            eta_min=0.1 * config.lr, # from llama paper
            verbose=True
        )
    else:
        raise KeyError(f'Invalid scheduler type {config.scheduler}')
    
    # init dataloaders
    match config.task:
        case 'lpixel_pretrain':
            train_loader = get_pixel_pretrain_dataloader(
                paths=config.dataset_paths,
                batch_size=config.sub_size,
                num_workers=config.mp_workers,
                render_config=config.render_config,
                n_skip=(config.current_step - 1) * config.num_grad_acc_step,
                rank=config.rank,
                world_size=config.world_size,
                seed=config.seed,
                model_type='None'
            )
            dev_loader = None
        case _:
            raise NotImplementedError(f'Task {config.task} has not been implemented yet!')
        
    scaler = None
    if config.mix_precision == 'fp16':
        scaler = ShardedGradScaler()
        
    optim_parts = {
        'optim': optim,
        'scheduler': scheduler,
        'scaler': scaler
    }

    init_progress_bar(config)
    init_render(config.render_config)

    return (model, train_loader, dev_loader, optim_parts)

@timeit
def train(config: ExpConfig):
    model, train_loader, dev_loader, optim_parts = init_exp(config)
    output('Experiment parepared, begin to train')
    dist_sync(config)

    train_loader = iter(train_loader)

    output(f'Total steps: {config.total_steps}')
    output(f'Current step: {config.current_step}')
    output(f'Subbatch size per GPU: {config.sub_size}')
    output(f'Num GPUs: {config.num_gpu}')
    output(f'Num acc steps: {config.num_grad_acc_step}')
    output(f'Batch size: {config.batch_size}')

    while config.current_step <= config.total_steps:
        output(f'Step: {config.current_step}')
        running_loss: float = 0.0
        model.train()
        with ExitStack() as stack:
            train_gacc_stack(stack, config, model)
            for _ in range(config.num_grad_acc_step - 1):
                graph: TGraph = next(train_loader)
                batch = graph.to_pixel().to(config.device_id)
                attention_mask = graph.get_attention_mask().to(config.device_id)
                loss = model(pixel_values=batch, attention_mask=attention_mask, coder_grad=False).loss / config.num_grad_acc_step
                backward(loss, optim_parts)

                running_loss += loss.item()

        with ExitStack() as stack:
            train_stack(stack, config, model)

            graph: TGraph = next(train_loader)
            batch = graph.to_pixel().to(config.device_id)
            attention_mask = graph.get_attention_mask().to(config.device_id)
            loss = model(pixel_values=batch, attention_mask=attention_mask, coder_grad=False).loss / config.num_grad_acc_step
            backward(loss, optim_parts)

            running_loss += loss.item()

        step(config, optim_parts, model)
        running_loss = distributed_average(running_loss, config.device_id)
        output(f'Loss: {running_loss}')
        log({'training loss': running_loss})
        config.current_step += 1
