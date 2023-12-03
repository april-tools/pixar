import os
from contextlib import ExitStack
from functools import partial
import math

import torch
from torch.optim import(
    AdamW,
    SGD
)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from .train_config import (
    ExpConfig,
    init_wandb,
    get_config
)
from ..text_graph import TGraph
from ..utils import init_render, timeit, get_num_patches
from ..dataprocess import get_pretrain_dataloader
from ..modeling import (
    LatentModel,
    LPixelForMLM,
    LatentGPT2,
    Discriminator
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
    dist_sync,
    prepare_model,
    save_exp
)

import wandb

from transformers import logging
from logging import Logger
logger: Logger = None

def init_exp(config: ExpConfig) -> tuple[LatentModel, callable, callable, dict, Discriminator, dict]:
    '''
    Initialize models, wraps, optimizers, and dataloaders
    '''
    model, optim_parts, discriminator, disc_optim_parts = prepare_model(config)
    global logger
    logger = logging.get_logger(f'RANK{config.rank}')
        
    # init dataloaders
    match config.task:
        case 'gan_pretrain':
            def train_loader_fn(n_skip: int) -> DataLoader:
                return get_pretrain_dataloader(
                    data_conf=config.pretrain_dataset_config,
                    render_conf=config.render_config,
                    cache_path=config.dataset_cache_path,
                    batch_size=config.sub_size,
                    n_skip=n_skip,
                    num_workers=config.mp_workers,
                    rank=config.rank,
                    world_size=config.world_size,
                    mask_ratio=config.mask_ratio,
                    mask_type=config.mask_type
                )
            dev_loader_fn = None
        case _:
            raise NotImplementedError(f'Task {config.task} has not been implemented yet!')

    return (model, train_loader_fn, dev_loader_fn, optim_parts, discriminator, disc_optim_parts)

@timeit
def train(config: ExpConfig):
    model, train_loader_fn, dev_loader_fn, optim_parts, discriminator, disc_optim_parts = init_exp(config)
    model: LatentModel
    output('Experiment parepared, begin to train')
    dist_sync(config)

    compressor = None
    if model.compressor is not None:
        compressor = model.compressor.__class__(config.compressor_path)
        compressor.eval()
        
    train_loader = train_loader_fn(config._num_trained_samples)
    train_loader = iter(train_loader)
        
    output(f'Total steps: {config.total_steps}')
    output(f'Current step: {config.current_step}')
    output(f'Subbatch size per GPU: {config.sub_size}')
    output(f'Num GPUs: {config.num_gpu}')
    output(f'Num acc steps: {config.num_grad_acc_step}')
    output(f'Batch size: {config.batch_size}')
    
    begin_step = config.current_step
    grad_ratio: float = -0.1
    while config.current_step <= config.gan_total_steps + begin_step:
        output(f'Step: {config.current_step}')
        running_recon_loss: float = 0.0
        running_gan_loss: float = 0.0
        running_loss: float = 0.0
        running_disc_loss: float = 0.0

        model.backbone.train()
        model.compressor.eval() if model.compressor is not None else ...
        fakes: list[TGraph] = []
        reals: list[TGraph] = []
        
        running_recon_grad: float = 0.0
        running_gan_grad: float = 0.0

        output(f'Gan ratio {config.currnt_gan_ratio}')
        with ExitStack() as stack:
            train_gacc_stack(stack, config, model.backbone)

            for _ in range(config.num_grad_acc_step - 1):
                try:
                    graph: TGraph = next(train_loader)
                except StopIteration:
                    train_loader = train_loader_fn(0)
                    train_loader = iter(train_loader)
                    graph: TGraph = next(train_loader)
                    output(f'Epoch {config.epoch} finished!')
                    config.epoch += 1
                    
                graph.set_device(config.device_id)
                reals.append(graph)

                pred = model(graph)
                fakes.append(pred)  # collect the predictions for discriminator training

                # The backbone try to maximum the probability of generate real images, so it's 1 here
                recon_loss: torch.Tensor = pred.loss / config.num_grad_acc_step
                backward(recon_loss, optim_parts, retain_graph=True)
                if isinstance(model.backbone, DDP):
                    recon_grad = model.backbone.module.out_proj.weight.grad.mean().item() - running_recon_grad
                else:
                    model: LatentModel
                    recon_grad = model.backbone.out_proj.weight.grad.mean().item() - running_recon_grad
                recon_grad = abs(recon_grad)
                running_recon_grad += recon_grad
                if grad_ratio < 0:
                    gan_loss: torch.Tensor = discriminator.forward(pred, 1)[1] / config.num_grad_acc_step

                else:
                    gan_loss: torch.Tensor = discriminator.forward(pred, 1)[1] / config.num_grad_acc_step * (grad_ratio * config.currnt_gan_ratio + 1e-8)
                # loss: torch.Tensor = gan_loss * config.currnt_gan_ratio + (1 - config.currnt_gan_ratio) * recon_loss
                
                # calculate the gradients for the language backbone
                backward(gan_loss, optim_parts)
                if grad_ratio < 0:
                    gan_grad = model.backbone.module.out_proj.weight.grad.mean().item() - running_recon_grad
                else:
                    gan_grad = (model.backbone.module.out_proj.weight.grad.mean().item() - running_recon_grad) / (grad_ratio * config.currnt_gan_ratio + 1e-8)
                gan_grad = abs(gan_grad)
                running_gan_grad += gan_grad
                
                if grad_ratio < 0:
                    running_loss += recon_loss
                else:
                    running_loss += recon_loss + gan_loss * (grad_ratio * config.currnt_gan_ratio + 1e-8)
                running_gan_loss += gan_loss.item()
                running_recon_loss += recon_loss.item()

        with ExitStack() as stack:
            train_stack(stack, config, model.backbone)

            try:
                graph: TGraph = next(train_loader)
            except StopIteration:
                train_loader = train_loader_fn(0)
                train_loader = iter(train_loader)
                graph: TGraph = next(train_loader)
                output(f'Epoch {config.epoch} finished!')
                config.epoch += 1
                    
            if (config.current_step % config.eval_freq == 0 or config.current_step == 1) and config.rank == 0:
                print(f'Save image input at step {config.current_step}')
                graph.to_file(config.image_sample_path('input'))

            graph.set_device(config.device_id)
            reals.append(graph)

            pred = model(graph)
            fakes.append(pred)  # collect the predictions for discriminator training

            # The backbone try to maximum the probability of generate real images, so it's 1 here
            recon_loss: torch.Tensor = pred.loss / config.num_grad_acc_step
            backward(recon_loss, optim_parts, retain_graph=True)
            if isinstance(model.backbone, DDP):
                recon_grad = model.backbone.module.out_proj.weight.grad.mean().item() - running_recon_grad
            else:
                model: LatentModel
                recon_grad = model.backbone.out_proj.weight.grad.mean().item() - running_recon_grad
            recon_grad = abs(recon_grad)
            running_recon_grad += recon_grad
            if grad_ratio < 0:
                gan_loss: torch.Tensor = discriminator.forward(pred, 1)[1] / config.num_grad_acc_step

            else:
                gan_loss: torch.Tensor = discriminator.forward(pred, 1)[1] / config.num_grad_acc_step * (grad_ratio * config.currnt_gan_ratio + 1e-6)
            # loss: torch.Tensor = gan_loss * config.currnt_gan_ratio + (1 - config.currnt_gan_ratio) * recon_loss
            
            # calculate the gradients for the language backbone
            backward(gan_loss, optim_parts)
            if grad_ratio < 0:
                gan_grad = model.backbone.module.out_proj.weight.grad.mean().item() - running_recon_grad
            else:
                gan_grad = (model.backbone.module.out_proj.weight.grad.mean().item() - running_recon_grad) / (grad_ratio * config.currnt_gan_ratio + 1e-6)
            gan_grad = abs(gan_grad)
            running_gan_grad += gan_grad
            # loss: torch.Tensor = gan_loss * config.currnt_gan_ratio + (1 - config.currnt_gan_ratio) * recon_loss
            
            # calculate the gradients for the language backbone

            if grad_ratio < 0:
                running_loss += recon_loss
            else:
                running_loss += recon_loss + gan_loss * (grad_ratio * config.currnt_gan_ratio + 1e-8)
            running_gan_loss += gan_loss.item()
            running_recon_loss += recon_loss.item()
            


        # Update the Latent model
        if grad_ratio < 0:
            optim_parts['optim'].zero_grad()    # do not update at the fisrt step
        step(config, optim_parts, model.backbone, update_progress_bar=False)
        
        
        running_recon_grad = distributed_average(running_recon_grad, config.device_id)
        running_gan_grad = distributed_average(running_gan_grad, config.device_id)
        grad_ratio = running_recon_grad / (running_gan_grad + 1e-8)
        running_loss = distributed_average(running_loss, config.device_id)
        running_gan_loss = distributed_average(running_gan_loss, config.device_id)
        running_recon_loss = distributed_average(running_recon_loss, config.device_id)
        
        logger.debug(f'grad_ratio:{grad_ratio}')
        
        output(f'recon_grad_scale: {running_recon_grad}')
        log({'recon_grad_scale': running_recon_grad})
        
        output(f'gan_grad_scale: {running_gan_grad}')
        log({'gan_grad_scale': running_gan_grad})
        
        output(f'grad_ratio: {grad_ratio}')
        log({'grad_ratio': grad_ratio})
        
        output(f'Loss: {running_loss}')
        log({'Loss': running_loss})

        output(f'Recon_loss: {running_recon_loss}')
        log({'Recon_loss': running_recon_loss})

        output(f'Gan_loss: {running_gan_loss}')
        log({'Gan_loss': running_gan_loss})        

        # Train the discriminator
        with ExitStack() as stack:
            train_gacc_stack(stack, config, model.backbone)
            disc_optim_parts['optim'].zero_grad()

            for real, fake in list(zip(reals, fakes))[:-1]:
                loss_r = discriminator.forward(real.detach_(), 1)[1] / config.num_grad_acc_step / 2
                backward(loss_r, disc_optim_parts)

                loss_f = discriminator.forward(fake.detach_(), 0)[1] / config.num_grad_acc_step / 2
                backward(loss_f, disc_optim_parts)


                running_disc_loss += loss_r.item() + loss_f.item()

        with ExitStack() as stack:
            train_gacc_stack(stack, config, model.backbone)
            disc_optim_parts['optim'].zero_grad()

            for real, fake in list(zip(reals, fakes))[-1:]:
                loss_r = discriminator.forward(real.detach_(), 1)[1] / config.num_grad_acc_step / 2
                backward(loss_r, disc_optim_parts)

                loss_f = discriminator.forward(fake.detach_(), 0)[1] / config.num_grad_acc_step / 2
                backward(loss_f, disc_optim_parts)


                running_disc_loss += loss_r.item() + loss_f.item()

            if (config.current_step % config.eval_freq == 0 or config.current_step == 1) and model.compressor is None and config.rank == 0:
                print(f'Save image output at step {config.current_step}')
                graph = real.detach_()
                results = fake.detach_()
                results.set_device('cpu')
                results.to_file(config.image_sample_path('output'))
                results.to_file(config.image_sample_path('output_with_mask'))
                # graph.set_device('cpu')
                # interleave = TGraph.reconstruct(graph, results, True)
                # interleave.to_file(config.image_sample_path('reconstruction'))

        # Update the discriminator
        step(config, disc_optim_parts, discriminator)
        running_disc_loss = distributed_average(running_disc_loss, config.device_id)
        output(f'Discriminator_loss: {running_disc_loss}')
        log({'Discriminator_loss': running_disc_loss})

        reals = []
        fakes = []

        if config.current_step % config.save_freq == 0:
            save_exp(model, config, str(config.current_step), discriminator)

        dist_sync(config)

        config.current_step += 1

    save_exp(model, config, 'last', discriminator)
    dist_sync(config)
