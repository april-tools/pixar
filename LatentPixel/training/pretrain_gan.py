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
    Discriminator,
    LlamaDiscriminator
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

def last_layer_grad(model: DDP | LatentModel) -> float:
    if isinstance(model.backbone, DDP):
        return torch.clone(model.backbone.module.out_proj.weight.grad).detach()
    else:
        return torch.clone(model.backbone.out_proj.weight.grad).detach()

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
        output(f'################# Step: {config.current_step} #################')
        running_recon_loss: float = 0.0
        running_gan_loss: float = 0.0
        running_loss: float = 0.0
        running_disc_loss: float = 0.0

        model.backbone.train()
        model.compressor.eval() if model.compressor is not None else ...
        fakes: list[TGraph] = []
        reals: list[TGraph] = []
        
        running_recon_grad: torch.Tensor = torch.tensor(0.0, requires_grad=False)
        running_gan_grad: torch.Tensor = torch.tensor(0.0, requires_grad=False)
        pre_grad: torch.Tensor = torch.tensor(0.0, requires_grad=False)
        running_acc: float = 0.0

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

                # calculate reconstruction loss and the gradient at the ouput layer
                recon_loss: torch.Tensor = pred.loss / config.num_grad_acc_step
                backward(recon_loss, optim_parts, retain_graph=True)
                current_grad = last_layer_grad(model)
                running_recon_grad = running_recon_grad + current_grad - pre_grad
                pre_grad = current_grad

                # The backbone try to maximum the probability of generate real images, so it's 1 here
                if config.discriminator_path == 'self':
                    discriminator: LlamaDiscriminator
                    if grad_ratio < 0:
                        gan_loss: torch.Tensor = discriminator.forward(graph, pred, 1, config.num_gan_sample)[1] / config.num_grad_acc_step
                    else:
                        gan_loss: torch.Tensor = discriminator.forward(graph, pred, 1, config.num_gan_sample)[1] / config.num_grad_acc_step * (grad_ratio * config.currnt_gan_ratio + 1e-8)
                else:
                    if grad_ratio < 0:
                        gan_loss: torch.Tensor = discriminator.forward(pred, 1)[1] / config.num_grad_acc_step
                    else:
                        gan_loss: torch.Tensor = discriminator.forward(pred, 1)[1] / config.num_grad_acc_step * (grad_ratio * config.currnt_gan_ratio + 1e-8)
                    
                # calculate the gradients for the language backbone from discriminator
                backward(gan_loss, optim_parts)
                current_grad = last_layer_grad(model)
                if grad_ratio < 0:
                    running_gan_grad = running_gan_grad + (current_grad - pre_grad)
                else:
                    running_gan_grad = running_gan_grad + (current_grad - pre_grad) / (grad_ratio * config.currnt_gan_ratio + 1e-8)
                pre_grad = current_grad
                
                if grad_ratio < 0:
                    running_loss += recon_loss.item()
                else:
                    running_loss += (recon_loss + gan_loss).item()
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
            current_grad = last_layer_grad(model)
            running_recon_grad = running_recon_grad + current_grad - pre_grad
            pre_grad = current_grad
            
            if config.discriminator_path == 'self':
                discriminator: LlamaDiscriminator
                if grad_ratio < 0:
                    gan_loss: torch.Tensor = discriminator.forward(graph, pred, 1, config.num_gan_sample)[1] / config.num_grad_acc_step
                else:
                    gan_loss: torch.Tensor = discriminator.forward(graph, pred, 1, config.num_gan_sample)[1] / config.num_grad_acc_step * (grad_ratio * config.currnt_gan_ratio + 1e-8)
            else:
                if grad_ratio < 0:
                    gan_loss: torch.Tensor = discriminator.forward(pred, 1)[1] / config.num_grad_acc_step
                else:
                    gan_loss: torch.Tensor = discriminator.forward(pred, 1)[1] / config.num_grad_acc_step * (grad_ratio * config.currnt_gan_ratio + 1e-8)
                
            
            # calculate the gradients for the language backbone from discriminator
            backward(gan_loss, optim_parts)
            current_grad = last_layer_grad(model)
            if grad_ratio < 0:
                running_gan_grad = running_gan_grad + (current_grad - pre_grad)
            else:
                running_gan_grad = running_gan_grad + (current_grad - pre_grad) / (grad_ratio * config.currnt_gan_ratio + 1e-8)
            pre_grad = current_grad
            
            if grad_ratio < 0:
                running_loss += recon_loss.item()
            else:
                running_loss += (recon_loss + gan_loss).item()
            running_gan_loss += gan_loss.item()
            running_recon_loss += recon_loss.item()
            


        # Update the Latent model
        if grad_ratio < 0:
            optim_parts['optim'].zero_grad()    # do not update at the fisrt step
        step(config, optim_parts, model.backbone, update_progress_bar=False)
                
        running_recon_grad = distributed_average(running_recon_grad, config.device_id).abs().mean().item()
        running_gan_grad = distributed_average(running_gan_grad, config.device_id).abs().mean().item()
        
        grad_ratio = running_recon_grad / (running_gan_grad + 1e-8)
        
        running_loss = distributed_average(running_loss, config.device_id)
        running_gan_loss = distributed_average(running_gan_loss, config.device_id)
        running_recon_loss = distributed_average(running_recon_loss, config.device_id)
        grad_ratio = distributed_average(grad_ratio, config.device_id)
        
        logger.info(f'grad_ratio:{grad_ratio}')
        
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
        disc_optim_parts['optim'].zero_grad()
        with ExitStack() as stack:
            train_gacc_stack(stack, config, model.backbone)

            for real, fake in list(zip(reals, fakes))[:-1]:
                if config.discriminator_path == 'self':
                    acc_r, loss_r = discriminator.forward(real.detach_(), real.detach_(), 1, config.num_gan_sample, True)
                else:
                    acc_r, loss_r = discriminator.forward(real.detach_(), 1)
                loss_r = loss_r / config.num_grad_acc_step / 2
                acc_r = acc_r / config.num_grad_acc_step / 2
                backward(loss_r, disc_optim_parts)

                if config.discriminator_path == 'self':
                    acc_f, loss_f = discriminator.forward(real.detach_(), fake.detach_(), 0, config.num_gan_sample)
                else:
                    acc_f, loss_f = discriminator.forward(fake.detach_(), 0)
                loss_f = loss_f / config.num_grad_acc_step / 2
                acc_f = acc_f / config.num_grad_acc_step / 2
                backward(loss_f, disc_optim_parts)


                running_disc_loss += loss_r.item() + loss_f.item()
                running_acc += acc_f + acc_r
        
        with ExitStack() as stack:
            train_gacc_stack(stack, config, model.backbone)

            for real, fake in list(zip(reals, fakes))[-1:]:
                if config.discriminator_path == 'self':
                    acc_r, loss_r = discriminator.forward(real.detach_(), real.detach_(), 1, config.num_gan_sample, True)
                else:
                    acc_r, loss_r = discriminator.forward(real.detach_(), 1)
                loss_r = loss_r / config.num_grad_acc_step / 2
                acc_r = acc_r / config.num_grad_acc_step / 2
                backward(loss_r, disc_optim_parts)

                if config.discriminator_path == 'self':
                    acc_f, loss_f = discriminator.forward(real.detach_(), fake.detach_(), 0, config.num_gan_sample)
                else:
                    acc_f, loss_f = discriminator.forward(fake.detach_(), 0)
                loss_f = loss_f / config.num_grad_acc_step / 2
                acc_f = acc_f / config.num_grad_acc_step / 2
                backward(loss_f, disc_optim_parts)

                running_disc_loss += loss_r.item() + loss_f.item()
                running_acc += acc_f + acc_r

            if (config.current_step % config.eval_freq == 0 or config.current_step == 1) and model.compressor is None and config.rank == 0:
                print(f'Save image output at step {config.current_step}')
                graph = real.detach_()
                results = fake.detach_()
                results.set_device('cpu')
                results.to_file(config.image_sample_path('output'))
                
        # Update the discriminator
        step(config, disc_optim_parts, discriminator)
        running_disc_loss = distributed_average(running_disc_loss, config.device_id)
        output(f'Discriminator_loss: {running_disc_loss}')
        log({'Discriminator_loss': running_disc_loss})
        output(f'Discriminator_acc: {running_acc}')
        log({'Discriminator_acc': running_acc})
        reals = []
        fakes = []

        if config.current_step % config.save_freq == 0:
            save_exp(model, config, str(config.current_step), discriminator)

        dist_sync(config)

        config.current_step += 1

    save_exp(model, config, 'last', discriminator)
    dist_sync(config)
