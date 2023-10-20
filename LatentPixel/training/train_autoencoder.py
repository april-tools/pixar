from torch.utils.data import DataLoader
from torch import Tensor

from .train_config import ExpConfig
from .train_utils import (
    timeit,
    prepare_model,
    output,
    dist_sync,
    ExitStack,
    train_gacc_stack,
    backward,
    train_stack,
    step,
    distributed_average,
    log,
    save_exp
)
from ..modeling import (
    LatentModel,
    CNNAutoencoder,
    CNNAutoencoderConfig,
    Compressor
)
from ..dataprocess import get_pixel_pretrain_dataloader
from ..text_graph import TGraph


@timeit
def init_exp(config: ExpConfig) -> tuple[Compressor, DataLoader, DataLoader, dict]:
    '''
    Initialize models, wraps, optimizers, and dataloaders
    '''
    model, optim_parts = prepare_model(config)
        
    # init dataloaders
    match config.task:
        case 'lpixel_pretrain':
            train_loader = get_pixel_pretrain_dataloader(
                paths=config.dataset_paths,
                batch_size=config.sub_size,
                num_workers=config.mp_workers,
                render_config=config.render_config,
                mask_ratio=config.mask_ratio,
                mask_type=config.mask_type,
                n_skip=(config.current_step - 1) * config.num_grad_acc_step,
                rank=config.rank,
                world_size=config.world_size,
                seed=config.seed,
                max_len=config.max_len
            )
            dev_loader = None
        case _:
            raise NotImplementedError(f'Task {config.task} has not been implemented yet!')

    return (model, train_loader, dev_loader, optim_parts)

@timeit
def train(config: ExpConfig):
    model, train_loader, dev_loader, optim_parts = init_exp(config)
    model: Compressor
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
                print('Patch_len: ', graph.patch_len)
                graph.set_device(config.device_id)
                preds: TGraph = model(graph)

                loss: Tensor = preds.loss / config.num_grad_acc_step

                backward(loss, optim_parts)

                running_loss += loss.item()

        with ExitStack() as stack:
            train_stack(stack, config, model)

            graph: TGraph = next(train_loader)                
            graph.set_device(config.device_id)

            results: TGraph = model(graph)
            loss = results.loss / config.num_grad_acc_step
            backward(loss, optim_parts, config)

            running_loss += loss.item()
            
            if (config.current_step % config.eval_freq == 0 or config.current_step == 1) and config.rank == 0:
                print(f'Save image input at step {config.current_step}')
                graph.set_device('cpu')
                graph.to_file(config.image_sample_path('input'))
                print(f'Save image output at step {config.current_step}')
                results.set_device('cpu')
                results.float()
                results.to_file(config.image_sample_path('output'))

        step(config, optim_parts, model)
        running_loss = distributed_average(running_loss, config.device_id)
        output(f'Loss: {running_loss}')
        log({'training loss': running_loss})

        if config.current_step % config.save_freq == 0:
            save_exp(model, config, str(config.current_step))

        dist_sync(config)

        config.current_step += 1

    save_exp(model, config, 'last')
    dist_sync(config)
