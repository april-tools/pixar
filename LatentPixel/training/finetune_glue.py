from contextlib import ExitStack

import torch
from torch.utils.data import DataLoader
from torch.distributed import gather

from ..text_graph import TGraph
from .train_config import (
    ExpConfig,
)
from .train_utils import (
    prepare_model_for_glue,
    output,
    dist_sync,
    train_gacc_stack,
    InfLoader,
    backward,
    step,
    distributed_average,
    log,
    eva_stack,
    save_exp,
    train_stack
)
from ..utils import (
    timeit
)
from ..modeling import (
    LatentModel
)
from ..dataprocess import get_glue_dataset
from ..metrics import Metric
        
def init_exp(config: ExpConfig) -> tuple[LatentModel, DataLoader, DataLoader, dict]:
    '''
    Initialize models, wraps, optimizers, and dataloaders
    '''
    model, optim_parts = prepare_model_for_glue(config)
        
    # init dataloaders
    train_loader, dev_loaders, metrics, num_label = get_glue_dataset(task=config.glue_task, config=config)

    return (model, train_loader, dev_loaders, metrics, optim_parts)

def evaluate(model: LatentModel, loaders: list[DataLoader], config: ExpConfig, metrics: list[Metric]) -> None:
    model.eval()
    with ExitStack() as stack:
        eva_stack(stack, config, model.backbone)
        for lidx, loader in enumerate(loaders):
            running_metrics: list[Metric] = [metric() for metric in metrics]
            print(f'Begin to evluate the {lidx}th validation loader')
            print(f'{len(loaders)} validation sets in total')
            bidx = 0
            for graph in loader:
                graph: TGraph
                graph.set_device(config.device_id)
                
                preds: torch.Tensor = model(graph).predcits.flatten()
                gold = graph.labels.flatten()
                print(f'RANK{config.rank} {bidx} preds shpe', preds.shape)
                print(f'RANK{config.rank} {bidx} gold shape', gold.shape)
                if preds.shape[0] != config.sub_size:
                    print(f'RANK {config.rank}: There are only {preds.shape[0]} samples in the last batch, padding with padding value -95288')
                    padded_preds = torch.ones(config.sub_size, dtype=preds.dtype, device=preds.device) * -95288
                    padded_golds = torch.ones(config.sub_size, dtype=gold.dtype, device=gold.device) * -95288

                    padded_preds[:preds.shape[0]] = preds
                    padded_golds[:gold.shape[0]] = gold

                    preds = padded_preds
                    gold = padded_golds
                    print(f'RANK {config.rank} padded preds', preds)
                    print(f'RANK {config.rank} padded golds', gold)
                
                gold_gathered = [torch.clone(gold) for _ in range(config.num_gpu)] if config.rank == 0 else []
                preds_gathered = [torch.clone(preds) for _ in range(config.num_gpu)] if config.rank == 0 else []
                gather(gold, gold_gathered)
                gather(preds, preds_gathered)
                print(f'RANK{config.rank} {bidx} gathered ')
                bidx += 1

                if config.rank == 0:
                    golds = torch.cat(gold_gathered)
                    assert golds.dim() == 1
                    preds = torch.cat(preds_gathered)
                    assert preds.dim() == 1
                    golds = golds.tolist()
                    preds = preds.tolist()
                    assert len(golds) == len(preds)

                    print(f'RANK {config.rank}: {len(golds)} golden samples gathered from all ranks')
                    print(f'RANK {config.rank}: {len(preds)} golden samples gathered from all ranks')
                    print('Filter the padding values')
                    golds_filtered = [g for g in golds if float(g) != -95288.0]
                    preds_filtered = [p for p in preds if float(p) != -95288.0]


                    print(f'RANK {config.rank}: {len(golds_filtered)} golden samples left')
                    print(f'RANK {config.rank}: {len(preds_filtered)} golden samples left')
                    
                    for met in running_metrics:
                        met.accumulate(golden=golds_filtered, compare=preds_filtered)

                    print('Metric accumulated on rank 0')
            
            if config.rank == 0:
                for met in running_metrics:
                    name = f'{met.metric_name()}_{lidx}'
                    value = met.result()
                    
                    log({name: value})
                    config.update_metric(name, value)
                    print(f'{len(met.golden)} samples in the validation set')
                    print(f'{len(met.compare)} samples in the validation set')
                    print('Show the first 32 predictions:')
                    print('Golden', met.golden[:32])
                    print('Preds', met.compare[:32])


def train(config: ExpConfig):
    # Initialize model and dataloaders
    
    model, train_loader, dev_loaders, metrics, optim_parts = init_exp(config)
    output('Experiment for GLUE prepared, begin to train')
    
    train_loader = InfLoader(train_loader, config)
    
    output(f'Total steps: {config.total_steps}')
    output(f'Current step: {config.current_step}')
    output(f'Subbatch size per GPU: {config.sub_size}')
    output(f'Num GPUs: {config.num_gpu}')
    output(f'Num acc steps: {config.num_grad_acc_step}')
    output(f'Batch size: {config.batch_size}')
    
    while config.current_step <= config.total_steps:
        output(f'Step: {config.current_step}')
        running_loss: float = 0.0
        
        model.backbone.train()
        model.coder.eval() if model.coder is not None else ...
        
        with ExitStack() as stack:
            train_gacc_stack(stack, config, model.backbone)
            
            for _ in range(config.num_grad_acc_step - 1):
                graph: TGraph = train_loader.next()
                graph.set_device(config.device_id)
                
                loss = model(graph).loss / config.num_grad_acc_step
                
                backward(loss, optim_parts)
                
                running_loss += loss.item()
                
        with ExitStack() as stack:
            train_stack(stack, config, model.backbone)
            
            graph: TGraph = train_loader.next()
            graph.set_device(config.device_id)
            
            loss = model(graph).loss / config.num_grad_acc_step
            
            backward(loss, optim_parts)
            
            running_loss += loss.item()
            
        step(config, optim_parts, model.backbone)
        
        running_loss = distributed_average(running_loss, config.device_id)
        output(f'Step {config.current_step} loss: {running_loss}')
        log({'training loss': running_loss})
        
        if config.current_step % config.eval_freq == 0:
            evaluate(model, dev_loaders, config, metrics)
            save_exp(model, config, str(config.current_step))
            
        dist_sync(config)
        
        config.current_step += 1
        