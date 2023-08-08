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
    save_exp
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
    train_loader, dev_loaders, metrics, num_label = get_glue_dataset(task=config.task, config=config)

    return (model, train_loader, dev_loaders, metrics, optim_parts)

def evaluate(model: LatentModel, loaders: list[DataLoader], config: ExpConfig, metrics: list[Metric]) -> None:
    model.eval()
    with ExitStack() as stack:
        eva_stack(stack, config, model.backbone)
        for lidx, loader in enumerate(loaders):
            running_metrics: list[Metric] = [metric() for metric in metrics]
            for graph in loader:
                graph: TGraph
                graph.set_device(config.device_id)
                
                preds: torch.Tensor = model(graph).predcits.flatten()
                gold = graph.labels.flatten()
                
                gold_gathered = []
                preds_gathered = []
                gather(gold, gold_gathered)
                gather(preds, preds_gathered)

                if config.rank == 0:
                    golds = torch.cat(gold_gathered)
                    assert golds.dim() == 1
                    preds = torch.cat(preds_gathered)
                    assert preds.dim() == 1
                    golds = golds.tolist()
                    preds = preds.tolist()
                    
                    for met in running_metrics:
                        met.accumulate(golden=golds, compare=preds)
            
            if config.rank == 0:
                for met in running_metrics:
                    name = f'{met.metric_name()}_{lidx}'
                    value = met.result()
                    
                    log({name: value})
                    config.update_metric(name, value)
            
            dist_sync(config)


def train(config: ExpConfig):
    # Initialize model and dataloaders
    
    model, train_loader, dev_loaders, metrics, optim_parts = init_exp(config)
    output('Experiment for GLUE prepared, begin to train')
    dist_sync(config)
    
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
            
            for _ in range(config.num_grad_acc_step):
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
        