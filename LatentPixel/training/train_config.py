from __future__ import annotations
import os
from os import PathLike, makedirs
from os.path import join, exists
from pathlib import Path
from time import strftime
from collections import defaultdict
from typing import Any, TypeVar

from dataclasses import dataclass, field, asdict
import json

from ..config import RenderConfig, PretrainDatasetConfig
from ..utils import params2dict
from ..metrics import (
    MC,
    Accuracy,
    F1,
    PC,
    SC
)

import wandb

SEED = 42
RENDER_PATH = 'storage/pixel-base'
CHECK_PATH = 'storage/checkpoints'
WANDB_PROJECT = 'PIXEL++'
WANDB_TEAM = 'mlp-awesome'

# glue metrics
GLUE_META = [
    ('cola', ([MC], 2)),
    ('sst2', ([Accuracy], 2)),
    ('mrpc', ([Accuracy, F1], 2)),
    ('stsb', ([PC, SC], 1)),
    ('qqp', ([Accuracy, F1], 2)),
    ('mnli', ([Accuracy], 3)),
    ('qnli', ([Accuracy], 2)),
    ('rte', ([Accuracy], 2)),
    ('wnli', ([Accuracy], 2))
]

# experiemnt hyperparameters and their default values
# fields not begin with _ are arguments
# all of these fields will update to wandb
@dataclass
class ExpConfig:
    
    # you can use these fields as command line arguments
    model: str = 'LPixelForMLM' # or LatentGPT2 or LPixelForClassification
    init_path: str | PathLike = ''
    backbone_path: str | PathLike = ''
    compressor_path: str | PathLike = ''
    compressor_name: str = ''
    discriminator_path: str | PathLike = ''
    render_path: str | PathLike = RENDER_PATH
    checkpoint_path: str | PathLike = CHECK_PATH
    dataset_paths: list[str | PathLike] = field(default_factory=lambda: ['']) 
    prerendered: bool = False
    dataset_cache_path: str | PathLike = 'storage/cache'
    dataset_num_shards: int = 256
    seed: int = SEED
    exp_type: str = 'debug'
    task: str = 'lpixel_pretrain'
    finetune_task: str = ''
    glue_task: str = ''
    stage: int = 1
    optim: str = 'AdamW'
    scheduler: str = 'CosineAnnealingLR'
    warm_up_step: int = -1
    batch_size: int = 256   # The real batch_size, it should be equal to num_gpu * sub_size * num_gradient_acc_steps
    sub_size: int = 128     # actual batch size for 1 gradient acc step on 1 gpu
    eval_batch_size: int = 256
    eval_num_batch: int = 12
    lr: float = 1e-5
    beta1: float = 0.9
    beta2: float = 0.95
    decay: float = 0.01
    momentum: float = 0.95
    clip: float = 1.0
    gan_total_steps: int = 3000
    gan_delay_steps: int = 100
    gan_ratio: float = 0.5
    gan_ratio_warm_up_steps: int = 100
    gan_lr: float = 1e-5
    gan_lr_warm_up_steps: int = 100
    mask_ratio: float = 0.25
    mask_type: str = 'span'
    total_steps: int = 4000 # number of parameter update steps
    stop_step: int = 4000   # number of steps of current train
    max_token: int = 512    # 1024 for gpt2, 2048 for llama 7b
    eval_freq: int = 100
    begin_eval: int = 0
    save_freq: int = 1000
    best_save_freq: int = 100
    test_gpu_usibility: bool = False
    dpi: int = 120
    font_size: int = 8
    pad_size: int = 3
    pixels_per_patch: int = 16
    patch_len: int = 1
    max_seq_length: int = 529
    max_len: int = 1000  # max number of characters in one image
    compress_ratio: int = 8
    font_file: str = 'GoNotoCurrent.ttf'
    num_channel: int = 3
    binary: bool = False
    rgb: bool = True
    latent_norm: bool = False # whether to normalize the input in the latent space
    
    torch_compile: bool = False # whether to compile the model into a static graph (refer to pytorch 2.0)
    dynamic_shape: bool = False # whether to use dynamic input shape while compiling
    mix_precision: str = 'no'   # mixed precision could be bf16, fp16 or no
    half_precision: bool = False    # train in pure fp16 precision
    half_coder: bool = False
    gradient_checkpointing: bool = False    # save memory but reduce more computation

    num_gpu_per_node: int = 1
    num_node: int = 2
    mp_workers: int = 4 # number of paraller tokenizer workers per gpu

    shard_strategy: str = 'no'    # sharding strategy, grad or full or no
    offload_to_cpu: bool = False    # offload sharded data to cpu memory, reduce g memory cost but increase cpu memory cost
    backward_prefetch: str = 'pre'  # backward prefetch policy, pre or post

    on_cpu: bool = False    # train on CPU, seems like a useless option
    current_step: int = 1   # current training step, this will be updated automatically, do not neet to specify
    num_trained_tokens: int = 0
    shuffle_dataset: bool = False
    init: bool = False
    no_ckpt: bool = False
    no_log: bool = False
    is_continue_train: bool = False
    
    # below fields are not command line arguments
    _best_loss: float = 1e9
    _best_loss_step: int = -1
    _best_dev_loss: float = 1e9
    _best_dev_loss_step: int = -1
    _timestamp: str = None
    _continued: bool = False
    _name: str = None
    _epoch: int = 1
    _num_trained_samples: int = 0
    _num_grad_acc_step: int = -1
    _num_gpu: int = -1
    _rank: int = -1
    _world_size: int = -1
    _device_id: int = -1
    _local_world_size: int = -1
    _render_config: dict = None
    _metrics: defaultdict = field(default_factory=dict) 
    _max_metrics: defaultdict = field(default_factory=dict) 
    _max_metrics_step: defaultdict = field(default_factory=dict) 
    _min_metrics: defaultdict = field(default_factory=dict) 
    _min_metrics_step: defaultdict = field(default_factory=dict)
    _begin_ckpt_path: str = None
    _begin_gan_step: int = None
    
    def update_metric(self, name: str, value: float):
        if name not in self._metrics:
            self._metrics[name] = []
            self._max_metrics[name] = -9999999999
            self._min_metrics[name] = 9999999999
            self._max_metrics_step[name] = -1
            self._min_metrics_step[name] = -1
        self._metrics[name].append(value)
        print(f'{name}: {value} at step {self.current_step}')
        
        if value > self._max_metrics[name]:
            self._max_metrics[name] = value
            self._max_metrics_step[name] = self.current_step
            print(f'Maximum value of {name} {value} updated at step {self.current_step}')
            
        if value < self._min_metrics[name]:
            self._min_metrics[name] = value
            self._min_metrics_step[name] = self.current_step
            print(f'Minimum value of {name} {value} updated at step {self.current_step}')
            
    @property
    def pretrain_dataset_config(self) -> PretrainDatasetConfig:
        self.dataset_paths.sort(reverse=True)
        return PretrainDatasetConfig(
            dataset_paths=self.dataset_paths,
            max_len=self.max_len,
            seed=self.seed,
            shuffle=self.shuffle_dataset,
            num_shards=self.dataset_num_shards,
            prerendered=self.prerendered
        )

    @property
    def currnt_gan_ratio(self) -> float:
        if self._begin_gan_step is None:
            self._begin_gan_step = self.current_step
            
        current_step = self.current_step - self._begin_gan_step
        if current_step > self.gan_ratio_warm_up_steps:
            ratio = self.gan_ratio
        else:
            ratio = current_step / self.gan_ratio_warm_up_steps * self.gan_ratio
        return ratio
            
    @property
    def num_labels(self) -> int:
        if self.finetune_task != 'glue':
            raise KeyError(f'{self.finetune_task} is not a classfication task')
                
        return dict(GLUE_META)[self.glue_task][1]

    @property
    def render_config(self) -> RenderConfig:
        if self._render_config is not None:
            return RenderConfig(**self._render_config)
        rconf = RenderConfig(
            path=self.render_path,
            dpi=self.dpi,
            font_size=self.font_size,
            pixels_per_patch=self.pixels_per_patch,
            pad_size=self.pad_size,
            font_file=self.font_file,
            patch_len=self.patch_len,
            max_seq_length=self.max_seq_length,
            rgb=self.rgb,
            binary=self.binary
        )
        self._render_config = rconf.to_dict()
        return rconf
    
    @property
    def latent_patch_size(self) -> int:
        return self.pixels_per_patch // self.compress_ratio

    @property
    def local_world_size(self) -> int:
        if self._local_world_size >= 0:
            return self._local_world_size
        if self.distributed:
            self._local_world_size = int(os.environ['LOCAL_WORLD_SIZE'])
            print('local_world_size:', self._local_world_size)
        else:
            self._local_world_size = 1
        return self._local_world_size

    @property
    def rank(self) -> int:
        if self._rank >= 0:
            return self._rank
        if self.distributed:
            self._rank = int(os.environ['RANK'])
        else:
            self._rank = 0
        return self._rank
    
    @property
    def world_size(self) -> int:
        if self._world_size >= 0:
            return self._world_size
        if self.distributed:
            self._world_size = int(os.environ['WORLD_SIZE'])
        else:
            self._world_size = 1
        return self._world_size

    @property
    def device_id(self) -> int:
        if self._device_id >= 0:
            return self._device_id
        if self.distributed:
            self._device_id = int(os.environ['LOCAL_RANK'])
            print('device_id', self._device_id)
        else:
            self._device_id = self.rank
        return self._device_id

    @property
    def num_gpu(self):
        if self._num_gpu > 0:
            return self._num_gpu
        self._num_gpu = self.num_gpu_per_node * self.num_node
        print(f'There are {self._num_gpu} gpus')
        return self._num_gpu
    
    @property
    def epoch(self) -> int:
        return self._epoch
    
    @epoch.setter
    def epoch(self, cur_epoch) -> None:
        self._epoch = cur_epoch
        return
    
    @property
    def timestamp(self) -> str:
        if self._timestamp is not None:
            return self._timestamp
        self._timestamp = strftime('%Y%m%d-%H%M%S')
        return self._timestamp
    
    @property
    def best_loss(self) -> float:
        return self._best_loss
    
    @best_loss.setter
    def best_loss(self, loss: float) -> None:
        self._best_loss = loss
        self._best_loss_step = self.current_step
        return
    
    @property
    def best_dev_loss(self) -> float:
        return self._best_dev_loss
    
    @best_dev_loss.setter
    def best_dev_loss(self, loss: float) -> None:
        self._best_dev_loss = loss
        self._best_loss_step = self.current_step
    
    @property
    def log_path(self) -> str | PathLike:
        path = join(
            self.checkpoint_path,
            self.exp_type,
            self.task,
            self.model,
            self.timestamp,
            'log'
        )
        if not exists(path):
            makedirs(path, exist_ok=True)
        return path
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def num_grad_acc_step(self) -> int:
        if self._num_grad_acc_step > 0:
            return self._num_grad_acc_step
        self._num_grad_acc_step = self.batch_size // (self.sub_size * self.num_gpu)
        self.batch_size = self.sub_size * self.num_gpu * self._num_grad_acc_step
        if self._num_grad_acc_step <= 0:
            raise ValueError(f"0 or negative graident accumulation obversed [{self._num_grad_acc_step}], plecase check batch size settings!")
        return self._num_grad_acc_step

    @property
    def distributed(self) -> bool:
        return True

    def model_path(self, name: str) -> str | PathLike:
        path = join(
            self.checkpoint_path,
            self.exp_type,
            self.task,
            self.model,
            self.timestamp,
            name
        )
        if not exists(path):
            makedirs(path, exist_ok=True)
        return path
    
    def image_sample_path(self, name: str) -> str | PathLike:
        path = join(self.model_path('img_samples'), str(self.current_step), name)
        if not exists(path):
            makedirs(path, exist_ok=True)
        return path
        
    def backbone_ckpt_path(self, name: str) -> str | PathLike:
        path = join(self.model_path(name), 'backbone')
        if not exists(path):
            makedirs(path, exist_ok=True)
        return path
    
    def coder_ckpt_path(self, name: str) -> str | PathLike:
        path = join(self.model_path(name), 'coder')
        if not exists(path):
            makedirs(path, exist_ok=True)
        return path
    
    def discriminator_ckpt_path(self, name: str) -> str | PathLike:
        path = join(self.model_path(name), 'discriminator')
        return path

    def optim_path(self, name: str) -> str | PathLike:
        return join(self.model_path(name), 'optim.bin')
    
    def scheduler_path(self, name: str) -> str | PathLike:
        return join(self.model_path(name), 'scheduler.bin')
    
    def config_path(self, name) -> str | PathLike:
        return join(self.model_path(name), 'exp_config.json')
    
    @property
    def continued(self) -> bool:
        return self._continued
    
    def save(self, name: str) -> str:
        self._name = name
        path = self.config_path(name)
        states = json.dumps(self.__dict__, indent=2)
        with open(path, 'w') as f:
            f.write(states)
            
        return states
    
    @classmethod
    def from_checkpoint(cls, path: str | PathLike) -> None | ExpConfig:
        config_path = join(path, 'exp_config.json')
        if exists(config_path):
            with open(config_path, 'r') as fin:
                config_dict = json.load(fin)
            exp_config = cls(**config_dict)
            exp_config._continued = True
        else:
            return None 
        return exp_config
    
    def continue_training(self) -> ExpConfig | None:
        ckpt_path = Path(self.backbone_path).parent.__str__()
        old = ExpConfig.from_checkpoint(ckpt_path)
        if old:
            self.current_step = old.current_step
            self._num_trained_samples = old._num_trained_samples
            self.total_steps = old.total_steps
            self._begin_ckpt_path = ckpt_path
            self._continued = True
            print(f'Find checkpoint of previous training, continue training at step {self.current_step}/{self.total_steps}')
            print(f'There are {self._num_trained_samples} samples trained.')
        return self
    
    def load_optim_path(self) -> str | None:
        if self._begin_ckpt_path:
            return join(self._begin_ckpt_path, 'optim.bin')
        return None
    
    def load_scheduler_path(self) -> str:
        if self._begin_ckpt_path:
            return join(self._begin_ckpt_path, 'scheduler.bin')
        return None

def init_wandb(config: ExpConfig) -> None:
    wandb.login(key=os.environ['WANDB_KEY'])
    wandb.init(
        job_type=config.exp_type,
        tags=[config.exp_type, config.model, config.task] + (['continued'] if config.continued else []),
        name=config.exp_type + config.timestamp,
        project=WANDB_PROJECT,
        entity=WANDB_TEAM,
        dir=config.log_path,
        config=config.__dict__  # upload all the exp configs to wandb
    )
            
# if the experiment continues from the past, use the old config, otherwise 
# init a new one
def get_config() -> ExpConfig:
    # build the argument parser and parse the commandline arguments
        
    return ExpConfig(**params2dict(asdict(ExpConfig())))
 