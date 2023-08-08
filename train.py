from LatentPixel.training.pretrain_latent import train, get_config, init_wandb
from LatentPixel.training.finetune_glue import train as train_glue
from LatentPixel.training.train_utils import init_dist_environ

if __name__ == '__main__':
    config = get_config()
    print('Training config initialized')
    init_dist_environ(config)
    print(config.__dict__)
    if config.rank == 0:
        init_wandb(config)
        
    if config.finetune_task == 'glue':
        train_glue(config)
    else:
        train(config)
