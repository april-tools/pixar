from LatentPixel.training.pretrain_latent import train, get_config, init_wandb
from LatentPixel.training.finetune_glue import train as train_glue
from LatentPixel.training.pretrain_gan import train as train_gan
from LatentPixel.training.train_autoencoder import train as train_autoencoder
from LatentPixel.training.train_utils import init_dist_environ
import os
import logging

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    logger.setLevel(logging.ERROR)

if __name__ == '__main__':
    os.system("taskset -p 0xffffffffffffffffffffff %d" % os.getpid())
    print('Parse training scripts')
    config = get_config()
    print('Training config parsed')
    if config.is_continue_train:
        config.continue_training()  # check if it's a continued training
    print('Training config initialized')
    init_dist_environ(config)
    print(config.__dict__)
    if config.rank == 0:
        init_wandb(config)
        
    if config.finetune_task == 'glue':
        train_glue(config)
    else:
        if len(config.discriminator_path) > 0:
            train_gan(config)
        elif config.model in ('CNNAutoencoder'):
            train_autoencoder(config)
        else:
            train(config)
