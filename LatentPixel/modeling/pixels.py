from os import PathLike
from dataclasses import dataclass

import torch
from torch import nn

from transformers import PreTrainedModel, PretrainedConfig
from diffusers.models import AutoencoderKL

from pixel import PIXELForPreTraining
from pixel import PIXELConfig, PIXELForPreTrainingOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .datatypes import Coder

logger = logging.get_logger(__name__)


class LPixelForPreTraining(nn.Module):
    
    def __init__(
        self, 
        coder_type: Coder, 
        pixel_path: str | PathLike, 
        coder_path: str | PathLike, 
        mask_ratio: float,
        image_size: tuple[int, int],
        patch_size: int
        ):
        super().__init__()
        # save the init configurations
        self.coder_type = coder_type
        self.pixel_path = pixel_path
        self.coder_path = coder_path
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        
        # load the coder
        self.coder: AutoencoderKL = coder_type.value.from_pretrained(coder_path)
        self.coder_config: dict = self.coder.config
        self.latent_channels: int = self.coder_config['latent_channels']

        # load the pixel
        self.pixel_config: PIXELConfig = PIXELConfig.from_pretrained(pixel_path)
        self.pixel_config.mask_ratio = mask_ratio
        self.pixel_config.image_size = image_size
        self.pixel_config.patch_size = patch_size
        self.pixel: PIXELForPreTraining = PIXELForPreTraining.from_pretrained(pixel_path, config=self.pixel_config, ignore_mismatched_sizes=True)

        # set the connection layers
        self.connection_layers = nn.ModuleList([
            self.pixel.vit.embeddings,
            self.pixel.decoder.decoder_pred
        ])
        
    def forward(
        self,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        patch_mask: torch.Tensor | None = None,
        reconstruct: bool | None = False
    ) -> PIXELForPreTrainingOutput | torch.Tensor:
        latent = self.coder.encode(pixel_values).latent_dist.mode()
        result: PIXELForPreTrainingOutput = self.pixel(pixel_values=latent, attention_mask=attention_mask, patch_mask=patch_mask)

        if not reconstruct:
            return result
        
        r_latent = result.logits
        
        return r_latent