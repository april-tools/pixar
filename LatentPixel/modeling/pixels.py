import os
from os import PathLike
from dataclasses import dataclass

import torch
from torch import nn

from transformers import PreTrainedModel, PretrainedConfig
from diffusers.models import AutoencoderKL

from pixel import PIXELForPreTraining, PIXELEmbeddings
from pixel import PIXELConfig, PIXELForPreTrainingOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from ..text_graph import TGraph
from ..config import ModelType

logger = logging.get_logger(__name__)


class LPixelForPreTraining(nn.Module):
    
    def __init__(
        self, 
        coder_type: ModelType, 
        pixel_path: str | PathLike, 
        coder_path: str | PathLike, 
        mask_ratio: float,
        image_size: tuple[int, int],
        patch_size: int,
        keep_decoder: bool = False
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
        if not keep_decoder:
            del self.coder.decoder
            self.coder.decoder = None

        # load the pixel
        self.pixel_config: PIXELConfig = PIXELConfig.from_pretrained(pixel_path)
        self.pixel_config.mask_ratio = mask_ratio
        self.pixel_config.image_size = image_size
        self.pixel_config.patch_size = patch_size
        self.pixel_config.num_channels = self.latent_channels
        self.pixel: PIXELForPreTraining = PIXELForPreTraining.from_pretrained(pixel_path, config=self.pixel_config, ignore_mismatched_sizes=True)
        self.pixel.vit.embeddings = PIXELEmbeddings(self.pixel_config)

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
        coder_grad: bool | None = False
    ) -> PIXELForPreTrainingOutput | torch.Tensor:
        if coder_grad:
            latent = self.coder.encode(pixel_values).latent_dist.mode()
        else:
            with torch.no_grad():
                latent = self.coder.encode(pixel_values).latent_dist.mode()

        # run the pixel model
        result: PIXELForPreTrainingOutput = self.pixel(pixel_values=latent, attention_mask=attention_mask, patch_mask=patch_mask)

        if self.coder.decoder is None:
            return result
        
        if coder_grad:
            pixel_out = self.pixel.unpatchify(result.logits)
            r_latent = TGraph()
            r_latent._value = pixel_out
            r_latent = r_latent.unsquarelize(self.pixel_config.patch_size)._value
            
            reconstructed = self.coder.decode(r_latent).sample
            result.logits = reconstructed
        else:
            with torch.no_grad():
                pixel_out = self.pixel.unpatchify(result.logits)
                r_latent = TGraph()
                r_latent._value = pixel_out
                r_latent = r_latent.unsquarelize(self.pixel_config.patch_size)._value
                
                reconstructed = self.coder.decode(r_latent).sample
                result.logits = reconstructed

        return result
    
    def get_connection_layers(self) -> nn.ModuleList:
        return self.connection_layers
    
    def save_model(self, path: str | PathLike, only_pixel: bool = False) -> None:
        pixel_path = os.path.join(path, 'pixel')
        coder_path = os.path.join(path, 'coder')
        self.pixel.save_pretrained(pixel_path)
        
        if not only_pixel:
            self.coder.save_pretrained(coder_path)
