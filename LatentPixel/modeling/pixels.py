from typing import Any, Callable, Iterator
import os
from os import PathLike

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
from .latent_model import LatentModel

logger = logging.get_logger(__name__)


class LPixelForPreTraining(LatentModel):
    
    def __init__(
        self, 
        coder_type: ModelType, 
        mask_ratio: float,
        image_size: tuple[int, int],
        patch_size: int,
        pixel_path: str | PathLike = '', 
        coder_path: str | PathLike = '', 
        ckpt_path: str | PathLike = '',
        keep_decoder: bool = False,
        init_connection_layer: bool = False
        ):
        super().__init__()
        # save the init configurations
        self.coder_type = coder_type
        self.pixel_path = pixel_path
        self.coder_path = coder_path
        self.ckpt_path = ckpt_path
        self.mask_ratio = mask_ratio
        self.image_size = image_size
        self.patch_size = patch_size
        self.keep_decoder = keep_decoder

        if len(self.pixel_path) == 0:
            self.pixel_path = os.path.join([self.ckpt_path, 'pixel'])
        if len(self.coder_path) == 0:
            self.coder_path = os.path.join([self.ckpt_path, 'coder'])
                
        # load the coder
        self.load_coder(coder_path)
        if not keep_decoder:
            print('unload the decoder')
            del self.coder.decoder
            self.coder.decoder = None
            del self.coder.post_quant_conv
            self.coder.post_quant_conv = None

        # load the pixel
        self.load_backbone(pixel_path, init_connection_layer=init_connection_layer)

    def load_coder(self, path: str | os.PathLike, config: Any = None) -> nn.Module:
        # load the coder
        self.coder: AutoencoderKL = AutoencoderKL.from_pretrained(path)
        self.coder_config: dict = self.coder.config
        self.latent_channels: int = self.coder_config['latent_channels']
        return self.coder
    
    def load_backbone(self, path: str | PathLike, config: Any = None, init_connection_layer: bool = False) -> nn.Module:
        self.pixel_config: PIXELConfig = PIXELConfig.from_pretrained(path)
        self.pixel_config.mask_ratio = self.mask_ratio
        self.pixel_config.image_size = self.image_size
        self.pixel_config.patch_size = self.patch_size
        self.pixel_config.num_channels = self.latent_channels
        # self.pixel_config.norm_pix_loss = False
        self.pixel: PIXELForPreTraining = PIXELForPreTraining.from_pretrained(path, config=self.pixel_config, ignore_mismatched_sizes=True)
        if init_connection_layer:
            print('Reinitialize the connection layers')
            self.pixel.vit.embeddings = PIXELEmbeddings(self.pixel_config)
            self.pixel.decoder.decoder_pred = nn.Linear(self.pixel_config.decoder_hidden_size, self.pixel_config.patch_size ** 2 * self.pixel_config.num_channels, bias=True)
        else:
            print('Reuse the connection layers')
        return self.pixel

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
        
        result.logits = self.pixel.unnormalize_pred(latent, result.logits)
        
        if coder_grad:
            pixel_out = self.pixel.unpatchify(result.logits)
            r_latent = TGraph()
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
    
    def get_connection_layers(self) -> nn.Module:
        return nn.ModuleList([
            self.pixel.vit.embeddings,
            self.pixel.decoder.decoder_pred
        ])
    
    def get_backbone_parameters(self) -> Any:
        return self.pixel.parameters()
    
    def get_connection_params(self) -> Iterator[nn.Parameter]:
        return self.get_connection_layers().parameters()
    
    def save_backbone(self, path: str | PathLike) -> None:
        if isinstance(self.pixel, PIXELForPreTraining):
            self.pixel.save_pretrained(path)
        else:
            self.pixel.module.save_pretrained(path)

    def save_coder(self, path: str | PathLike) -> None:
        if self.keep_decoder:
            self.coder.save_pretrained(path)
        else:
            print('Abandon the coder saving since the decoder is deleted.')

    def save_model(self, path: str | PathLike, only_backbone: bool = False) -> None:
        pixel_path = os.path.join(path, 'pixel')
        coder_path = os.path.join(path, 'coder')
        self.pixel.save_pretrained(pixel_path)
        
        if not only_backbone:
            self.coder.save_pretrained(coder_path)

    def compile(self) -> None:
        self.pixel = torch.compile(self.pixel, mode='max-autotune', dynamic=False)

    def wrap(self, wrap_fn: Callable[..., Any]) -> None:
        self.pixel.vit.embeddings = wrap_fn(self.pixel.vit.embeddings)
        self.pixel.vit.encoder = wrap_fn(self.pixel.vit.encoder)
        self.pixel.vit.layernorm = wrap_fn(self.pixel.vit.layernorm)
        self.pixel.decoder.decoder_embed = wrap_fn(self.pixel.decoder.decoder_embed)
        self.pixel.decoder.wrapped = wrap_fn(self.pixel.decoder.wrapped)
        self.pixel.decoder.decoder_layers = wrap_fn(self.pixel.decoder.decoder_layers)
        self.pixel.decoder.decoder_norm = wrap_fn(self.pixel.decoder.decoder_norm)
        self.pixel.decoder.decoder_pred = wrap_fn(self.pixel.decoder.decoder_pred)
