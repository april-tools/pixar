from typing import Any, Callable, Iterator
import os
from os import PathLike

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torchvision.transforms import PILToTensor, Compose, ToPILImage, Normalize

from transformers import PreTrainedModel, PretrainedConfig
from diffusers.models import AutoencoderKL

from pixel import PIXELForPreTraining, PIXELEmbeddings
from pixel import PIXELConfig, PIXELForPreTrainingOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from LatentPixel.text_graph import TGraph

from ..text_graph import TGraph
from ..config import LATENT_DEFAULT_MEAN, LATENT_DEFAULT_STD
from .latent_model import LatentModel

logger = logging.get_logger(__name__)


class LPixelForMLM(LatentModel):

    _latent_norm = Compose([Normalize(LATENT_DEFAULT_MEAN, LATENT_DEFAULT_STD)])
    _inv_latent_norm = Compose([Normalize(- LATENT_DEFAULT_MEAN / LATENT_DEFAULT_STD, 1 / LATENT_DEFAULT_STD)])

    def load_coder(self, path: str | PathLike) -> nn.Module:
        if path is None or len(path) == 0:
            self.coder = None
            self.latent_size = self.img_size
            print(f'Coder path is none, do not load coder for this model')
            return None
        
        print(f'loading the coder from {path}')
        coder: AutoencoderKL = AutoencoderKL.from_pretrained(path)
        assert self.latent_size[0] == coder.config['latent_channels']
        self.coder = coder
        return self.coder
    
    def save_coder(self, path: str | PathLike) -> None:
        if self.coder is None:
            print('Abandon the coder saving since the decoder is deleted.')
            return
        elif self.coder.decoder is None:
            print('Abandon the coder saving since the decoder is deleted.')
            return
        
        self.coder.save_pretrained(path)
    
    def load_backbone(self, path: str | PathLike) -> nn.Module:
        if path is None or len(path) == 0:
            print('No backbone')
            return None
        pixel_config = PIXELConfig.from_pretrained(path)
        pixel_config.image_size = self.latent_size[1:]
        pixel_config.patch_size = self.latent_size[1]
        pixel_config.num_channels = self.latent_size[0]
        pixel_config.norm_pix_loss = False
        self.backbone_config = pixel_config
        self.backbone: PIXELForPreTraining = PIXELForPreTraining.from_pretrained(path, config=pixel_config, ignore_mismatched_sizes=True)
        self.backbone.vit.embeddings.gen_mode = True
        return self.backbone
    
    def save_backbone(self, path: str | PathLike) -> None:
        if isinstance(self.backbone, PIXELForPreTraining):
            self.backbone.save_pretrained(path)
        elif isinstance(self.backbone, DistributedDataParallel):
            self.backbone.module.save_pretrained(path)
        else:
            raise NotImplementedError(f'Saving for {type(self.backbone)} has not been implemented!')
        
        print(f'PIXEL backbone saved!')
    
    @torch.no_grad()
    def encode(self, img: TGraph) -> TGraph:
        pixel_values = img.unsquarelize().to_SD()
        with torch.no_grad():
            latent = self.coder.encode(pixel_values).latent_dist.mode()
        
        return TGraph.from_value(
            value=latent,
            patch_size=self.latent_size[1],
            attention_mask=img.attention_mask,
            patch_mask=img.patch_mask,
            num_text_patches=img.patch_mask,
            num_gen_patches=img.num_gen_patches
        )
    
    @torch.no_grad()
    def decode(self, img: TGraph) -> TGraph:
        decoded = self.coder.decode(img.value).sample
        result = TGraph.from_SD(
            img=decoded, 
            do_clip=True,
            attention_mask=img.attention_mask,
            patch_mask=img.patch_mask
        )
        result.patch_size = self.img_size[1]

        return result
    
    @property
    def has_decoder(self) -> bool:
        if self.coder is None:
            return False
        if self.coder.decoder is None:
            return False
        
        return True
    
    def latent_forward(self, img: TGraph) -> TGraph:
        output: PIXELForPreTrainingOutput = self.backbone(
            pixel_values=self._latent_norm(img.value) if self.coder is not None else img.to_pixel(),
            attention_mask=img.attention_mask,
            patch_mask=img.patch_mask
            )
        if self.coder is None:
            return TGraph.from_pixel(output, True, patch_size=self.latent_size[1]).unsquarelize()
        
        if isinstance(self.backbone, PIXELForPreTraining):
            logits = self.backbone.unpatchify(output.logits)
            logits = self._inv_latent_norm(logits)
        else:
            logits = self.backbone.module.unpatchify(output.logits)
            logits = self._inv_latent_norm(logits)

        return TGraph.from_value(
            value=logits,
            attention_mask=img.attention_mask,
            patch_mask=output.mask,
            num_text_patches=img.num_text_patches,
            loss=output.loss,
            patch_size=self.latent_size[1]
        ).unsquarelize()
    
    def get_connection_layers(self) -> nn.Module:
        return nn.ModuleList([
            self.backbone.vit.embeddings,
            self.backbone.decoder.decoder_pred
        ])
    
    def get_backbone_parameters(self) -> Iterator[nn.Parameter]:
        return self.backbone.parameters()
    
    def get_connection_params(self) -> Iterator[nn.Parameter]:
        return self.get_connection_layers().parameters()
    
    def init_connection_layers(self) -> None:
        print('Reinitialize the connection layers for the latent pixel')
        self.backbone.vit.embeddings = PIXELEmbeddings(self.backbone_config)
        self.backbone.decoder.decoder_pred = nn.Linear(
            self.backbone_config.decoder_hidden_size,
            self.latent_size[1] ** 2 * self.latent_size[0], 
            bias=True
        )

    def delete_unused_layers(self) -> None:
        if self.coder is not None:
            del self.coder.decoder
            self.coder.decoder = None
            self._has_decoder = False
            print('The decoder of the coder is deleted')
        else:
            print('There is no coder for this model, skip the deletion')
        
class LPixelForClassification(LatentModel):

    pass