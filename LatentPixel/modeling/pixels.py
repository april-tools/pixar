from typing import Any, Callable, Iterator
import os
from os import PathLike

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from transformers import PreTrainedModel, PretrainedConfig
from diffusers.models import AutoencoderKL

from pixel import PIXELForPreTraining, PIXELEmbeddings
from pixel import PIXELConfig, PIXELForPreTrainingOutput
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from LatentPixel.text_graph import TGraph

from ..text_graph import TGraph
from ..config import ModelType
from .latent_model import LatentModel

logger = logging.get_logger(__name__)


class LPixelForMLM(LatentModel):

    def load_coder(self, path: str | PathLike) -> nn.Module:
        if path is None:
            self.coder = None
            self.latent_size = self.img_size
            print(f'Coder path is none, do not load coder for this model')
            return None
        
        self.coder: AutoencoderKL = AutoencoderKL.from_pretrained(path)
        assert self.latent_size[0] == self.coder.config['latent_channels']
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
        pixel_config = PIXELConfig.from_pretrained(path)
        pixel_config.image_size = self.latent_size[1:]
        pixel_config.patch_size = self.latent_size[1]
        pixel_config.num_channels = self.latent_size[0]
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
        decoded = self.coder.decode(img._value).sample
        result = TGraph.from_SD(
            img=decoded, 
            do_clip=True,
            attention_mask=img.attention_mask,
            patch_mask=img.attention_mask
        )
        result.patch_size = self.img_size[1]

        return result
    
    def latent_forward(self, img: TGraph) -> TGraph:
        output: PIXELForPreTrainingOutput = self.backbone(
            pixel_values=img._value,
            attention_mask=img.attention_mask,
            patch_mask=img.patch_mask
            )
        logits = self.backbone.unnormalize_pred(img._value, output.logits)
        logits = self.backbone.unpatchify(logits)

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
            print('The decoder of the coder is deleted')
        else:
            print('There is no coder for this model, skip the deletion')
        
class LPixelForClassification(LatentModel):

    pass