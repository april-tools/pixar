import os
from os import PathLike
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss, BCEWithLogitsLoss
from transformers import LlamaModel, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP, LlamaRMSNorm, _make_causal_mask, _expand_mask, LlamaPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, SequenceClassifierOutputWithPast
from transformers import logging
from diffusers import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel
from copy import deepcopy
import random

from LatentPixel.text_graph import TGraph

from .latent_model import LatentModel
from ..utils import mask2img

#for flash attention
from transformers.utils import is_flash_attn_available
if is_flash_attn_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa
    from transformers.models.llama.modeling_llama import LlamaFlashAttention2
    

logger = logging.get_logger(__name__)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        if getattr(config, "flash", True) and is_flash_attn_available():
            logger.warning_once("flash attention enabled!")
        else:
            logger.warning_once("flash attention disabled!")
        self.self_attn = (
            LlamaAttention(config=config)
            if not getattr(config, "flash", False)
            else LlamaFlashAttention2(config=config)
        )
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        padding_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class LlamaForPatchCausalInference(LlamaModel):
    
    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.hidden_size


        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Model parallel
        self.gradient_checkpointing = False
        
        self.config = config
        self.init_projection()

        # Initialize weights and apply final processing
        self.post_init()
        
    def set_trainable_layers(self, num: int):
        for param in self.parameters():
            param.requires_grad = False
        print(f'Only allow the last {num} layers in generator be trainable')
        for layer in self.layers[-num:]:
            layer: LlamaDecoderLayer
            for param in layer.parameters():
                param.requires_grad = True
        
        for param in self.norm.parameters():
            param.requires_grad = True
            
        for param in self.out_proj.parameters():
            param.requires_grad = True

    def init_projection(self) -> None:
        config = self.config
        self.in_proj = nn.Conv2d(
            in_channels=config.num_channel,
            out_channels=config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size * config.patch_len),
            stride=config.patch_size * config.patch_len,
            bias=False
        )
        self.out_proj = nn.ConvTranspose2d(
            self.embed_dim, 
            config.num_channel, 
            (config.patch_size, config.patch_size * config.patch_len),
            stride=config.patch_size * config.patch_len,
            bias=False
        )
        
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask
      
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ignore_out_proj: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # map the input_embeds into vectors saperated by patches
        inputs_embeds = self.in_proj(inputs_embeds)
        inputs_embeds = inputs_embeds.flatten(2).transpose(1, 2)

        # >>>>>>> below are copied from llamaModel
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
            padding_mask = None
        else:
            if 0 in attention_mask:
                padding_mask = attention_mask
            else:
                padding_mask = None

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # <<< before are copied from Llama
        if not ignore_out_proj:
            hidden_states = hidden_states.transpose(1, 2).unsqueeze(2)
            hidden_states = self.out_proj(hidden_states)
        else:
            hidden_states = None

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        
class LatentLlama(LatentModel):
    
    def load_backbone(self, path: str | PathLike, num_latent_channel: int, latent_patch_size: int, patch_len: int, num_labels: int, binary: bool) -> nn.Module:
        Llama_config = LlamaConfig.from_pretrained(path)
        
        setattr(Llama_config, 'num_channel', num_latent_channel)
        setattr(Llama_config, 'patch_size', latent_patch_size)
        setattr(Llama_config, 'patch_len', patch_len)
        setattr(Llama_config, 'binary', binary)
        
        Llama: LlamaForPatchCausalInference = LlamaForPatchCausalInference.from_pretrained(path, config=Llama_config, ignore_mismatched_sizes=True)
        self.backbone = Llama

        del self.backbone.embed_tokens
        self.backbone.embed_tokens = None
        
        return Llama
    
    def save_backbone(self, path: str | PathLike) -> None:
        if isinstance(self.backbone, LlamaForPatchCausalInference):
            self.backbone.save_pretrained(path)
        elif isinstance(self.backbone, DistributedDataParallel):
            self.backbone.module.save_pretrained(path)
        else:
            raise NotImplementedError(f'Saving for {type(self.backbone)} has not been implemented!')
        
        print(f'Llama backbone saved to {path}')
            
    def latent_forward(self, img: TGraph) -> TGraph:
        target = img.value.float()     # target should within range [0, 1]
        if self.compressor is None:
            img_values = target * 2 - 1 # map values from [0, 1] range to [-1, 1] range
        else:
            img_values = target
        output: BaseModelOutputWithPastAndCrossAttentions = self.backbone.forward(
            attention_mask=img.attention_mask,
            inputs_embeds=img_values
        )
        pred = output.last_hidden_state

        # attention_mask = mask2img(img.attention_mask, self.latent_patch_size, self.patch_len)
        # patch_width = self.latent_patch_size * self.patch_len
        # attention_mask.unsqueeze_(1)    # add the channel dimension
        # attention_mask = attention_mask[..., patch_width:]

        # loss = (pred[..., :-patch_width] - img_values[..., patch_width:]) ** 2
        # loss = (loss * attention_mask).sum() / (attention_mask.sum() * self.num_latent_channel)  # mean loss on removed patches

        loss = self.forward_loss(pred, target, img.attention_mask)

        if not self.binary and self.compressor is None:
            pred = (pred + 1) / 2   #   map value from [-1, 1] to [0, 1]

        result = TGraph.from_value(
            value=pred,
            attention_mask=img.attention_mask,
            patch_mask=img.patch_mask,
            num_text_patches=img.num_text_patches,
            loss=loss,
            patch_size=self.latent_patch_size
        )
        result._binary = False

        return result

    def forward_loss(self, pred: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        patch_width = self.latent_patch_size * self.patch_len   # the width of a patch
        pred = pred[..., :-patch_width] # ignore the last patch of predicion, because we don't have the answer for it
        target = target[..., patch_width:]  # ignore the first patch of input, because we don't have it's previous patch
        mask = mask2img(attention_mask, self.latent_patch_size, self.patch_len) # make [11100] mask into a mask image
        mask.unsqueeze_(1)
        mask = mask[..., patch_width:]
        bs, c, h, w = pred.shape

        if self.binary and self.compressor is None:
            mask = mask.reshape(-1).contiguous()
            loss = nn.BCEWithLogitsLoss(reduction='none').forward(pred.reshape(-1).contiguous(), target.reshape(-1).contiguous()) * mask
            loss = loss.reshape(bs, c * h * w)
            mask = mask.reshape(bs, c * h * w)
            loss = loss.sum(dim=1) / mask.sum(dim=1)
            loss = loss.mean()  # average among the batch
        else:
            loss = ((pred - target)**2 * mask).sum() / (mask.sum() * self.num_latent_channel)   # MSE loss

        return loss
        
    def get_connection_layers(self) -> nn.Module:
        return nn.ModuleList([
            self.backbone.in_proj,
            self.backbone.out_proj
        ])
    
    def init_connection_layers(self) -> None:
        print('init the connection layers')
        self.backbone.init_projection()
        return
    
    def delete_unused_layers(self) -> None:
        if self.compressor is not None:
            print('delete the decoder')
            del self.compressor.decoder
            self.compressor.decoder = None
        return


class LlamaForPatchSequenceClassification(LlamaModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.embed_dim = config.hidden_size


        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Model parallel
        self.gradient_checkpointing = False
        
        self.config = config
        self.init_projection()

        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=True)

        # Initialize weights and apply final processing
        self.post_init()

    def init_projection(self) -> None:
        config = self.config
        self.in_proj = nn.Conv2d(
            in_channels=config.num_channel,
            out_channels=config.hidden_size,
            kernel_size=(config.patch_size, config.patch_size * config.patch_len),
            stride=config.patch_size * config.patch_len,
            bias=False
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        # map the input_embeds into vectors saperated by patches
        inputs_embeds = self.in_proj(inputs_embeds)
        inputs_embeds = inputs_embeds.flatten(2).transpose(1, 2)
        eos_index = attention_mask.sum(dim=1, dtype=torch.int) - 1

        # >>>>>>> below are copied from llamaModel
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        seq_length_with_past = seq_length
        past_key_values_length = 0

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            seq_length_with_past = seq_length_with_past + past_key_values_length

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past), dtype=torch.bool, device=inputs_embeds.device
            )
            padding_mask = None
        else:
            if 0 in attention_mask:
                padding_mask = attention_mask
            else:
                padding_mask = None

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer), hidden_states, attention_mask, position_ids
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    padding_mask=padding_mask,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        # <<< before are copied from Llama

        eos_logits = hidden_states[torch.arange(hidden_states.shape[0], dtype=torch.int), eos_index]
        logits = self.score(eos_logits)


        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=logits, 
            hidden_states=hidden_states,
            past_key_values=next_cache
        )


class LatentLlamaForSequenceClassification(LatentModel):
    
    def load_backbone(self, path: str | PathLike, num_latent_channel: int, latent_patch_size: int, patch_len: int, num_labels: int, binary: bool) -> nn.Module:
        Llama_config = LlamaConfig.from_pretrained(path)
        
        setattr(Llama_config, 'num_channel', num_latent_channel)
        setattr(Llama_config, 'patch_size', latent_patch_size)
        setattr(Llama_config, 'patch_len', patch_len)
        setattr(Llama_config, 'binary', binary)
        setattr(Llama_config, 'num_labels', num_labels)
        
        Llama: LlamaForPatchSequenceClassification = LlamaForPatchSequenceClassification.from_pretrained(path, config=Llama_config, ignore_mismatched_sizes=True)
        self.backbone = Llama

        del self.backbone.embed_tokens
        self.backbone.embed_tokens = None

        if self.compressor is not None:
            del self.compressor.decoder
            self.compressor.decoder = None

        if num_labels > 1:
            self.loss_fn = nn.CrossEntropyLoss(reduction='mean')
        elif num_labels == 1:
            self.loss_fn = nn.MSELoss(reduction='mean')
        
        return Llama
    
    def save_backbone(self, path: str | PathLike) -> None:
        if isinstance(self.backbone, LlamaForPatchCausalInference):
            self.backbone.save_pretrained(path)
        elif isinstance(self.backbone, DistributedDataParallel):
            self.backbone.module.save_pretrained(path)
        else:
            raise NotImplementedError(f'Saving for {type(self.backbone)} has not been implemented!')
        
        print(f'Llama backbone saved to {path}')
            
    def latent_forward(self, img: TGraph) -> TGraph:
        target = img.value.float()     # target should within range [0, 1]
        if self.compressor is None:
            img_values = target * 2 - 1 # map values from [0, 1] range to [-1, 1] range
        else:
            img_values = target
        output: BaseModelOutputWithPastAndCrossAttentions = self.backbone.forward(
            attention_mask=img.attention_mask,
            inputs_embeds=img_values
        )
        pred = output.last_hidden_state
        result = TGraph()
        result._value = pred
        result.loss = self.forward_loss(pred, img.labels, None)

        return result

    def forward_loss(self, pred: torch.Tensor, target: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.loss_fn.forward(pred, target=target)
        
    def get_connection_layers(self) -> nn.Module:
        return nn.ModuleList([
            self.backbone.in_proj,
        ])
    
    def init_connection_layers(self) -> None:
        print('init the connection layers')
        self.backbone.init_projection()
        return
    
    def delete_unused_layers(self) -> None:
        if self.compressor is not None:
            print('delete the decoder')
            del self.compressor.decoder
            self.compressor.decoder = None
        return

def cache_at(cache: tuple, idx: int) -> tuple | None:
    idx += 1    # shift the index
    target = []
    for c1, c2 in cache:
        target.append((c1[:, :, :idx, :], c2[:, :, :idx, :]))
    return tuple(target)

class LlamaDiscriminator(nn.Module):
    
    def __init__(self, backbone: LlamaForPatchCausalInference) -> None:
        super().__init__()
        self.llama = deepcopy(backbone)
        config = self.llama.config
        del self.llama.out_proj
        
        self.head = nn.Linear(config.hidden_size, 2)
        self.config = config
        
    def set_trainable_layers(self, num: int):
        for param in self.llama.parameters():
            param.requires_grad = False
        print(f'Only allow the last {num} layers in discriminator be trainable')
        for layer in self.llama.layers[-num:]:
            layer: LlamaDecoderLayer
            for param in layer.parameters():
                param.requires_grad = True
        
        for param in self.llama.norm.parameters():
            param.requires_grad = True
        
    def forward_loss(self, logits: torch.Tensor, mask: torch.Tensor, target: int) -> tuple[float, torch.Tensor]:
        """
        logits: [n_batch, n_patch, 2]
        mask:   [n_batch, n_patch]
        target: 1 or 0
        return acc and loss
        """
        lossfn = nn.CrossEntropyLoss(reduction='none')

        logits = logits.flatten(0, 1)
        mask = mask.flatten()
        target = torch.ones_like(mask, dtype=torch.long) * target
        preds = logits.argmax(-1)
                
        # Average the loss per patch
        loss = (lossfn(logits, target) * mask).sum() / (mask.sum() + 1e-8)
        
        # calculate acc
        acc = ((preds == target) * mask).sum() / (mask.sum() + 1e-8)
        
        return acc, loss
        
    def forward(self, real: TGraph, gen: TGraph, target: int, num_samples: int, only_real: bool=False) -> tuple[float, torch.Tensor]:
        """
        Target is 0 or 1, 1 means real, 0 means fake
        """
        # calculate the kv cache
        real_images = real.value * 2.0 - 1
        if real_images.device != self.llama.in_proj.weight.device:
            real_images = real_images.to(device=self.llama.in_proj.weight.device)
        attn_mask = real.attention_mask
        
        # only one pass when input is real
        if target == 1 and only_real:
            out = self.llama.forward(inputs_embeds=real_images, output_hidden_states=True, ignore_out_proj=True)
            logits = self.head.forward(out.hidden_states[-1])
            acc, loss = self.forward_loss(logits, attn_mask, 1)
            return acc, loss
        
        cache = self.llama.forward(inputs_embeds=real_images, attention_mask=attn_mask, ignore_out_proj=True).past_key_values
        
        num_patches = real._value.shape[-1] // real.patch_width
        ids = random.sample(range(num_patches), num_samples)
        attn_mask = attn_mask[:, ids]
        
        # calculate logits each fake patch
        seq = []
        for idx in ids:
            patch: torch.Tensor = gen[idx]._value * 2 - 1
            if patch.device != self.llama.in_proj.weight.device:
                patch = patch.float().to(self.llama.in_proj.weight.device)
            kvs = cache_at(cache, idx)
            out = self.llama.forward(inputs_embeds=patch, past_key_values=kvs, output_hidden_states=True, use_cache=True, ignore_out_proj=True)
            logits = out.hidden_states[-1]
            seq.append(logits)
        
        logits = self.head.forward(torch.cat(seq, dim=1).contiguous())
        acc, loss = self.forward_loss(logits, attn_mask, target)
                
        return acc, loss

    def save(self, folder: str | PathLike) -> None:
        self.config.save_pretrained(folder)
        path = os.path.join(folder, 'llama_discriminator.bin')
        torch.save(self.state_dict(), path)
