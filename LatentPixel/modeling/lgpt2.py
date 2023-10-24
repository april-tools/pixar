import os
from os import PathLike
from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import GPT2Model, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers import logging
from diffusers import AutoencoderKL
from torch.nn.parallel import DistributedDataParallel

from LatentPixel.text_graph import TGraph

from .latent_model import LatentModel
from ..utils import mask2img


logger = logging.get_logger(__name__)


class GPT2ForPatchCausalInference(GPT2Model):
    
    def __init__(self, config: GPT2Config):
        super().__init__(config)

        self.embed_dim = config.hidden_size

        # self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.wpe = nn.Embedding(config.max_position_embeddings, self.embed_dim)

        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([GPT2Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        self.gradient_checkpointing = False
        
        self.config = config
        self.init_projection()

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
        self.out_proj = nn.ConvTranspose2d(
            self.embed_dim, 
            config.num_channel, 
            (config.patch_size, config.patch_size * config.patch_len),
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

        # >>>>>>> below are copied from GPT2Model
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past_key_values is None:
            past_length = 0
            past_key_values = tuple([None] * len(self.h))
        else:
            past_length = past_key_values[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # GPT2Attention mask.
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure layer_past is on same device as hidden_states (might not be correct)
                if layer_past is not None:
                    layer_past = tuple(past_state.to(hidden_states.device) for past_state in layer_past)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, use_cache, output_attentions)

                    return custom_forward

                outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            hidden_states = outputs[0]
            if use_cache is True:
                presents = presents + (outputs[1],)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2 if use_cache else 1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3 if use_cache else 2],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )
            
        # <<< before are copied from GPT2
        hidden_states = hidden_states.transpose(1, 2).unsqueeze(2)
        hidden_states = self.out_proj(hidden_states)

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
        
class LatentGPT2(LatentModel):
        
    def load_backbone(self, path: str | PathLike, num_latent_channel: int, latent_patch_size: int, patch_len: int, num_labels: int, binary: bool) -> nn.Module:
        gpt2_config = GPT2Config.from_pretrained(path)
        
        setattr(gpt2_config, 'num_channel', num_latent_channel)
        setattr(gpt2_config, 'patch_size', latent_patch_size)
        setattr(gpt2_config, 'patch_len', patch_len)
        setattr(gpt2_config, 'binary', binary)
        
        gpt2: GPT2ForPatchCausalInference = GPT2ForPatchCausalInference.from_pretrained(path, config=gpt2_config, ignore_mismatched_sizes=True)
        
        return gpt2
    
    def save_backbone(self, path: str | PathLike) -> None:
        if isinstance(self.backbone, GPT2ForPatchCausalInference):
            self.backbone.save_pretrained(path)
        elif isinstance(self.backbone, DistributedDataParallel):
            self.backbone.module.save_pretrained(path)
        else:
            raise NotImplementedError(f'Saving for {type(self.backbone)} has not been implemented!')
        
        print(f'gpt2 backbone saved to {path}')
            
    def latent_forward(self, img: TGraph) -> TGraph:
        target = img._value.float()     # target should within range [0, 1]
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
        mask = mask.reshape(-1).contiguous()

        if self.binary and self.compressor is None:
            loss = nn.BCEWithLogitsLoss(reduction='none').forward(pred.reshape(-1).contiguous(), target.reshape(-1).contiguous()) * mask
            loss = loss.reshape(bs, c * h * w)
            mask = mask.reshape(bs, c * h * w)
            print(loss.shape)
            loss = loss.sum(dim=1) / mask.sum(dim=1)
            print(loss.shape)
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
        if self.backbone.wte is not None:
            print('Delete the embedding layer')
            del self.backbone.wte
            self.backbone.wte = None
        if self.compressor is None:
            print('No unused layers to delete')
            return
        print('delete the decoder')
        del self.compressor.decoder
        self.compressor.decoder = None

    def autoregressive_generate(self, prompt: TGraph, gen_idx: int, num_new_patches: int) -> TGraph:
        if self.compressor is not None:
            encoded = self.encode(prompt)
        else:
            encoded = prompt
            
        for idx in range(num_new_patches):
            print(f'generate the {idx} th patch')
            generated = self.latent_forward(encoded)
            encoded._value[..., gen_idx * self.latent_patch_size: (gen_idx + 1) * self.latent_patch_size] = generated._value[..., (gen_idx - 1) * self.latent_patch_size: gen_idx * self.latent_patch_size] 
            gen_idx += 1
            if gen_idx > 528:
                break
        
        return encoded
