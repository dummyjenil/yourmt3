# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
# ==============================================================================
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
from typing import Optional, Tuple, Union, Dict
from einops import rearrange
from yourmt3.model.ops import count_parameters

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.utils import logging
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.modeling_t5 import (T5LayerNorm, T5LayerSelfAttention, T5LayerCrossAttention, T5LayerFF, Cache, DynamicCache, EncoderDecoderCache, StaticCache, is_torchdynamo_compiling)
from transformers.modeling_outputs import (BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions)
from transformers import T5Config, T5PreTrainedModel
from yourmt3.model.positional_encoding import FixedSinusoidalPositionalEmbedding
from yourmt3.model.ff_layer import get_ff_layer

logger = logging.get_logger(__name__)


class T5BlockYMT3(nn.Module):
    """T5 Block, modified to allow using different types of FF layers."""

    def __init__(self, config, has_relative_attention_bias=False, layer_idx: Optional[int] = None):
        super().__init__()
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer_idx = layer_idx
        self.layer.append(
            T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias, layer_idx=layer_idx))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, layer_idx=layer_idx))

        # FF layer
        if config.ff_layer_type == 't5_gmlp':
            self.layer.append(T5LayerFF(config))
        elif config.ff_layer_type == 'moe':
            config.moe_num_experts = 8
            config.moe_topk = 2
            config.hidden_act = 'silu'
            moe = get_ff_layer(config, input_size=config.d_model, widening_factor=config.ff_widening_factor)
            self.layer.append(moe)
        else:
            raise ValueError(f"Unknown FF layer type: {config.ff_layer_type}.")
        self.ff_layer_type = config.ff_layer_type

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        cache_position=None,
    ):
        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )
        hidden_states, past_key_value = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=past_key_value,
                query_length=cache_position[-1] + 1,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states, past_key_value = cross_attention_outputs[:2]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        if self.ff_layer_type == 't5_gmlp':
            hidden_states = self.layer[-1](hidden_states)
        elif self.ff_layer_type == 'moe':
            hidden_states = hidden_states + self.layer[-1](hidden_states)[0]  # residual connection outside the MoE
        else:
            raise ValueError(f"Unknown FF layer type: {self.ff_layer_type}.")

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (past_key_value,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return outputs  # hidden-states, past_key_value, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)

class T5StackYMT3(T5PreTrainedModel):
    """
    T5Stack, modified for YMT3 with:
    - absolute sinusoidal absolute positional encoding
    """

    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.is_decoder = config.is_decoder

        # Positional encoding (modified)
        self.use_t5_trainable_pe = False
        self.additive_pe = None

        pos_enc_type = getattr(config, 'position_encoding_type', 'sinusoidal')
        if pos_enc_type in ['sinusoidal']:
            self.additive_pe = FixedSinusoidalPositionalEmbedding(config.num_max_positions,
                                                                  embedding_dim=config.d_model)
            self.block = nn.ModuleList(
                [T5BlockYMT3(config, has_relative_attention_bias=False, layer_idx=i) for i in range(config.num_layers)])
        elif pos_enc_type == 'trainable':
            self.use_t5_trainable_pe = True
            # Stack blocks
            self.block = nn.ModuleList(
                [T5BlockYMT3(config, has_relative_attention_bias=bool(i == 0), layer_idx=i) for i in range(config.num_layers)])

        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.gradient_checkpointing = False
        

    def forward(
        self,
        # input_ids=None,
        inputs_embeds=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify {err_msg_prefix}inputs_embeds")
        

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
       
        batch_size, seq_length = input_shape

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(f"`use_cache` can only be set to `True` if {self} is used as a decoder")

        # initialize past_key_values
        return_legacy_cache = False
        return_self_attention_cache = False
        if self.is_decoder and (use_cache or past_key_values is not None):
            if isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
                return_self_attention_cache = True
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())
            elif not isinstance(past_key_values, EncoderDecoderCache):
                return_legacy_cache = True
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.48.0. "
                    "You should pass an instance of `EncoderDecoderCache` instead, e.g. "
                    "`past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = EncoderDecoderCache.from_legacy_cache(past_key_values)
            elif past_key_values is None:
                past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
        elif not self.is_decoder:
            # do not pass cache object down the line for encoder stack
            # it messes indexing later in decoder-stack because cache object is modified in-place
            past_key_values = None

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )
        
        # mod: additive absolute PE (sinusoidal)
        if self.additive_pe is not None:
            inputs_embeds = inputs_embeds + self.additive_pe(inputs_embeds.shape[1], past_key_values_length)
        else:
            pass  # trinable PE is implemented in T5Block

            
        if attention_mask is None and not is_torchdynamo_compiling():
            # required mask seq length can be calculated via length of past cache
            mask_seq_length = past_key_values_length + seq_length
            attention_mask = torch.ones(batch_size, mask_seq_length, device=inputs_embeds.device)

        if self.config.is_decoder:
            causal_mask = self._update_causal_mask(
                attention_mask,
                inputs_embeds,
                cache_position,
                past_key_values.self_attention_cache if past_key_values is not None else None,
                output_attentions,
            )
        elif attention_mask is not None:
            causal_mask = attention_mask[:, None, None, :]
            causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
            causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
        else:
            causal_mask = None

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(cross_attn_head_mask, self.config.num_layers)
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, layer_module in enumerate(self.block):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    causal_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                    return_dict,
                    cache_position,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, next_decoder_cache = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)


        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_self_attention_cache:
            next_cache = past_key_values.self_attention_cache
        if return_legacy_cache:
            next_cache = past_key_values.to_legacy_cache()

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


    # Copied from transformers.models.llama.modeling_llama.LlamaModel._update_causal_mask    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions:
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            min_dtype = torch.finfo(dtype).min
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    # Copied from transformers.models.llama.modeling_llama.LlamaPreTrainedModel._prepare_4d_causal_attention_mask_with_cache_position
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape
                `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache,
                to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

class T5DecoderYMT3(T5PreTrainedModel):

    def __init__(self, decoder_config: Optional[Dict] = None, config: Optional[T5Config] = None):
        if config is None:
            config = T5Config()
        if decoder_config is not None:
            config = copy.deepcopy(config)
            config.update(decoder_config)

        if hasattr(config, "ff_widening_factor"):
            config.d_ff = int(config.d_model) * int(config.ff_widening_factor)

        config.is_decoder = True
        config.is_encoder_decoder = False

        super().__init__(config)
        self.model_dim = config.d_model

        self.decoder = T5StackYMT3(config)

        # Initialize weights and apply final processing
        self.post_init()
    """temporary fix for torch.compile issue"""

    def forward(self, **kwargs):
        if self.training is True:
            return self._forward_compile(**kwargs)
        else:
            return self._forward_no_compile(**kwargs)

    def _forward_no_compile(self, **kwargs):
        return self._forward(**kwargs)

    @torch.compile
    def _forward_compile(self, **kwargs):
        return self._forward(**kwargs)

    def _forward(
        self,
        # input_ids: torch.LongTensor, # removed since embed_tokens is outside the decoder
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,  # decoder_attention_mask
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutputWithPastAndCrossAttentions]:

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if isinstance(encoder_hidden_states, BaseModelOutput):
            encoder_hidden_states = encoder_hidden_states.last_hidden_state

        # Decode
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        if not return_dict:
            return decoder_outputs
        else:
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=decoder_outputs[0],
                past_key_values=decoder_outputs[1],
                hidden_states=decoder_outputs[2] if len(decoder_outputs) > 2 else None,
                attentions=decoder_outputs[3] if len(decoder_outputs) > 3 else None,
                cross_attentions=decoder_outputs[4] if len(decoder_outputs) > 4 else None,
            )


class T5EncoderYMT3(T5PreTrainedModel):
    # _keys_to_ignore_on_load_missing = [r"encoder.embed_tokens.weight"]

    def __init__(self, encoder_config: Optional[Dict] = None, config: Optional[T5Config] = None):
        if config is None:
            config = T5Config()
        if encoder_config is not None:
            config = copy.deepcopy(config)
            config.update(encoder_config)

        if hasattr(config, "ff_widening_factor"):
            config.d_ff = int(config.d_model) * int(config.ff_widening_factor)

        config.is_decoder = False
        config.use_cache = False
        config.is_encoder_decoder = False

        super().__init__(config)
        self.model_dim = config.d_model

        self.encoder = T5StackYMT3(config)

        # Initialize weights and apply final processing
        self.post_init()

    """temporary fix for torch.compile issue"""

    def forward(self, **kwargs):
        if self.training is True:
            return self._forward_compile(**kwargs)
        else:
            return self._forward_no_compile(**kwargs)

    def _forward_no_compile(self, **kwargs):
        return self._forward(**kwargs)

    @torch.compile
    def _forward_compile(self, **kwargs):
        return self._forward(**kwargs)

    def _forward(
        self,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Encode
        encoder_outputs = self.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return encoder_outputs
        else:
            return BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )



class MultiChannelT5Decoder(T5PreTrainedModel):

    def __init__(self, decoder_config: Optional[Dict] = None, config: Optional[T5Config] = None):
        if config is None:
            config = T5Config()
        if decoder_config is not None:
            config = copy.deepcopy(config)
            config.update(decoder_config)

        if hasattr(config, "ff_widening_factor"):
            config.d_ff = int(config.d_model) * int(config.ff_widening_factor)

        config.is_decoder = True
        config.is_encoder_decoder = False

        super().__init__(config)
        self.model_dim = config.d_model
        self.decoder = T5StackYMT3(config)

        # Multi-channel parameters
        self.num_channels = config.num_channels

        # Initialize weights and apply final processing
        self.post_init()

    """temporary fix for torch.compile issue"""

    def forward(self, **kwargs):
        if self.training is True:
            return self._forward_compile(**kwargs)
        else:
            return self._forward_no_compile(**kwargs)

    def _forward_no_compile(self, **kwargs):
        return self._forward(**kwargs)

    @torch.compile
    def _forward_compile(self, **kwargs):
        return self._forward(**kwargs)

    def _forward(
        self,
        # input_ids: torch.LongTensor, # removed since embed_tokens is outside the decoder
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,  # decoder_attention_mask
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutputWithPastAndCrossAttentions]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        """
        Args:
            inputs_embeds: torch.FloatTensor (B, K, T, D), where K is the number of channels
            encoder_hidden_states: torch.FloatTensor (B, K, T, D), where K is the number of channels
        
        Returns:
            decoder_outputs: BaseModelOutputWithPastAndCrossAttentions
                last_hidden_state: torch.FloatTensor (B, K, T, D), where K is the number of channels
                past_key_values: Tuple[Tuple[torch.Tensor]]
                hidden_states: Tuple[torch.FloatTensor]
                attentions: Tuple[torch.FloatTensor]
                cross_attentions: Tuple[torch.FloatTensor]

        """
        if isinstance(encoder_hidden_states, BaseModelOutput):
            encoder_hidden_states = encoder_hidden_states.last_hidden_state
        #print('shape inputs embed', inputs_embeds.shape)
        # print('encoder inputs embed', encoder_hidden_states.shape)
        # Reshape inputs_embeds and encoder_hidden_states
        b, k, t, d = inputs_embeds.size()
        inputs_embeds = rearrange(inputs_embeds, 'b k t d -> (b k) t d')
        encoder_hidden_states = rearrange(encoder_hidden_states, 'b k t d -> (b k) t d')

        # K-channel Decoding
        decoder_outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        # Reshape decoder_outputs
        decoder_outputs['last_hidden_state'] = rearrange(decoder_outputs['last_hidden_state'],
                                                         '(b k) t d -> b k t d',
                                                         b=b,
                                                         k=k)

        if not return_dict:
            # Collecting values from decoder_outputs in a specific order
            outputs = (
                decoder_outputs['last_hidden_state'],
                decoder_outputs.get('past_key_values', None),
                decoder_outputs.get('hidden_states', None),
                decoder_outputs.get('attentions', None),
                decoder_outputs.get('cross_attentions', None),
            )
            return tuple(v for v in outputs if v is not None)
        else:
            return decoder_outputs  # ['last_hidden_state']: (B, K, T, D)
