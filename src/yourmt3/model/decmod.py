import copy
from typing import Optional, Tuple, Dict, Union
import torch
from torch import nn
from einops import rearrange
from transformers.utils import (
    is_torch_flex_attn_available,
    logging,
    is_torch_fx_proxy,
)
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import (BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions)
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
if is_torch_flex_attn_available():
    from torch.nn.attention.flex_attention import BlockMask

    from transformers.integrations.flex_attention import make_flex_block_causal_mask

from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache

from yourmt3.model.flashdec_helper import (
    YMT3DecoderConfig,
    YMT3DecoderPreTrainedModel,
    RMSNorm,
    gMLP,
    Attention,
    RotaryEmbedding,
)
from yourmt3.model.positional_encoding import FixedSinusoidalPositionalEmbedding

logger = logging.get_logger(__name__)


class YMT3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()

        self.pre_self_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Attention(config=config, layer_idx=layer_idx, is_causal=True)
        self.pre_cross_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.cross_attn = Attention(config=config, layer_idx=layer_idx, is_causal=False)# set causal flag to false
        self.pre_mlp_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = gMLP(config=config)
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states,
        cross_attention_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        cache_position=None,
        **kwargs: Unpack[FlashAttentionKwargs],
        ):
        
        residual = hidden_states
        #self attn residual operation:
        hidden_states = self.pre_self_norm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            cross_attention_states=None, #none for self attneiton
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )                                        
        hidden_states = residual + nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        residual = hidden_states
        #cross attn residual operation:
        hidden_states = self.pre_cross_norm(hidden_states)
        hidden_states, cross_attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            cross_attention_states=cross_attention_states,
            position_embeddings=None, # No rotary emb for cross attention
            attention_mask=None, # No masking for cross attention
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        residual = hidden_states
        #mlp residual opertation
        hidden_states = self.pre_mlp_norm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        hidden_states = residual + nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)


        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs

class YMT3DecoderStack(YMT3DecoderPreTrainedModel):
    """
    T5Stack, modified for YMT3 with:
    - absolute sinusoidal absolute positional encoding
    """

    def __init__(
        self,
        config,
    ):
        super().__init__(config)
        self.additive_pe = None
        self.rotary_emb = None
        if config.position_encoding_type == "sinusoidal":
            print("USING ABS EMBEDDING")
            self.additive_pe = FixedSinusoidalPositionalEmbedding(config.num_max_positions,
                                                                  embedding_dim=config.hidden_size)
        else:
            print("USING ROTARY EMBEDDING")
            self.rotary_emb = RotaryEmbedding(config)

        self.block = nn.ModuleList(
                [YMT3DecoderLayer(config, layer_idx=i) for i in range(config.num_layers)])

        self.final_layer_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.dropout = config.dropout

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
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        position_ids=None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (output_hidden_states
                                if output_hidden_states is not None else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_"
            raise ValueError(f"You have to specify {err_msg_prefix}inputs_embeds")
        

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
       
        batch_size, seq_length = input_shape
        
        # initialize past_key_values
        return_legacy_cache = False
        return_self_attention_cache = False
        if use_cache or past_key_values is not None:
            if past_key_values is None:
               past_key_values = EncoderDecoderCache(DynamicCache(), DynamicCache())
               
            elif isinstance(past_key_values, Cache) and not isinstance(past_key_values, EncoderDecoderCache):
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

        past_key_values_length = past_key_values.get_seq_length() if past_key_values is not None else 0
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values.self_attention_cache if past_key_values is not None else None,
            output_attentions,
        )

        hidden_states = inputs_embeds
        #sinusoidal encoding mod

        if self.additive_pe is not None:
            hidden_states = hidden_states + self.additive_pe(hidden_states.shape[1], past_key_values_length)
            position_embeddings = None

        else:
            position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        for i, layer_module in enumerate(self.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    encoder_hidden_states,
                    position_embeddings,
                    causal_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                    return_dict,
                    cache_position,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    cross_attention_states=encoder_hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=causal_mask,
                    past_key_value=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    return_dict=return_dict,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions += (layer_outputs[1],)
                all_cross_attentions += (layer_outputs[2],)


        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        next_cache = past_key_values if use_cache else None

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

class DecoderYMT3(YMT3DecoderPreTrainedModel):

    def __init__(self, decoder_config: Optional[Dict] = None, config: Optional[YMT3DecoderConfig] = None):
        
        if config is None:
            config = YMT3DecoderConfig()
        if decoder_config is not None:
            config = copy.deepcopy(config)
            config.update(decoder_config)

        super().__init__(config)
        self.model_dim = config.hidden_size
        self.decoder = YMT3DecoderStack(config)

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


class MultiChannelDecoderYMT3(YMT3DecoderPreTrainedModel):

    def __init__(self, decoder_config: Optional[Dict] = None, config: Optional[YMT3DecoderConfig] = None):
        if config is None:
            config = YMT3DecoderConfig()
        if decoder_config is not None:
            config = copy.deepcopy(config)
            config.update(decoder_config)

        super().__init__(config)
        self.model_dim = config.hidden_size
        self.decoder = YMT3DecoderStack(config)
        # Multi-channel parameters
        self.num_channels = config.num_channels
        self.training_chunk_size = config.training_chunk_size

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
        # Reshape input_embeds and encoder_hidden_states
        b, k, t, d = inputs_embeds.size()
        #print("BATCH SIZE:", b)
        # Temporary fix for memory issue, process channels in chunks:
        # --- Conditional Processing ---
        if self.training:
            # --- Training Mode: Process channels in chunks ---
            output_chunks = []
            for i in range(0, k, self.training_chunk_size):
                k_start = i
                k_end = min(i + self.training_chunk_size, k)
                current_k = k_end - k_start

                # Slice inputs for the current chunk
                chunk_inputs_embeds = inputs_embeds[:, k_start:k_end, :, :]
                chunk_encoder_hidden_states = encoder_hidden_states[:, k_start:k_end, :, :]

                # Reshape for the decoder stack's expected batch dimension (B * current_k, T, D)
                chunk_inputs_embeds = rearrange(chunk_inputs_embeds, 'b ck t d -> (b ck) t d', ck=current_k)
                chunk_encoder_hidden_states = rearrange(chunk_encoder_hidden_states, 'b ck t d -> (b ck) t d', ck=current_k)

                # Call the core decoder stack for the chunk
                # Explicitly disable cache/optional outputs for training chunks for simplicity
                chunk_decoder_outputs = self.decoder(
                    inputs_embeds=chunk_inputs_embeds,
                    attention_mask=None, # Use potentially adapted mask
                    encoder_hidden_states=chunk_encoder_hidden_states,
                    past_key_values=None, # Ignore past_key_values during training chunking
                    use_cache=False, # Disable cache during training chunking
                    output_attentions=False, # Disable attentions during training chunking
                    output_hidden_states=False, # Disable hidden states during training chunking
                    return_dict=True, # Easier to get last_hidden_state
                    cache_position=None, # Ignore cache_position during training chunking
                )

                # Reshape chunk output back to (B, current_k, T, D)
                last_hidden_state_chunk = rearrange(
                    chunk_decoder_outputs.last_hidden_state,
                    '(b ck) t d -> b ck t d', b=b, ck=current_k
                )
                output_chunks.append(last_hidden_state_chunk)

            # Concatenate results from all chunks along the K dimension
            final_last_hidden_state = torch.cat(output_chunks, dim=1).contiguous() # Shape: (B, K, T, D)

            # --- Construct Output ---
            if not return_dict:
                # Return only last_hidden_state as tuple, others are None/ignored in this path
                return (final_last_hidden_state,)
            else:
                # Return object with only last_hidden_state populated
                return BaseModelOutputWithPastAndCrossAttentions(
                    last_hidden_state=final_last_hidden_state,
                    past_key_values=None,
                    hidden_states=None,
                    attentions=None,
                    cross_attentions=None,
                )
        else:
            inputs_embeds = rearrange(inputs_embeds, 'b k t d -> (b k) t d')
            encoder_hidden_states = rearrange(encoder_hidden_states, 'b k t d -> (b k) t d')

            # K-channel Decoding
            decoder_outputs = self.decoder(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
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


