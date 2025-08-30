import torch
from torch import nn
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Union

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding that computes and caches cos/sin values on-demand
    for each sequence length encountered per device.
    Returns cos/sin separately, compatible with Llama apply_rotary_pos_emb.
    """
    def __init__(self, dim: int, base: int = 10000):
        """
        Args:
            dim (int): The feature dimension (head dimension) for RoPE.
            base (int): The base frequency for RoPE. Defaults to 10000.
        """
        super().__init__()
        self.dim = dim
        self.base = base

        # Calculate inverse frequencies (theta_i) only once
        # Use float32 for stability in calculation, register as buffer
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Cache to store computed cos/sin values, keyed by (seq_len, device)
        # Values will be tuples (cos_cached, sin_cached)
        self._cos_sin_cache: Dict[Tuple[int, torch.device], Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_cos_sin(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin values for a given sequence length and device, ensuring float32 precision."""

        # Ensure inv_freq is float32 (it should be from init, but explicit check is fine)
        inv_freq_float = self.inv_freq.float()

        # Generate positions [0, 1, ..., seq_len-1] as float32 on the correct device
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
        # Disable autocast for this block to ensure calculations are in float32
        with torch.autocast(device_type=device_type, enabled=False):
            # Calculate frequency grids (m * theta_i) in float32
            # Outer product: [seq_len] x [dim/2] -> [seq_len, dim/2]
            freqs = torch.einsum("i,j->ij", t, inv_freq_float)

            # Duplicate frequencies for sine and cosine components
            # [seq_len, dim/2] -> [seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # Calculate cos and sin in float32
            # Result shape: [seq_len, dim]
            cos_cached = emb.cos()
            sin_cached = emb.sin()

        return cos_cached, sin_cached # Return float32 tensors

    def forward(self, x, bsz) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates or retrieves cached RoPE embeddings for the input tensor's sequence length.

        Args:
            x (`torch.Tensor`): Input tensor. Used to determine bsz, seq_len, device, and dtype.
                               Shape typically [bsz, seq_len, ...], but only these dimensions matter.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: cos and sin tensors.
                                               Shape: [bsz, seq_len, dim]
        """
        _, seq_len, *_ = x.shape # Get batch size and sequence length
        device = x.device
        dtype = x.dtype

        cache_key = (seq_len, device)

        # Check cache first
        if cache_key not in self._cos_sin_cache:
            # Compute and cache if not found
            cos_cached, sin_cached = self._compute_cos_sin(seq_len, device)
            self._cos_sin_cache[cache_key] = (cos_cached, sin_cached)
        else:
            # Retrieve from cache
            cos_cached, sin_cached = self._cos_sin_cache[cache_key]

        # --- Prepare output ---
        # Expand batch dimension and cast to the input tensor's dtype

        # Expand shape: [seq_len, dim] -> [1, seq_len, dim] -> [bsz, seq_len, dim]
        cos = cos_cached.unsqueeze(0).expand(bsz, -1, -1).to(dtype)
        sin = sin_cached.unsqueeze(0).expand(bsz, -1, -1).to(dtype)

        return cos, sin

class RMSNormK(nn.Module):
    """
    RMS Normalization with K independent, learnable scaling factors (weights).

    Assumes input is flattened along the Batch (B) and View (K) dimensions.
    """
    def __init__(self, dim: int, k: int = 13, eps: float = 1e-6, is_4d = False, bias=False):
        """
        Args:
            k (int): Number of independent views/parameter sets.
            dim (int): Feature dimension (H).
            eps (float): Epsilon value for numerical stability in RMSNorm.
        """
        super().__init__()

        self.k = k
        self.dim = dim
        self.eps = eps
        self.is_4d = is_4d
        # Learnable weights for K views, stored together
        self.weight = nn.Parameter(torch.ones(k, dim))
        self.bias = nn.Parameter(torch.zeros(k, dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization with view-specific scaling weights.

        Args:
            x (torch.Tensor): Input tensor of shape [B*K, T, H] or [B*K, N, T, H].

        Returns:
            torch.Tensor: Normalized tensor of shape [B*K, T, H]/ [B*K, N, T, H].
        """
        input_dtype = x.dtype
        original_shape = x.shape

        if self.is_4d:
            B_times_K, N, T, D_h = original_shape
            x = x.reshape(B_times_K, N * T, D_h)
        
        B_times_K = x.shape[0]

        B = B_times_K // self.k

        # --- Shared RMS calculation (done in float32 for stability) ---
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True) # Shape: [B*K, T, 1]
        x = x * torch.rsqrt(variance + self.eps) # Shape: [B*K, T, H]

        # --- Apply K independent weights via broadcasting ---
        # Reshape weights: [K, H] -> [B*K, H] -> [B*K, 1, H]
        # Repeat weights B times to align with the flattened B*K dimension
        weights_repeated = self.weight.repeat(B, 1).unsqueeze(1)
        # Apply view-specific scaling
        x = weights_repeated * x # Shape: [B*K, T, H]

        if self.bias is not None:
            bias_repeated = self.bias.repeat(B, 1).unsqueeze(1)
            x = x + bias_repeated
        
        if self.is_4d:
            x = x.reshape(original_shape) 

        return x.to(input_dtype)

class gMLP(nn.Module):
    def __init__(self, hidden_size, bias=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.gate_proj = nn.Linear(hidden_size, 2*hidden_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, 2*hidden_size, bias=bias)
        self.act_fn = nn.SiLU()
        self.down_proj = nn.Linear(2*hidden_size, hidden_size, bias=bias)

    def forward(self, hidden_states):
        
        hidden_act = self.act_fn(self.gate_proj(hidden_states))
        hidden_linear = self.up_proj(hidden_states)
        hidden_states = hidden_act * hidden_linear
        hidden_states = self.down_proj(hidden_states)

        return hidden_states



class SelfAttention(nn.Module):
    """Multi-headed self-attention from 'Attention Is All You Need' paper"""

    def __init__(self, hidden_size, num_heads = 8, attention_dropout=0.05, k=13, is_causal = False):
        super().__init__()
        self.k = k
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = attention_dropout
        self.is_causal = is_causal
        
        self.q_norm = RMSNormK(self.head_dim, k=self.k, is_4d=True, bias=True)
        self.k_norm = RMSNormK(self.head_dim, k=self.k, is_4d=True, bias=True)
        
        self.q_proj = nn.Linear(
            hidden_size, num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, num_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            hidden_size, num_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            num_heads * self.head_dim, hidden_size, bias=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings,
        attention_mask: Optional[torch.Tensor]= None,
        output_attentions = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)
        
        #print("SHAPE Q", query_states.shape)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # SDPA with memory-efficient backend is bugged with non-contiguous inputs and custom attn_mask for some torch versions
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()


        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            scale=self.scaling,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal = self.is_causal)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)
        
        attn_output = self.o_proj(attn_output)
        return attn_output

class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8, k=13):
        """
        Args:
            enc_hidden_size (int): Hidden dimension of the input encoder states.
            dec_hidden_size (int): Hidden dimension of the adapter/decoder states (output of MLPs).
            k (int): Number of independent views to generate.
            emb_size (int): Dimension of the learnable view embeddings.
            n_iter (int): Number of recurrent iterations.
        """
        super().__init__()
        self.k = k
        self.hidden_size = hidden_size
        self.self_attn = SelfAttention(hidden_size=hidden_size, num_heads=num_heads, k=k, attention_dropout=0.0)
        self.mlp = gMLP(hidden_size=hidden_size)
        self.input_layernorm = RMSNormK(hidden_size, is_4d=False)
        self.post_attention_layernorm = RMSNormK(hidden_size, is_4d=False)
        self.pre_feedforward_layernorm = RMSNormK(hidden_size, is_4d=False)
        self.post_feedforward_layernorm = RMSNormK(hidden_size, is_4d=False)

    def forward(
        self, 
        hidden_states,
        position_embeddings,
        attention_mask,
        ):

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        hidden_states = residual + hidden_states

        residual = hidden_states

        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class MusicFMAdapter(nn.Module):
    """
    Adapter using k independent recurrent views processed by SHARED BottleneckGatedMLPs
    iteratively, with INDEPENDENT Layer Normalization parameters per view.
    Includes Pre-LN within MLP blocks and a final Post-LN for the state update.
    """
    def __init__(self, enc_hidden_size, dec_hidden_size, num_heads=8, k=13, emb_size=512, num_layers=3, n_iter=4):
        """
        Args:
            enc_hidden_size (int): Hidden dimension of the input encoder states.
            dec_hidden_size (int): Hidden dimension of the adapter/decoder states (output of MLPs).
            k (int): Number of independent views to generate.
            emb_size (int): Dimension of the learnable view embeddings.
            n_iter (int): Number of recurrent iterations.
        """
        super().__init__()
        self.k = k
        self.n_iter = n_iter
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.k, emb_size)
        self.embedding_proj = nn.Linear(emb_size, self.dec_hidden_size, bias=False)
        self.adapter = nn.Linear(self.enc_hidden_size + self.dec_hidden_size, self.dec_hidden_size, bias=False)

        self.in_norm = RMSNormK(dec_hidden_size, is_4d=False)
        self.out_norm = RMSNormK(dec_hidden_size, is_4d=False)
        
        self.rotary_emb = RotaryEmbedding(dim=dec_hidden_size//num_heads)
        self.adapter_layers = nn.ModuleList(
            [AdapterLayer(self.dec_hidden_size, k=self.k, num_heads=num_heads) for layer_idx in range(num_layers)]
        )
    def forward(self, *args):
        if self.training is True:
            # Pass the collected positional arguments along
            return self._forward_compile(*args)
        else:
            return self._forward_no_compile(*args)

    # This definition also accepts any positional arguments via *args
    def _forward_no_compile(self, *args):
        # Pass them along again
        return self._forward(*args)

    @torch.compile
    # Same pattern here
    def _forward_compile(self, *args):
        # Pass them along again
        return self._forward(*args)

    def _forward(self, hidden_states, attention_mask=None):
        """
        Args:
            hidden_states (torch.Tensor): Input tensor from encoder.
                Shape: [batch_size, seq_len, enc_hidden_size]

        Returns:
            torch.Tensor: The final states for the k views after n_iter iterations.
                Shape: [batch_size, k, seq_len, dec_hidden_size]
        """
        B, T, H_enc = hidden_states.shape
        device = hidden_states.device

        position_embeddings = self.rotary_emb(hidden_states, bsz =B*self.k)

        # --- Initialize recurrent_states ---
        initial_embeddings = self.embedding(torch.arange(self.k, device=device))
        recurrent_states = self.embedding_proj(initial_embeddings)
        recurrent_states = recurrent_states.unsqueeze(0).unsqueeze(2) # Shape: [1, k, 1, H_dec]
        recurrent_states = recurrent_states.expand(B, -1, T, -1) # Shape: [B, k, T, H_dec]

        # --- Pre-expand original hidden_states (optimization) ---
        expanded_hidden_states = hidden_states.unsqueeze(1).expand(-1, self.k, -1, -1)
        # Shape: [B, k, T, H_enc]
        
        # --- Iterative Processing ---
        for _ in range(self.n_iter):
            # 1. Prepare & Concatenate
            concat_input = torch.cat((expanded_hidden_states, recurrent_states), dim=-1)
            # Shape: [B, k, T, H_enc + H_dec]

            # 2. Reshape for processing (flatten B and k)
            B_orig, K_orig, T_orig, H_concat_dim = concat_input.shape

            adapter_input = concat_input.view(B_orig * K_orig, T_orig, H_concat_dim)
            # Shape: [B*k, T, H_enc + H_dec]

            # 3. Adapter projection (Shared) -> Input to transformer blocks
            recurrent_states = self.adapter(adapter_input)
            # Shape: [B*k, T, H_dec]
            recurrent_states = self.in_norm(recurrent_states)

            for layer in self.adapter_layers:
                recurrent_states = layer(
                    hidden_states=recurrent_states,
                    position_embeddings = position_embeddings,
                    attention_mask = attention_mask)

            recurrent_states = self.out_norm(recurrent_states)
            # 6. Update the recurrent state 's' with the FINAL NORMALIZED output
            recurrent_states = recurrent_states.view(B_orig, K_orig, T_orig, self.dec_hidden_size)
            # Shape: [B, k, T, H_dec]

        # --- Return Final State ---
        return recurrent_states

class TemporalUpsample(nn.Module):
    def __init__(self, 
                 input_dim=1024, 
                 output_dim=1024,
                 upsample_factor = 4):
        super(TemporalUpsample, self).__init__()
        
        # First upsampling:
        # T -> 2T with kernel_size=5
        # We'll set padding=2, output_padding=1 so it exactly doubles.
        self.upsample_factor = upsample_factor
        self.upsample1 = nn.ConvTranspose1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=5,
            stride=2,
            padding=2,
            output_padding=1
        )

        if self.upsample_factor == 4:
            # Second upsampling:
            # 2T -> 4T with kernel_size=3
            self.upsample2 = nn.ConvTranspose1d(
                in_channels=output_dim,
                out_channels=output_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )

    def forward(self, *args):
        if self.training is True:
            # Pass the collected positional arguments along
            return self._forward_compile(*args)
        else:
            return self._forward_no_compile(*args)

    # This definition also accepts any positional arguments via *args
    def _forward_no_compile(self, *args):
        # Pass them along again
        return self._forward(*args)

    @torch.compile
    # Same pattern here
    def _forward_compile(self, *args):
        # Pass them along again
        return self._forward(*args)

    def _forward(self, x):
        """
        x shape: [B, T, H] or [B, K, T, H]
        output:  [B, 4T, output_dim] or [B, K, 4T, output_dim]
        """
        is_k_present = (x.dim() == 4)  # Check if input has K dimension
        
        if is_k_present:
            B, K, T, H = x.shape
            x = x.view(B * K, T, H)  # Merge K into batch dimension

        # Switch to [B, H, T] for ConvTranspose1d
        x = x.transpose(1, 2)  # [B, input_dim, T]

        # 1st upsampling: T -> 2T
        x = self.upsample1(x)  # [B, output_dim, 2T]

        if self.upsample_factor == 4:
            # 2nd upsampling: 2T -> 4T
            x = self.upsample2(x)  # [B, output_dim, 4T]

        # Switch back to [B, 4T, output_dim]
        x = x.transpose(1, 2)

        if is_k_present:
            x = x.view(B, K, x.shape[1], x.shape[2])  # Restore original shape with K

        return x


def get_feat_length(input_frames, upsampling_factor = 4):
    def normal_round(n):
        if n - math.floor(n) < 0.5:
            return math.floor(n)
        return math.ceil(n)
    # Resample
    resampled_frames = input_frames * 24000 / 16000
    
    # Spectrogram temporal frames (with padding)
    spec_frames = resampled_frames / 240

    # Downsampling
    final_seq_len = normal_round(spec_frames / 4) * upsampling_factor

    return final_seq_len 

