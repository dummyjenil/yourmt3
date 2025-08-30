import torch
from torch import nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yourmt3.model.musicfm.modules.flash_conformer import Wav2Vec2ConformerEncoderLayer, Wav2Vec2ConformerConfig, Wav2Vec2ConformerRotaryPositionalEmbedding

class Stemifier(nn.Module):
    def __init__(self, num_iter=5):
        """
        A refinement block based on two stacked Wav2Vec2-Conformer encoder layers,
        applied iteratively. It assumes that the input is a sequence with a prepended
        conditioning token. After processing, the conditioning token is removed.
        """
        super(Stemifier, self).__init__()
        # Load a pretrained configuration.
        config = Wav2Vec2ConformerConfig.from_pretrained("facebook/wav2vec2-conformer-rope-large-960h-ft")
        config.is_causal = False
        config.position_embeddings_type = "rotary"
        # We create two encoder layers and will apply them repeatedly.
        self.embed_positions = Wav2Vec2ConformerRotaryPositionalEmbedding(config)
        self.layer1 = Wav2Vec2ConformerEncoderLayer(config)
        self.layer2 = Wav2Vec2ConformerEncoderLayer(config)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.num_iter = num_iter

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T+1, embed_dim). The first token in the sequence is the
               conditioning token (e.g. the learnable stem token).
               
        Returns:
            A tensor of shape (B, T, embed_dim) where the conditioning token has been removed.
        """

        relative_position_embeddings = self.embed_positions(x)

        for _ in range(self.num_iter):
            # Each conformer layer returns a tuple: (hidden_states, attn_weights).
            # We only care about the updated hidden_states.
            x, _ = self.layer1(x, relative_position_embeddings=relative_position_embeddings)
            x, _ = self.layer2(x, relative_position_embeddings=relative_position_embeddings)
        # Remove the conditioning token (first time-step) so that the output length is T.
        out = x[:, 1:, :]
        out = self.layer_norm(out)
        return out
        
class StemSeparator(nn.Module):
    def __init__(self, K=13, embed_dim=1024):
        """
        Args:
            K (int): Number of learnable tokens.
            embed_dim (int): Embedding dimension.
        """
        super().__init__()
        self.K = K
        self.embed_dim = embed_dim
        self.stemifier = Stemifier()
        self.learnable_tokens = nn.Parameter(torch.randn(K, embed_dim))
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C)
        Returns:
            torch.Tensor: Output tensor of shape (B, K, T, C)
        """
        B, T, C = x.shape
        assert C == self.embed_dim, f"Expected input with last dim {self.embed_dim}, but got {C}"
        
        # Expand learnable tokens for the batch: (B, K, C)
        tokens = self.learnable_tokens.unsqueeze(0).expand(B, self.K, -1)
        
        # Expand x so that it repeats for each of the K streams: (B, K, T, C)
        x_expanded = x.unsqueeze(1).expand(B, self.K, T, C)
        
        # Prepend the conditioning tokens along the time dimension.
        # tokens.unsqueeze(2) gives (B, K, 1, C)
        # After concatenation along dim=2, inputs has shape (B, K, T+1, C)
        inputs = torch.cat([tokens.unsqueeze(2), x_expanded], dim=2)
        
        # Merge the batch and stream dimensions for processing: (B*K, T+1, C)
        inputs = inputs.view(B * self.K, T + 1, C)
        
        # Process with the stemifier block.
        outputs = self.stemifier(inputs)  # Expected shape: (B*K, T, C)
        
        # Reshape back to separate batch and stream dimensions: (B, K, T, C)
        outputs = outputs.view(B, self.K, T, C)
        
        return outputs


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
            kernel_size=3,
            stride=2,
            padding=1,
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

    def forward(self, x):
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

# # Example usage
# temporal_expansion = TemporalUpsample(input_dim=1024, output_dim=512)
# decoder_output = torch.randn(16, 137, 1024)  # Example input tensor [B=16, T=10, D=1024]
# expanded_output = temporal_expansion(decoder_output)  # Shape: [16, 20, 512]
# print(expanded_output.shape)

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