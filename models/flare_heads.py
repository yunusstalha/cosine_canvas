from __future__ import annotations
"""
flare_heads.py
==============
Inner (dimension‑wise) autoregressive heads for FLARE.

The design is a trimmed‑down, readable re‑implementation of the AdaLN‑style
heads  discrete and GMM.  

Public API
----------
DiscreteARHead – predicts categorical logits for uniform‑quantised
  coefficients.  Use cross‑entropy during training.
GMMARHead – predicts parameters of a diagonal Gaussian Mixture Model
  per coefficient like ARINAR.  Provides nll() helper for negative log‑likelihood.

Both classes expose sample() for autoregressive generation.
"""

from typing import Tuple, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def apply_adaLN(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply AdaLN modulation (same as TokenBridge)."""
    # Print for debugging
    # print(f"apply_adaLN: x.shape={x.shape}, shift.shape={shift.shape}, scale.shape={scale.shape}")
    return x * (1 + scale) + shift



def build_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Upper‑triangular ‑inf mask for causal self‑attention."""
    m = torch.empty(seq_len, seq_len, device=device)
    m.fill_(float("-inf"))
    m.triu_(1)
    return m


# -----------------------------------------------------------------------------
# AR Causal Attention
# -----------------------------------------------------------------------------

class AutoregressiveAttention(nn.Module):
    """Single‑layer causal multi‑head self‑attention with optional KV caching."""

    def __init__(self, embed_dim: int, num_heads: int = 4, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim should be divisible by num_heads")        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Single linear projection for QKV
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        
        self.attn_drop = nn.Dropout(attn_drop)

        # Output FFN
        self.out_projection = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Optional cache for keys and values during autoregressive sampling.
        self.use_cache: bool = False
        self.k_cache = None
        self.v_cache = None

        # Normalisation
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        """Causal attention using `torch.nn.functional.scaled_dot_product_attention`.
        The built‑in kernel automatically selects Flash‑Attention / Triton
        implementations when available, giving a sizeable speed‑up over the
        manual softmax formulation.
        """
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # Apply normalisation to Q and K (layernorm)
        q = self.q_norm(q)  # [B, heads, N, head_dim]
        k = self.k_norm(k)  # [B, heads, N, head_dim]

        # Apply scaling here because SDPA does not take an explicit scale arg

        # --- KV‑cache handling (for sampling) ---------------------------------
        if self.use_cache:
            if self.k_cache is None:
                self.k_cache, self.v_cache = k, v
            else:
                self.k_cache = torch.cat([self.k_cache, k], dim=2)
                self.v_cache = torch.cat([self.v_cache, v], dim=2)
            k, v = self.k_cache, self.v_cache

        # SDPA expects shapes (B, heads, N, d)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )  # [B, heads, N, head_dim]

        out = attn_out.transpose(1, 2).reshape(B, N, C)
        out = self.proj_drop(self.out_projection(out))
        return out

    def reset_cache(self):
        """Clears the cached key and value tensors."""
        self.k_cache = None
        self.v_cache = None

class AdaptiveAutoRegressiveTransformer(nn.Module):
    """Tiny Causal Transformer block with AdaLN conditioning."""

    def __init__(self,  embed_dim: int, cond_dim: int, num_heads: int = 4, mlp_ratio: float = 4.0, proj_drop: float = 0.0, attn_drop: float = 0.0):

        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = AutoregressiveAttention(embed_dim, num_heads, attn_drop=attn_drop, proj_drop=proj_drop)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, embed_dim)
        )
        # AdaLN modulation – produce shift/scale + gates for attn & mlp (6*dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, 6 * embed_dim)
        )


    def forward(self, x: torch.Tensor, cond: torch.Tensor, attn_mask: torch.Tensor):
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.adaLN(cond).chunk(6, dim=-1)
        # Print shapes for debugging
        # print(f"shift_a: {shift_a.shape}, scale_a: {scale_a.shape}, gate_a: {gate_a.shape}")
        # print(f"shift_m: {shift_m.shape}, scale_m: {scale_m.shape}, gate_m: {gate_m.shape}")
        # print(f"x: {x.shape}, attn_mask: {attn_mask.shape}")
        x = x + gate_a.unsqueeze(1) * self.attn(
            apply_adaLN(self.norm1(x), shift_a.unsqueeze(1), scale_a.unsqueeze(1)),
            attn_mask)       
        x = x + gate_m.unsqueeze(1) * self.mlp(
            apply_adaLN(self.norm2(x), shift_m.unsqueeze(1), scale_m.unsqueeze(1))
        ) 
        return x


class FinalAdaLN(nn.Module):
    """Final AdaLN before prediction."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim, bias=True))

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        scale, shift = self.ada(cond).chunk(2, dim=-1)
        return apply_adaLN(self.norm(x), shift.unsqueeze(1), scale.unsqueeze(1))
# -----------------------------------------------------------------------------
# Discrete AR head (uniform quantisation)
# -----------------------------------------------------------------------------


class DiscreteARHead(nn.Module):
    """
    Discrete feature-wise autoregressive head (TokenBridge-inspired):
    - Takes 'conditioning' [B, conditioning_dim] from outer AR model.
    - Projects condition to 'feature_embed_dim' -> cond vector.
    - Builds an AR sequence: [cond] + [embedding of channel 0] + ...
    - Uses feature_pos_embed for channel position awareness.
    - Passes sequence through causal Transformer conditioned via AdaLN on 'cond'.
    - Outputs logits [B, C, num_bins] for each channel.
    - Handles optional input mask for selective processing (e.g., MAR).
    """

    def __init__(
        self,
        conditioning_dim: int,  # e.g. 768 (embedding dimension of the outer model)
        feature_embed_dim: int,  # e.g. 256 (embedding dimension of this head)
        num_channels: int,       # e.g. 16 for a VAE with 16 channels (Channel Number of the VAE)
        num_bins: int,           # e.g. 32 
        depth: int = 4,          # AR block layers
        num_heads: int = 8,      # # of heads in each AR block
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_bins = num_bins
        self.feature_embed_dim = feature_embed_dim # D

        # 1) Condition projection: [B, conditioning_dim] -> [B, feature_embed_dim]
        self.condition_proj = nn.Linear(conditioning_dim, feature_embed_dim)

        # 2) AR tokens for channels
        # For each channel except the last, we have an embedding for discrete indices
        # (the last channel doesn't need an embed for the "next" token)
        self.token_embeddings = nn.ModuleList([
            nn.Embedding(num_bins, feature_embed_dim)
            for _ in range(num_channels - 1)
        ])

        # 3) Learned channel embeddings: shape [1, C, D]
        #    This is added to each time-step so that the AR block knows which channel
        self.feature_pos_embed = nn.Parameter(torch.zeros(1, num_channels, feature_embed_dim))

        # 4) A small stack of causal AR blocks
        self.blocks = nn.ModuleList([
            AdaptiveAutoRegressiveTransformer(
                embed_dim=feature_embed_dim,
                cond_dim=feature_embed_dim,  # we do AdaLN with the same dimension
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_drop,
                attn_drop=attn_drop
            ) for _ in range(depth)
        ])
        
        self.final_adaln = FinalAdaLN(feature_embed_dim, feature_embed_dim)

        # 5) For each channel, a linear classifier to produce logits
        self.heads = nn.ModuleList([
            nn.Linear(feature_embed_dim, num_bins)
            for _ in range(num_channels)
        ])

        # 6) Causal mask of size [C, C], used for the full sequence
        full_causal_mask = build_causal_mask(num_channels)
        self.register_buffer("channel_causal_mask", full_causal_mask, persistent=False)

        # 7) TODO, params and normalization stuff needs initialization to
        self._init_weights()

    def _init_weights(self):
        # Initialize projections like TokenBridge (Xavier uniform)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                 if m.bias is not None:
                     nn.init.zeros_(m.bias)
                 if m.weight is not None: # AdaLN norms might have elementwise_affine=False
                     nn.init.ones_(m.weight)

        # Special init for AdaLN final layer bias/weights can sometimes help
        for block in self.blocks:
             nn.init.zeros_(block.adaLN[-1].bias) # Zero init bias of final AdaLN linear layer
             # Optionally zero init weight too, like DiT:
             nn.init.zeros_(block.adaLN[-1].weight)
        nn.init.zeros_(self.final_adaln.ada[-1].bias)
        nn.init.zeros_(self.final_adaln.ada[-1].weight)

        # Init positional embedding
        nn.init.trunc_normal_(self.feature_pos_embed, std=0.02)
    # --------------------------------------------------------------------------
    #  Forward pass (training)
    # --------------------------------------------------------------------------
    def forward(self,
            conditions: torch.Tensor, # [B, conditioning_dim], context z_k from outer model
            feature_targets: torch.Tensor, # [B, C] ground-truth discrete indices (already frequency-ordered)
            mask: Optional[torch.Tensor] = None # Optional: [B] boolean mask to select batch items
        ) -> torch.Tensor:
        """
        Predicts logits for each channel dimension based on the condition and
        previous ground-truth channel tokens (teacher forcing).

        Args:
            conditions: Conditioning vectors from the outer AR model.
            feature_targets: Ground-truth quantized indices for all channels [B, C].
            mask: Optional boolean mask [B] indicating which items in the batch to process.

        Returns:
            Logits of shape [B, C, num_bins] (or [M, C, num_bins] if mask is used).
            If mask is used, the output corresponds only to the masked items,
            but retains the C and num_bins dimensions. The caller needs to handle scattering if needed.
            *Correction:* Let's return the scattered full shape [B, C, num_bins] for consistency.
        """
        B = conditions.shape[0]
        device = conditions.device

        # Handle masking: Select subset of inputs if mask is provided
        if mask is not None:
            keep_idx = mask.nonzero(as_tuple=True)[0]
            if keep_idx.numel() == 0: # Handle case where mask is all False
                 return torch.zeros(B, self.num_channels, self.num_bins, device=device, dtype=conditions.dtype)
            conditions_eff = conditions[keep_idx]
            feature_targets_eff = feature_targets[keep_idx]
            M = conditions_eff.size(0) # Effective batch size
        else:
            conditions_eff = conditions
            feature_targets_eff = feature_targets
            M = B

        # 1) Project condition: [M, conditioning_dim] -> [M, D]
        cond = self.condition_proj(conditions_eff)

        # 2) Build AR input sequence x = [initial_state, embed(q_0), ..., embed(q_{C-2})]
        #    The initial state is derived from the condition.
        #    Sequence length is C.
        seq_list = [cond.unsqueeze(1)] # Start with projected condition [M, 1, D] as the first element
        for c in range(self.num_channels - 1):
            # Embed the ground-truth token for channel c
            emb = self.token_embeddings[c](feature_targets_eff[:, c]) # [M, D]
            seq_list.append(emb.unsqueeze(1)) # [M, 1, D]

        x = torch.cat(seq_list, dim=1) # Shape [M, C, D]

        # 3) Add channel positional embeddings
        x = x + self.feature_pos_embed # Shape [1, C, D] broadcasts

        # 4) Pass through AR blocks with AdaLN conditioning
        #    The causal mask ensures prediction at step c only uses info up to c-1
        #    We use the first C x C block of the precomputed causal mask
        active_causal_mask = self.channel_causal_mask[:self.num_channels, :self.num_channels]
        for blk in self.blocks:
            # Pass the base condition 'cond' [M, D] for AdaLN modulation
            x = blk(x, cond, active_causal_mask)

        # Apply final AdaLN layer
        x = self.final_adaln(x, cond) # Shape [M, C, D]

        # 5) Apply channel-wise prediction heads
        #    The output at sequence position c corresponds to the prediction FOR channel c
        logits_list = []
        for c in range(self.num_channels):
            logits_c = self.heads[c](x[:, c]) # Use output state at index c, shape [M, num_bins]
            logits_list.append(logits_c)
        logits_eff = torch.stack(logits_list, dim=1) # Shape [M, C, num_bins]

        # Scatter results back if masking was used
        if mask is not None:
            # Create output tensor of full batch size B
            logits = torch.zeros(B, self.num_channels, self.num_bins, device=device, dtype=logits_eff.dtype)
            logits[keep_idx] = logits_eff
            return logits
        else:
            return logits_eff

    # --------------------------------------------------------------------------
    #  Sampling pass (inference)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, conditions: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Autoregressively sample discrete channel indices [B, C], one channel at a time.

        Args:
            conditions: Conditioning vectors [B, conditioning_dim] from outer AR model.
            temperature: Sampling temperature (0 for argmax).

        Returns:
            Sampled quantized indices [B, C].
        """
        B = conditions.size(0)
        device = conditions.device
        D = self.feature_embed_dim

        # 1) Project condition: [B, conditioning_dim] -> [B, D]
        cond = self.condition_proj(conditions)

        # Enable KV cache for attention blocks
        self._enable_kv_cache(True)

        # Start sequence with the condition vector
        seq = cond.unsqueeze(1) # Initial sequence shape [B, 1, D]

        out_tokens = torch.zeros(B, self.num_channels, dtype=torch.long, device=device)

        for c in range(self.num_channels):
            # Current sequence length
            partial_len = seq.size(1) # Starts at 1, grows to C

            # Prepare input for this step: Add positional embedding
            # We only need to process the *last* element of the sequence input usually,
            # but the attention needs the full sequence history (handled by KV cache).
            # Let's pass the current sequence 'seq' augmented with pos embedding.
            x_in = seq + self.feature_pos_embed[:, :partial_len, :]

            # Pass through AR blocks. AdaLN uses the base condition 'cond'.
            # The causal mask is implicitly handled by processing one step at a time
            # with KV caching. We only need the output for the last token.
            # Attention mask for SDPA should be None or correctly shaped for cached keys/values.
            # For step c, query length is 1, key length is c+1. Causal mask should be fine.
            # Let's assume AutoregressiveAttention handles masking with cache correctly.
            x_proc = x_in
            for blk in self.blocks:
                x_proc = blk(x_proc, cond, attn_mask=None) # Rely on cache + causal nature

            # Apply final AdaLN
            x_out = self.final_adaln(x_proc, cond) # Shape [B, partial_len, D]

            # Get logits for the current channel 'c' using the output corresponding
            # to the *last* token in the processed sequence.
            logits = self.heads[c](x_out[:, -1]) # Shape [B, num_bins]

            # Apply temperature and sample
            if temperature == 0.0:
                next_idx = torch.argmax(logits, dim=-1) # [B]
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1).squeeze(-1) # [B]

            out_tokens[:, c] = next_idx

            # If not the last channel, prepare input for the next step
            if c < self.num_channels - 1:
                # Embed the sampled token
                next_emb = self.token_embeddings[c](next_idx) # Shape [B, D]
                # Append the embedding to the sequence history for the next iteration
                seq = torch.cat([seq, next_emb.unsqueeze(1)], dim=1) # Shape grows to [B, c+2, D]

        # Disable and clear KV cache
        self._enable_kv_cache(False)

        return out_tokens # Shape [B, C]

    def _enable_kv_cache(self, enable: bool):
        """Enable or disable KV caching in attention layers."""
        for blk in self.blocks:
            blk.attn.use_cache = enable
            if not enable: # Reset cache when disabling
                 blk.attn.reset_cache()