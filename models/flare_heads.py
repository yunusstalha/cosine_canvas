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
        if embed_dim % n_heads != 0:
            raise ValueError("embed_dim should be divisible by n_heads")        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim  -0.5

        # Single linear projection for QKV
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        
        self.attn_drop = nn.Dropout(attn_drop)

        # Output FFN
        self.out_projection = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Optional cache for keys and values during autoregressive sampling.
        self.use_cache: bool = False
        self.k_cache: None
        self.v_cache: None

        # Normalisation
        self.q_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.k_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

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
        q = q * self.scale # TODO NOT SUREEE

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
        self.attn = AutoregressiveAttention(embed_dim, num_heads, attn_drop=attn_drop proj_drop=proj_drop)
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
        x = x + gate_a * self.attn(apply_adaLN(self.norm1(x), shift_a, scale_a), attn_mask)
        x = x + gate_m * self.mlp(apply_adaLN(self.norm2(x), shift_m, scale_m))
        return x


class FinalAdaLN(nn.Module):
    """Final AdaLN before prediction."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        scale, shift = self.ada(cond).chunk(2, dim=-1)
        return apply_adaLN(self.norm(x), shift, scale)

# -----------------------------------------------------------------------------
# Discrete AR head (uniform quantisation)
# -----------------------------------------------------------------------------


class DiscreteARHead(nn.Module):
    """
    Discrete feature-wise autoregressive head:
    
    - Takes 'conditioning' [B, conditoning_dim] from your decoder.
    - Projects to 'feature_embed_dim' -> condition vector (cond).
    - Builds an AR sequence: [start_token + cond] + [embedding of channel 0] + ...
    - Passes the sequence through a tiny causal Transformer (C positions).
    - Finally, outputs logits [B, C, num_bins].
    
    If you only want to compute for certain positions, supply a boolean mask
    of shape [B] (with 1=keep). We'll gather those positions, run them through the
    AR, and then expand back out to [B,C,bins].
    """

    def __init__(
        self,
        conditoning_dim: int,  # e.g. 768
        feature_embed_dim: int,  # e.g. 256
        num_channels: int,       # e.g. 16 for a VAE with 16 channels
        num_bins: int,           # e.g. 32
        depth: int = 4,          # AR block layers
        num_heads: int = 8,      # # of heads in each AR block
        mlp_ratio: float = 4.0,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
    ): # This should acces to quantizer probably....
        super().__init__()
        self.num_channels = num_channels
        self.num_bins = num_bins

        # 1) Condition projection: [B, dec_emb] -> [B, feature_embed_dim]
        self.condition_proj = nn.Linear(conditoning_dim, feature_embed_dim)

        # 2) AR tokens for channels
        self.start_token = nn.Parameter(torch.zeros(1, 1, feature_embed_dim))
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
        
        self.feature_norm = nn.LayerNorm(feature_embed_dim)
        self.final_adaln = FinalAdaLN(feature_embed_dim, feature_embed_dim)

        # 5) For each channel, a linear classifier to produce logits
        self.heads = nn.ModuleList([
            nn.Linear(feature_embed_dim, num_bins)
            for _ in range(num_channels)
        ])

        # 6) Causal mask of size [C, C], used for the full sequence
        causal_mask = build_causal_mask(num_channels)
        self.register_buffer("channel_causal_mask", causal_mask)

        # 7) TODO, params and normalization stuff needs initialization to
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.start_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight, std=0.02)

    # --------------------------------------------------------------------------
    #  Forward pass (training)
    # --------------------------------------------------------------------------
    def forward(
        self,
        conditions: torch.Tensor,  # [B, conditoning_dim]
        feature_targets: torch.Tensor,   # [B, C] ground-truth discrete indices
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Returns: logits of shape [B, C, num_bins].
        If `mask` is given ([B] boolean), we only process the masked positions,
        then scatter the results back into [B, C, bins].
        """
        B = conditions.shape[0]

        # gather only masked positions if user wants
        if mask is not None:
            keep_idx = mask.nonzero(as_tuple=True)[0]
            conditions = conditions[keep_idx]
            feature_targets  = feature_targets[keep_idx]
        M = conditions.size(0)

        # 1) condition
        cond = self.condition_proj(conditions)  # [M, D]

        # 2) Build sequence: (cond + start_token) + token_embeds(c)
        seq = [cond.unsqueeze(1) + self.start_token]  # shape [M,1,D]
        for c in range(self.num_channels - 1):
            emb = self.token_embeddings[c](feature_targets[:, c])  # [M, D]
            seq.append(emb.unsqueeze(1))
        x = torch.cat(seq, dim=1)  # [M, C, D]

        # 3) Add channel embedding => [M, C, D]
        x = x + self.feature_pos_embed

        # 4) Pass through each AR block
        for blk in self.blocks:
            x = blk(x, cond, self.channel_causal_mask[: x.size(1), : x.size(1)])

        # final LN
        x = self.final_ln(x)  # [M,C,D]

        # 5) channel-wise logits
        logits_list = []
        for c in range(self.num_channels):
            logits_c = self.heads[c](x[:, c])  # [M, num_bins]
            logits_list.append(logits_c)
        logits = torch.stack(logits_list, dim=1)  # [M, C, num_bins]

        # if we masked out positions, put them back
        if mask is not None:
            out = torch.zeros(B, self.num_channels, self.num_bins, device=logits.device, dtype=logits.dtype)
            out[keep_idx] = logits
            return out
        else:
            return logits

    # --------------------------------------------------------------------------
    #  Sampling pass (inference)
    # --------------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, conditions: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Autoregressively sample discrete channels [B, C], one channel at a time.
        """
        B = conditions.size(0)
        cond = self.condition_proj(conditions)  # [B,D]

        # Turn on KV cache if your blocks support it
        self._enable_kv_cache(True)

        # Start sequence: shape [B, 1, D]
        seq = cond.unsqueeze(1) + self.start_token
        out_tokens = []

        for c in range(self.num_channels):
            # add channel embedding for the partial sequence
            # (which is length c+1 at iteration c)
            partial_len = seq.size(1)
            x = seq + self.feature_pos_embed[:, :partial_len, :]

            # pass x through AR blocks
            for blk in self.blocks:
                x = blk(x, cond, self.channel_causal_mask[:partial_len, :partial_len])
            x = self.final_ln(x)

            # logits for channel c => [B, bins]
            logits = self.heads[c](x[:, -1]) / (temperature if temperature > 0 else 1.0)
            probs  = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, 1).squeeze(-1)  # [B]
            out_tokens.append(next_idx)

            # embed next token if c < C-1
            if c < self.num_channels - 1:
                next_emb = self.token_embeddings[c](next_idx)  # [B,D]
                seq = torch.cat([seq, next_emb.unsqueeze(1)], dim=1)

        self._enable_kv_cache(False)
        return torch.stack(out_tokens, dim=1)  # [B,C]

    def _enable_kv_cache(self, enable: bool):
        """If your AR blocks implement a 'use_cache' toggle, we reset + enable caching."""
        for blk in self.blocks:
            blk.attn.use_cache = enable
            blk.attn.reset_cache()