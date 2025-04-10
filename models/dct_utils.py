from __future__ import annotations
"""
modules.py  
==============================================
Fast utility components for the FLARE implementation. All transformation matrices and the
zig‑zag scan pattern are pre‑computed **once** during construction and
cached as non‑trainable buffers so the per‑batch overhead is a pair of `torch.matmul` calls.

Implemented utilities
---------------------
create_dct_matrix(N) – returns the orthonormal DCT‑II matrix of size N×N.
FrequencyOrderer – converts latent tensors [B, C, H, W] ⇄ frequency sequences [B, H·W, C] using the cached DCT matrices and zig‑zag order. 
UniformQuantizer – dimension‑wise uniform quantiser.
"""

from typing import Tuple
import math

import torch
import torch.nn as nn

def create_dct_matrix(N: int, dtype=torch.float32, device="cpu") -> torch.Tensor:
    """Return the orthonormal DCT‑II matrix **D ∈ ℝ^{N×N}** such that
    **X̂ = D · X** computes the 1‑D DCT along the last dimension.
    """
    k = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # [N,1]
    n = torch.arange(N, device=device, dtype=dtype).unsqueeze(0)  # [1,N]
    D = torch.cos(math.pi / N * (n + 0.5) * k)  # [N,N]
    D[0] *= 1.0 / math.sqrt(2.0)                # scale first row
    D *= math.sqrt(2.0 / N)
    return D

def _zigzag_indices(h: int, w: int) -> torch.Tensor:
    """Generate a JPEG‑style zig‑zag traversal order for an h×w block.
    
    With this convention the ordering for an 8×8 block starts as:
    0, 1, 8, 16, 9, 2, ...
    """
    idx = []
    for s in range(h + w - 1):
        diag = []
        for y in range(s + 1):
            x = s - y
            if y < h and x < w:
                diag.append(y * w + x)
        if s % 2 == 0:
            diag = diag[::-1]
        idx.extend(diag)
    return torch.tensor(idx, dtype=torch.long)

class FrequencyOrderer(nn.Module):
    """Convert latent tensors **[B, C, H, W]** to / from frequency‑ordered
    sequences using cached DCT matrices.
    """

    def __init__(self, H: int, W: int):
        super().__init__()
        self.H, self.W = H, W
        # Pre‑compute transformation matrices and register as buffers so they
        # move automatically with .to(device) and are excluded from gradients.
        D_h = create_dct_matrix(H)  # [H,H]
        D_w = create_dct_matrix(W)  # [W,W]
        self.register_buffer("D_h", D_h, persistent=False)
        self.register_buffer("D_w", D_w, persistent=False)
        # Also store the transposes for the inverse transform
        self.register_buffer("D_h_T", D_h.t(), persistent=False)
        self.register_buffer("D_w_T", D_w.t(), persistent=False)
        # Zig‑zag order + inverse map
        zz = _zigzag_indices(H, W)
        self.register_buffer("zigzag", zz, persistent=False)            # [H·W]
        inv = torch.empty_like(zz)
        inv[zz] = torch.arange(H * W)
        self.register_buffer("inv_zigzag", inv, persistent=False)

    # ----------------------- helpers -----------------------
    @property
    def seq_len(self) -> int:
        return self.H * self.W

    # ------------------ forward transforms ----------------

    def to_sequence(self, latent: torch.Tensor) -> torch.Tensor:
        """Convert latent [B,C,H,W] → seq [B,H·W,C]."""
        B, C, H, W = latent.shape
        assert (H, W) == (self.H, self.W), "Latent resolution mismatch"
        # Apply DCT:  (B,C,H,W) → (B,C,H,W)
        #   First along height:  X̂ = D_h · X
        x = torch.matmul(self.D_h, latent)            # [H,H]·[B,C,H,W] broadcast on last dim
        #   Then along width:   X̂ = X̂ · D_w^T
        x = torch.matmul(x, self.D_w_T)               # [B,C,H,W]
        # Rearrange via zig‑zag
        x = x.view(B, C, H * W)
        x = x[..., self.zigzag]                       # [B,C,H·W]
        seq = x.permute(0, 2, 1).contiguous()         # [B,H·W,C]
        return seq

    # ------------------ inverse transforms ---------------

    def from_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        """Convert **seq [B,H·W,C]** → **latent [B,C,H,W]**."""
        B, N, C = seq.shape
        assert N == self.seq_len, "Sequence length mismatch"
        # Place coefficients back into spatial grid
        coeffs = seq.permute(0, 2, 1).contiguous()    # [B,C,N]
        tmp = torch.zeros(B, C, N, device=seq.device, dtype=seq.dtype)
        tmp[..., self.zigzag] = coeffs
        tmp = tmp.view(B, C, self.H, self.W)
        # Inverse DCT: first width then height (transpose order)
        x = torch.matmul(tmp, self.D_w)               # [B,C,H,W]
        x = torch.matmul(self.D_h_T, x)               # [B,C,H,W]
        return x

class UniformQuantizer(nn.Module):
    """Dimension‑wise uniform quantiser (post‑training)."""

    def __init__(self, num_bins: int, v_min: float, v_max: float):
        super().__init__()
        assert num_bins > 1, "Need at least two bins"
        self.num_bins = num_bins
        self.register_buffer("v_min", torch.tensor(v_min))
        self.register_buffer("v_max", torch.tensor(v_max))
        edges = torch.linspace(v_min, v_max, num_bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2
        self.register_buffer("edges", edges)
        self.register_buffer("centres", centres)

    # ------------------------------------------------------
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x_clamped = x.clamp(self.v_min.item(), self.v_max.item())
        idx = torch.bucketize(x_clamped, self.edges) - 1  # [0,num_bins-1]
        idx = idx.clamp(0, self.num_bins - 1)
        x_hat = self.centres[idx]
        return idx, x_hat

    def dequantise(self, idx: torch.Tensor) -> torch.Tensor:
        return self.centres[idx]
