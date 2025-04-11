
# FLARE Dimension-Wise Heads & DCT Utilities

This document describes the implementation of the dimension‑wise autoregressive “heads” for FLARE, as well as the DCT/zigzag utility for frequency ordering.

## Overview

In FLARE, each image (or each latent representation) is transformed into a sequence of tokens that we want to predict autoregressively. Concretely:

1. **VAE Latent:** We start with a `[B, C, H, W]` latent from a VAE (or a standard image `[B, 3, H, W]`, but typically a smaller `C`).
2. **DCT & Zig‑zag:** We apply the 2D DCT to each spatial feature map and then reorder the coefficients in a zig‑zag pattern.  
   - Result: a sequence `[B, N, C]` where `N = H*W` and each “token” is a `C`-dim vector of DCT coefficients (one coefficient per channel).
3. **Outer AR Model:** A Transformer or other AR backbone processes these `N` tokens (lowest freq → highest freq). At each step `k`, it outputs a context vector `z_k ∈ ℝ^(conditioning_dim)`.

4. **Dimension‑wise Head:** For that single token (the set of `C` coefficients at step `k`), we use a dimension‑wise AR approach to predict them:
   - The “Discrete” head uses uniform quantization of each coefficient and predicts them as categorical indices (with cross‑entropy).
   - The “Continuous” head uses a GMM to model each coefficient’s real value (with NLL).

### Folder Structure

- **`flare_heads.py`**  
  Contains the two main heads:
  - `DiscreteARHead`  
  - `GMMARHead`  
  plus smaller utility modules for AdaLN and a tiny causal Transformer.

- **`dct_utils.py`**  
  Implements:
  - `FrequencyOrderer`: DCT + zig‑zag and inverse.
  - `UniformQuantizer`: Post‑training uniform quantization.

---

## DCT & Zig‑zag (dct_utils.py)

### FrequencyOrderer(H, W)

A module that precomputes:
- The orthonormal DCT‑II matrices for height (`H x H`) and width (`W x W`).
- A zig‑zag permutation to reorder the `[H x W]` frequency bands from low to high frequency.

**Key Methods**:
- `to_sequence(latent)`:  
  Takes an input `[B, C, H, W]` and returns `[B, H*W, C]` by applying 2D‑DCT and then zig‑zag flattening.
- `from_sequence(seq)`:  
  The inverse of `to_sequence(...)`. Takes `[B, H*W, C]` in zig‑zag order, reconstructs the 2D DCT planes, then applies the inverse DCT to get back `[B, C, H, W]`.

**Usage Example**:
```python
from dct_utils import FrequencyOrderer
import torch

# Suppose we have a latent [B, C, H, W]
latent = torch.randn(4, 16, 16, 16)  # e.g. B=4, C=16, H=16, W=16

freq_orderer = FrequencyOrderer(H=16, W=16)
seq = freq_orderer.to_sequence(latent)  # shape [4, 256, 16]
reconstructed = freq_orderer.from_sequence(seq)  # back to [4, 16, 16, 16]

print(seq.shape, reconstructed.shape)
# (torch.Size([4, 256, 16]), torch.Size([4, 16, 16, 16]))
```

### UniformQuantizer(num_bins, v_min, v_max)

Used for **post‑training** discrete quantization.  
- `forward(x) -> (idx, x_hat)`:  
  - `idx`: The integer bin indices for each value in `x`.
  - `x_hat`: The “dequantized” center values of those bins.  
- `dequantise(idx) -> x_hat`:  
  Converts back from bin indices to the center bin values.

**Usage Example**:
```python
from dct_utils import UniformQuantizer
import torch

x = torch.randn(4, 16)  # 16 channels, for instance
# Suppose from data analysis you found v_min=-3.0, v_max=3.0, and want 32 bins
quantizer = UniformQuantizer(num_bins=32, v_min=-3.0, v_max=3.0)

idx, x_hat = quantizer(x)
print(idx.shape, x_hat.shape)
# e.g. torch.Size([4,16]), torch.Size([4,16])

# Later, if you just have idx and want the continuous approximation:
x_hat2 = quantizer.dequantise(idx)
```

---

## Dimension-Wise Heads (flare_heads.py)

Both heads implement a small causal Transformer that operates *over the channel dimension* (C).  
For each channel `c` from 0 to C-1, the head is fed:
- The **condition** vector from the outer AR model.
- The previously known channels (in teacher forcing or sampling mode).
- Produces the distribution for the current channel.

### 1. DiscreteARHead(...)

This head handles **discrete** uniform quantization indices. It expects integer channel values in `[0, num_bins-1]`.

**Constructor Arguments** (key ones):
- `conditioning_dim`: dimension of the context vector from the outer AR model.
- `feature_embed_dim`: internal size for the dimension‑wise Transformer. 
- `num_channels`: how many channels we will autoregress over.
- `num_bins`: how many discrete bins are used per channel.
- `depth`, `num_heads`: the depth (number of Transformer blocks) and attention heads.

**Forward Pass**:  
- `forward(conditions, feature_targets, mask=None)`
  - `conditions`: `[B, conditioning_dim]` context.  
  - `feature_targets`: `[B, C]` ground‑truth discrete indices of the channels, for teacher forcing.  
  - `mask`: optionally mask out some batch elements.  
- Returns `[B, C, num_bins]` logits for each channel’s distribution.

**Sampling**:
- `sample(conditions, temperature=1.0) -> [B, C]`
- Runs an internal AR loop from channel=0 to channel=C-1.

<details>
<summary>Example Code Snippet</summary>

```python
from flare_heads import DiscreteARHead
import torch

# Suppose the outer AR model gives us a context vector z_k of size [B, 768]
# and we have C=16 channels. We used 32 bins for uniform quantization.
discrete_head = DiscreteARHead(
    conditioning_dim=768,
    feature_embed_dim=256,
    num_channels=16,
    num_bins=32,
    depth=4,
    num_heads=8,
)

# Forward pass (training)
B = 4
conditions = torch.randn(B, 768)
feature_targets = torch.randint(0, 32, (B, 16))  # ground truth discrete indices
logits = discrete_head(conditions, feature_targets)
print("Logits shape:", logits.shape)
# [4, 16, 32]

# Cross-entropy loss
loss = torch.nn.functional.cross_entropy(
    logits.permute(0, 2, 1),
    feature_targets
)
loss.backward()

# Sampling (inference mode)
with torch.no_grad():
    sampled_indices = discrete_head.sample(conditions, temperature=1.0)
print("Sampled shape:", sampled_indices.shape)
# [4, 16]
```

</details>

---

### 2. GMMARHead(...)

This head implements a **Gaussian Mixture Model** approach for each channel, similar to ARINAR. Instead of discrete bins, each channel’s value is modeled continuously via a mixture of Gaussians.

**Constructor Arguments**:
- `conditioning_dim`: dimension of the outer AR context vector.
- `feature_embed_dim`: internal dimension of the channel Transformer.
- `num_channels`: number of channels to predict dimension‑wise.
- `num_gaussians`: how many mixture components per channel.
- `depth`, `num_heads`: number of blocks / attention heads.

**Forward Pass**:
- `forward(conditions, feature_targets, mask=None)`
- Produces a scalar negative log-likelihood (NLL) loss for the GMM.

**Sampling**:
- `sample(conditions, temperature=1.0) -> [B, C]`
- Autoregressively samples each channel from the predicted GMM.

<details>
<summary>Example Code Snippet</summary>

```python
from flare_heads import GMMARHead
import torch

# Suppose the outer AR context is [B, 512], we have 8 channels, and 6 gaussians per channel.
gmm_head = GMMARHead(
    conditioning_dim=512,
    feature_embed_dim=256,
    num_channels=8,
    num_gaussians=6,
    depth=4,
    num_heads=4,
)

B = 4
conditions = torch.randn(B, 512)
feature_targets = torch.randn(B, 8)  # continuous ground truth

# Forward pass (training)
loss = gmm_head(conditions, feature_targets)
loss.backward()
print("NLL Loss:", loss.item())

# Sampling
with torch.no_grad():
    sampled_values = gmm_head.sample(conditions, temperature=0.8)
print("Sampled shape:", sampled_values.shape)
# [4, 8]
```

</details>

---

## Putting It All Together

In a **typical FLARE pipeline**:

1. You would **encode** an image into VAE latents `[B, C, H, W]`.
2. Use `FrequencyOrderer(H, W)` to get `[B, N, C]` in frequency order (`N = H * W`).
3. Train an **outer Transformer** to process these `N` tokens in a causal manner. At each step `k`, that outer Transformer outputs a context `conditions_k`.
4. Then, for each token’s channel dimension:
   - Use either `DiscreteARHead` (with quantized indices) or `GMMARHead` (with continuous GMM).
   - Produce the next set of channel values.

The code in `flare_heads.py` shows how to do that dimension‑wise pass for each token’s channels. The *outer loop* over `k` in `[0..N-1]` is not in these classes; it’s part of your higher‑level FLARE model.

**That’s all!** With these heads and the DCT tools, you have the building blocks to implement FLARE’s dimension‑wise token modeling in the frequency domain.
