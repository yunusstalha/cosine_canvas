# flare.py
from functools import partial
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.stats as stats
from torch.utils.checkpoint import checkpoint
from timm.models.vision_transformer import Block # Using ViT blocks like MAR/TokenBridge

# Import inner AR heads
from .flare_heads import DiscreteARHead, GMMARHead 

from .dct_utils import FrequencyOrderer, UniformQuantizer
from typing import Optional


# Placeholder for quantization range finding function
# def find_dct_coefficient_range(dataset, vae_encoder, dct_func, num_components):
#     # Logic to iterate dataset, encode, DCT, flatten, and find min/max
#     # This would likely happen *outside* the model definition during setup
#     # Returns quant_min, quant_max (potentially per-channel or global)
#     print("Warning: Placeholder find_dct_coefficient_range called.")
#     return -5.0, 5.0 # Dummy values



class FLARE(nn.Module):
    """
    FLARE: Frequency-Latent Autoregressive Model

    Combines a pre-trained VAE's latent space with DCT-based frequency ordering
    and codebook-free autoregressive modeling (Discrete Uniform Quantization or GMM)
    using an MAE-style outer Transformer backbone and dimension-wise inner AR heads.
    """
    def __init__(self,
                 # --- Architecture Config ---
                 img_size=256,              # For calculating H, W of latent space
                 vae_stride=16,             # VAE downsampling factor
                 encoder_embed_dim=768,     # Dimension of outer Transformer blocks
                 encoder_depth=14,
                 encoder_num_heads=12,
                 decoder_embed_dim=768,     # Usually same as encoder for MAE
                 decoder_depth=14,
                 decoder_num_heads=12,
                 mlp_ratio=4.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 grad_checkpointing=False,

                 # --- VAE/Input Config ---
                 vae_embed_dim=16,          # Number of channels (C) in VAE latent space

                 # --- FLARE Specific Config ---
                 head_type="discrete_uniform", # 'discrete_uniform' or 'gmm'
                 buffer_size=1,             # For class token (can be adjusted)
                 mask_ratio_min=0.7,        # For MAR-style training
                 label_drop_prob=0.1,
                 class_num=1000,            # Number of classes for conditioning

                 # --- Inner AR Head Config (Passed to DiscreteARHead or GMMARHead) ---
                 inner_ar_embed_dim=256,    # Dimension inside the inner AR head's Transformer
                 inner_ar_depth=4,
                 inner_ar_num_heads=8,
                 # --- Discrete Path Config ---
                 num_bins=32,               # Number of uniform quantization bins
                 quant_min=-5.0,            # Min value for DCT coefficient range (needs proper calculation)
                 quant_max=5.0,             # Max value for DCT coefficient range (needs proper calculation)
                 # --- GMM Path Config ---
                 num_gaussians=10,           # Number of components in GMM if 

                 **kwargs # Capture any unused args, like ones MAR had
                 ):
        super().__init__()

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.vae_embed_dim = vae_embed_dim # C: Number of VAE channels
        self.head_type = head_type
        self.num_bins = num_bins if head_type == "discrete_uniform" else None
        self.quant_min = quant_min if head_type == "discrete_uniform" else None
        self.quant_max = quant_max if head_type == "discrete_uniform" else None
        self.grad_checkpointing = grad_checkpointing

        # --- Calculate Sequence Length and Get Frequency Orderer for DCT---
        self.latent_h = self.latent_w = img_size // vae_stride
        self.freq_orderer = FrequencyOrderer(self.latent_h, self.latent_w)
        self.num_freq_components = self.freq_orderer.seq_len # N = H*W

        
        # --- Input Token Dimension for Outer AR Model ---
        # Each "token" in the sequence is the C-dimensional vector of DCT coeffs
        # for a specific frequency component.
        self.token_embed_dim = self.vae_embed_dim # C

        # --------------------------------------------------------------------------
        # Class Embedding for CFG
        # --------------------------------------------------------------------------
        self.num_classes = class_num
        # self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.class_emb = nn.Embedding(class_num + 1, encoder_embed_dim)

        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        self.buffer_size = buffer_size # Typically 1 for just the class token

        # --------------------------------------------------------------------------
        # MAR variant masking ratio (from MAR)
        # --------------------------------------------------------------------------
        self.mask_ratio_generator = stats.truncnorm(
            (mask_ratio_min - 1.0) / 0.25, 0, loc=1.0, scale=0.25
        )

        # --------------------------------------------------------------------------
        # Outer AR Encoder Specifics (MAE Style)
        # --------------------------------------------------------------------------
        # Project C-dimensional input token to encoder_embed_dim
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6) # Optional LayerNorm after proj

        # Positional embedding for the frequency sequence + class token buffer
        self.encoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.num_freq_components + self.buffer_size, encoder_embed_dim)
        )

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout)
            for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # Outer AR Decoder Specifics (MAE Style)
        # --------------------------------------------------------------------------
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(
            torch.zeros(1, self.num_freq_components + self.buffer_size, decoder_embed_dim)
        )

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        # Optional: Extra pos embed added *before* feeding to inner AR head (like TokenBridge's output_pos_embed)
        # Might help differentiate frequency steps for the inner head's conditioning
        self.decoder_output_pos_embed = nn.Parameter(
             torch.zeros(1, self.num_freq_components, decoder_embed_dim)
        )

        # --------------------------------------------------------------------------
        # Inner AR Head Instantiation
        # --------------------------------------------------------------------------
        if head_type == "discrete_uniform":
            print(f"FLARE: Initializing DiscreteARHead with {num_bins} bins.")
            self.inner_ar_head = DiscreteARHead(
                conditioning_dim=decoder_embed_dim, # Output dim of outer decoder
                feature_embed_dim=inner_ar_embed_dim,
                num_channels=self.vae_embed_dim,    # C
                num_bins=num_bins,
                depth=inner_ar_depth,
                num_heads=inner_ar_num_heads,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_dropout,
                attn_drop=attn_dropout,
            )
            # Initialize uniform quantization parameters
            print(f"FLARE: Initializing UniformQuantizer with {num_bins} bins from {quant_min:.4f} to {quant_max:.4f}")
            self.quantizer = UniformQuantizer(num_bins, quant_min, quant_max)

        elif head_type == "gmm":
            print(f"FLARE: Initializing GMMARHead with {num_gaussians} Gaussians.")
            self.inner_ar_head = GMMARHead(
                conditioning_dim=decoder_embed_dim, # Output dim of outer decoder
                feature_embed_dim=inner_ar_embed_dim,
                num_channels=self.vae_embed_dim,    # C
                num_gaussians=num_gaussians,
                depth=inner_ar_depth,
                num_heads=inner_ar_num_heads,
                mlp_ratio=mlp_ratio,
                proj_drop=proj_dropout,
                attn_drop=attn_dropout,
            )
        else:
            raise ValueError(f"Unsupported head_type: {head_type}")

        # Initialize weights for the outer model components
        self.initialize_weights()
        
        # Head Batch Mul (from ARINAR - might not be needed if inner head handles batching)
        # Let's remove it for now unless performance dictates its need.
        # self.head_batch_mul = 1 


    def initialize_weights(self):
        # Initialize class emb, fake latent, mask token, pos embeds (like MAR)
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_output_pos_embed, std=.02)

        # Initialize nn.Linear and nn.LayerNorm using helper (like MAR)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # Xavier uniform following ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    # --------------------------------------------------------------------------
    # Masking and Order Sampling (Adapted from MAR - No changes needed here)
    # --------------------------------------------------------------------------
    def sample_orders(self, bsz):
        orders = []
        for _ in range(bsz):
            order = np.array(list(range(self.num_freq_components))) # N = H*W
            np.random.shuffle(order)
            orders.append(order)
        orders = torch.from_numpy(np.array(orders)).long()
        return orders

    def random_masking(self, x, orders):
        """
        Generates a random mask for the frequency sequence based on MAR strategy.
        Input x shape: [B, N, C] or [B, N, D_enc] - needs N dimension.
        Output mask shape: [B, N]
        """
        bsz, seq_len, _ = x.shape # seq_len here is N = num_freq_components
        if seq_len != self.num_freq_components:
             raise ValueError(f"Input sequence length {seq_len} does not match expected {self.num_freq_components}")

        mask_rate = self.mask_ratio_generator.rvs(1)[0]
        num_masked_tokens = int(np.ceil(seq_len * mask_rate))
        
        # Ensure orders is on the same device as x
        orders_dev = orders.to(x.device)
        
        mask = torch.zeros(bsz, seq_len, device=x.device, dtype=torch.bool) # Use bool mask
        
        # Gather indices to mask
        indices_to_mask = orders_dev[:, :num_masked_tokens] # [B, num_masked]
        
        # Use scatter_ to put True at masked positions
        mask.scatter_(dim=1, index=indices_to_mask, value=True)
        
        return mask # Boolean mask: True means MASKED

    # --------------------------------------------------------------------------
    # MAE Encoder Forward
    # --------------------------------------------------------------------------
    def forward_mae_encoder(self, x_seq, mask, class_embedding):
        """
        Projects input, handles buffer/class token, adds pos embedding,
        selects unmasked+buffer tokens, passes through encoder blocks.
        Returns ONLY the encoded tensor.

        Args:
            x_seq: Input frequency sequence [B, N, C] (continuous)
            mask: Boolean mask [B, N], True indicates MASKED tokens
            class_embedding: [B, encoder_embed_dim]

        Returns:
            Encoded representation of UNMASKED tokens + class token [B, num_unmasked+1, encoder_embed_dim]
        """
        B, N, C = x_seq.shape
        if N != self.num_freq_components:
             raise ValueError(f"Input sequence length {N} does not match expected {self.num_freq_components}")

        # Project input tokens [B, N, C] -> [B, N, D_enc]
        x = self.z_proj(x_seq)
        embed_dim = x.shape[-1] # Should be encoder_embed_dim

        numeric_mask = mask.float() # [B, N], 0.0=unmasked, 1.0=masked

        # concat buffer
        x = torch.cat([
            torch.zeros(B, self.buffer_size, embed_dim, device=x.device, dtype=x.dtype),
            x
        ], dim=1) # Shape [B, buffer+N, D_enc]

        mask_with_buffer = torch.cat([
            torch.zeros(B, self.buffer_size, device=mask.device, dtype=numeric_mask.dtype),
            numeric_mask # Use numeric mask here
        ], dim=1) # Shape [B, buffer+N], 0.0=buffer/unmasked, 1.0=masked

        if self.training and self.label_drop_prob > 0.:
            drop_latent_mask = (torch.rand(B, device=x.device) < self.label_drop_prob).unsqueeze(-1)
            # Ensure class_embedding is correctly shaped and typed
            class_embedding_bc = class_embedding.to(x.dtype) # [B, D_enc]
            fake_latent_bc = self.fake_latent.repeat(B, 1).to(x.dtype) # [B, D_enc]
            effective_class_embedding = torch.where(drop_latent_mask, fake_latent_bc, class_embedding_bc)
        else:
            effective_class_embedding = class_embedding.to(x.dtype) # [B, D_enc]

        # Place class embedding into the buffer part of x
        x[:, :self.buffer_size] = effective_class_embedding.unsqueeze(1) # [B, 1, D_enc] -> broadcasts if buffer_size > 1? Assumes buffer_size=1

        # encoder position embedding (Add to the whole sequence including buffer)
        x = x + self.encoder_pos_embed_learned # [1, buffer+N, D_enc] broadcasts
        x = self.z_proj_ln(x)

        # dropping: Select buffer and unmasked tokens
        # (1.0 - mask_with_buffer) is 1.0 for buffer/unmasked, 0.0 for masked
        keep_indices = (1.0 - mask_with_buffer).nonzero(as_tuple=True)
        x = x[keep_indices].reshape(B, -1, embed_dim) # [B, num_buffer+num_unmasked, D_enc]

        # apply Transformer blocks
        for blk in self.encoder_blocks:
            if self.grad_checkpointing and self.training and not torch.jit.is_scripting():
                 x = checkpoint(blk, x)
            else:
                 x = blk(x)
        x = self.encoder_norm(x)
        return x
    # --------------------------------------------------------------------------
    # MAE Decoder Forward
    # --------------------------------------------------------------------------
    # --------------------------------------------------------------------------
    # MAE Decoder Forward (Updated to match TokenBridge/ARINAR)
    # --------------------------------------------------------------------------
    def forward_mae_decoder(self, x_encoded, mask):
        """
        Embeds encoder output (buffer+unmasked), reconstructs the full sequence
        using mask tokens and the original mask, adds pos embedding, passes
        through decoder blocks. Adds a final output-specific pos embedding
        and removes the class token before returning.

        Args:
            x_encoded: Output from encoder [B, num_buffer+num_unmasked, D_enc]
            mask: Boolean mask [B, N], True indicates MASKED tokens (from original input)

        Returns:
            Decoded representation for ALL sequence tokens (including mask tokens) [B, N, D_dec]
            (Excludes the class token, includes decoder output positional embedding)
        """
        # Embed tokens into decoder space
        # x_encoded is [B, num_buffer+num_unmasked, D_enc]
        x = self.decoder_embed(x_encoded) # Shape: [B, num_buffer+num_unmasked, D_dec]
        B, num_encoded_tokens, D_dec = x.shape
        N = self.num_freq_components
        full_seq_len_with_buffer = self.buffer_size + N

        # --- Start: Following TokenBridge/ARINAR Logic ---

        # Convert boolean mask (True=masked) to numeric (0=unmasked, 1=masked)
        numeric_mask = mask.float() # Shape: [B, N]

        # Create mask_with_buffer (0=buffer/unmasked, 1=masked)
        mask_with_buffer = torch.cat([
            torch.zeros(B, self.buffer_size, device=mask.device, dtype=numeric_mask.dtype),
            numeric_mask
        ], dim=1) # Shape: [B, buffer+N]

        # Create placeholder for the full sequence (buffer + N tokens) filled with mask tokens
        mask_tokens_placeholder = self.mask_token.repeat(
            B, full_seq_len_with_buffer, 1
        ).to(x.dtype) # Shape: [B, buffer+N, D_dec]
        x_after_pad = mask_tokens_placeholder.clone()

        # Indices corresponding to buffer and UNMASKED tokens in the full sequence
        # (1.0 - mask_with_buffer) is 1.0 for buffer/unmasked, 0.0 for masked
        keep_indices = (1.0 - mask_with_buffer).nonzero(as_tuple=True)
        # keep_indices is a tuple: (batch_indices, sequence_indices_with_buffer)

        # Place the encoded tokens `x` into the placeholder `x_after_pad`
        # x has shape [B, num_buffer+num_unmasked, D_dec]
        # We need to place the elements of x at the positions specified by keep_indices in x_after_pad
        # Reshape x to match the number of keep_indices locations before placing
        if x.shape[1] != keep_indices[0].shape[0] // B: # Basic check
             # This should not happen if encoder output size matches keep indices count
             raise ValueError("Mismatch between encoder output size and number of keep indices.")

        # Reshape x to match the total number of kept tokens across the batch
        x_reshaped = x.reshape(-1, D_dec) # Shape: [TotalKeptTokens, D_dec]

        # Use the keep_indices to place the reshaped encoder output into the placeholder
        x_after_pad[keep_indices] = x_reshaped

        # Now x_after_pad contains the buffer & unmasked tokens from the encoder,
        # and mask tokens elsewhere. Shape: [B, buffer+N, D_dec]

        # Add decoder positional embeddings (learned) to the reconstructed sequence
        x = x_after_pad + self.decoder_pos_embed_learned # [B, buffer+N, D_dec]

        # Apply Transformer blocks
        for blk in self.decoder_blocks:
             if self.grad_checkpointing and self.training and not torch.jit.is_scripting():
                 x = checkpoint(blk, x)
             else:
                 x = blk(x)
        x = self.decoder_norm(x) # [B, buffer+N, D_dec]

        # Remove class token (buffer) part
        decoder_out = x[:, self.buffer_size:, :] # [B, N, D_dec]

        # Add separate positional embeddings for the output prediction head (only for freq tokens)
        # This corresponds to output_pos_embed / diffusion_pos_embed_learned
        decoder_out = decoder_out + self.decoder_output_pos_embed # [B, N, D_dec]

        # --- End: Following TokenBridge/ARINAR Logic ---

        return decoder_out


  
        # --------------------------------------------------------------------------
        # Inner AR Head Loss Calculation
        # --------------------------------------------------------------------------
    def forward_inner_ar_loss(self, decoder_output, target_features, mask):
        """
        Calculates the loss using the appropriate inner AR head.
        Selects the decoder outputs and target features only for the masked positions using the boolean mask. 
        It then calls the appropriate self.inner_ar_head.forward_loss method.
        Targets are indices for discrete_uniform and continuous values for gmm.

        Args:
            decoder_output: Output from the decoder [B, N, D_dec]
            target_features: Ground truth features (continuous or indices) [B, N, C] or [B, N, C] (long)
                            Shape depends on head_type.
            mask: Boolean mask [B, N], True indicates MASKED tokens (these are the targets)

        Returns:
            loss: Scalar loss value from the inner AR head.
        """
        # Select the decoder outputs corresponding to MASKED positions
        # decoder_output is [B, N, D_dec]
        # mask is [B, N]
        # masked_output should be [num_masked_total, D_dec]
        masked_output = decoder_output[mask]

        # Select the target features corresponding to MASKED positions
        # target_features is [B, N, C]
        # mask is [B, N]
        # masked_targets should be [num_masked_total, C]
        # We need to expand the mask to match the C dimension
        masked_targets = target_features[mask]

        if masked_output.shape[0] == 0:
            # Handle edge case where no tokens are masked (e.g., mask_ratio=0)
            # Return zero loss with correct device and requires_grad status
            return torch.tensor(0.0, device=decoder_output.device, requires_grad=True)

        if self.head_type == "discrete_uniform":
            # DiscreteARHead.forward expects 'conditions' and 'feature_targets', returns logits
            # It also has an optional 'mask' argument for batch filtering, which we don't need here
            # as we have already filtered the batch items via mask_bool selection.
            predicted_logits = self.inner_ar_head(
                conditions=masked_output,        # Correct name
                feature_targets=masked_targets  # Correct name
                # mask=None (default)
            )
            loss = F.cross_entropy(
                predicted_logits.reshape(-1, self.num_bins), # Note: num_bins needs to be accessible (e.g., self.num_bins)
                masked_targets.reshape(-1) # Target indices should already be long
            )
        elif self.head_type == "gmm":
                # GMMARHead.forward expects 'conditions' and 'feature_targets', returns NLL loss directly
                loss = self.inner_ar_head(
                    conditions=masked_output,
                    feature_targets=masked_targets # GMM expects float targets here
                    # mask=None (default)
                )
        return loss

    # --------------------------------------------------------------------------
    # Main Forward Pass
    # --------------------------------------------------------------------------
    def forward(self, latent_tensor, labels):
        """
        Orchestrates the forward pass for training FLARE.
        1. Converts VAE latent tensor to frequency sequence.
        2. Prepares targets (continuous or quantized indices).
        3. Gets class embeddings (with dropout).
        4. Generates mask using random ordering.
        5. Runs MAE encoder (TokenBridge/ARINAR style).
        6. Runs MAE decoder (TokenBridge/ARINAR style).
        7. Calculates loss using the inner AR head on masked positions.

        Args:
            latent_tensor: Input latent tensor from VAE encoder [B, C, H, W]
            labels: Class labels [B]

        Returns:
            loss: Scalar loss value.
        """
        B, C, H, W = latent_tensor.shape
        # --- 0. Convert latent to frequency sequence ---
        # Ensure freq_orderer is on the same device
        self.freq_orderer = self.freq_orderer.to(latent_tensor.device)
        x_seq = self.freq_orderer.to_sequence(latent_tensor) # [B, N, C], N=H*W

        # --- 1. Prepare Inputs & Targets ---
        # Target is original continuous sequence for GMM, indices for Discrete
        target_features = x_seq.detach().clone()
        input_for_encoder = x_seq # Use original continuous sequence as input

        if self.head_type == "discrete_uniform":
            if self.quantizer is None:
                 raise ValueError("Quantizer must be initialized for discrete_uniform head type.")
            # Quantize the sequence to get target indices
            with torch.no_grad(): # Don't need gradients through quantization indices
                # Quantizer needs to be on the correct device
                self.quantizer = self.quantizer.to(x_seq.device)
                quantized_indices, x_seq_dequantized = self.quantizer(x_seq) # [B, N, C], [B, N, C]
            target_features = quantized_indices # Target for discrete head is the indices
            # Optional: Use dequantized values as input to encoder
            # input_for_encoder = x_seq_dequantized

        # --- 2. Get Class Embedding ---
        # Drop labels based on prob (using your existing logic)
        if self.training and self.label_drop_prob > 0.:
             label_mask = torch.rand(B, device=labels.device) > self.label_drop_prob
             # Use num_classes as the dummy index for dropped labels if needed by embedding layer
             effective_labels = torch.where(label_mask, labels, torch.tensor(self.num_classes, device=labels.device))
        else:
             effective_labels = labels
        class_embedding = self.class_emb(effective_labels).to(input_for_encoder.dtype) # [B, D_enc]

        # --- 3. Generate Mask ---
        orders = self.sample_orders(B) # [B, N]
        # Generate boolean mask (True=masked) - consistent with random_masking output
        mask_bool = self.random_masking(input_for_encoder, orders) # [B, N] boolean

        # --- 4. Encoder Pass (Updated Call) ---
        # Input to encoder is continuous x_seq (or optionally dequantized)
        # Pass the boolean mask
        x_encoded = self.forward_mae_encoder(input_for_encoder, mask_bool, class_embedding)
        # x_encoded: [B, num_buffer+num_unmasked, D_enc] (No ids_restore returned)

        # --- 5. Decoder Pass (Updated Call) ---
        # Pass the boolean mask
        decoder_output = self.forward_mae_decoder(x_encoded, mask_bool)
        # decoder_output: [B, N, D_dec]

        # --- 6. Inner AR Head Loss ---
        # Pass the boolean mask for selecting targets/outputs for loss calculation
        # forward_inner_ar_loss expects the boolean mask
        loss = self.forward_inner_ar_loss(decoder_output, target_features, mask_bool)

        return loss
    def mask_by_order(self, num_masked: int, orders: torch.Tensor, bsz: int, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Creates a boolean mask based on orders.
        Marks the first 'num_masked' elements according to 'orders' as True (masked/unknown).

        Args:
            num_masked: Target number of tokens to be masked (True).
            orders: Permutation orders [B, N].
            bsz: Batch size.
            seq_len: Sequence length N.
            device: Torch device.

        Returns:
            Boolean mask [B, N] where True indicates masked.
        """
        num_masked = max(0, min(seq_len, int(num_masked))) # Clamp to [0, N]

        mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device) # Initialize mask to False (known)
        # Select the indices that should be masked based on the random order
        indices_to_mask = orders[:, :num_masked] # Shape [B, num_masked]

        if indices_to_mask.numel() > 0: # Check if there's anything to mask
            # Use scatter_ to mark these positions as True (masked)
            mask.scatter_(dim=1, index=indices_to_mask, value=True)

        return mask # True means masked/unknown
    @torch.no_grad()
    def sample_tokens(self,
                      bsz: int,
                      num_iter: int = 16, # Number of decoding steps (like MAR)
                      cfg_scale: float = 1.0,
                      cfg_schedule="linear", # Or "constant"
                      labels: Optional[torch.Tensor] = None,
                      temperature: float = 1.0,
                      top_k: Optional[int] = None, # Optional top-k for discrete head
                      progress: bool = False # Optional progress bar
                     ):
        """
        Iterative sampling using the MAE backbone and inner AR head, with CFG.
        Adapts MAR/MaskGIT iterative decoding for FLARE's frequency sequence.

        Args:
            bsz: Batch size.
            num_iter: Number of iterative decoding steps.
            cfg_scale: Classifier-Free Guidance scale. If <= 1.0, disabled.
            cfg_schedule: How CFG scale changes over iterations ('linear' or 'constant').
            labels: Optional class labels [B]. If None, unconditional generation forced.
            temperature: Sampling temperature for inner head (GMM stddev scale / softmax temp).
            top_k: Optional top-k sampling for discrete head logits. Not implemented in DiscreteARHead.sample yet.
            progress: Show tqdm progress bar.

        Returns:
            Generated sequence of continuous DCT coefficients [B, N, C].
        """
        self.eval() # Set model to evaluation mode
        device = self.encoder_pos_embed_learned.device # Get device from a parameter
        N = self.num_freq_components # H*W
        C = self.vae_embed_dim
        D_dec = self.decoder_embed.out_features # Decoder dimension

        # --- 1. Prepare Class Embeddings for CFG ---
        if labels is None:
            print("Warning: No labels provided for sampling, forcing unconditional generation (cfg_scale=1.0)")
            labels = torch.randint(0, self.num_classes, (bsz,), device=device) # Dummy labels for shape
            cfg_scale = 1.0 # Force disable CFG

        cond_emb = self.class_emb(labels).to(device=device, dtype=self.z_proj.weight.dtype) # [B, D_enc]

        if cfg_scale > 1.0:
            # Use the dedicated unconditional index (num_classes)
            uncond_labels = torch.full_like(labels, self.num_classes)
            uncond_emb = self.class_emb(uncond_labels).to(device=device, dtype=cond_emb.dtype) # [B, D_enc]
            # Duplicate batch for CFG
            batch_emb = torch.cat([cond_emb, uncond_emb], dim=0) # [2*B, D_enc]
            eff_bsz = 2 * bsz
        else:
            batch_emb = cond_emb # [B, D_enc]
            eff_bsz = bsz

        # --- 2. Initialization ---
        # Sample random orders for masking schedule
        orders = self.sample_orders(bsz).to(device) # [B, N]
        if cfg_scale > 1.0:
            orders = torch.cat([orders, orders], dim=0) # Duplicate for CFG batch

        # Start with all tokens masked (True = unknown/masked)
        current_mask = torch.ones(eff_bsz, N, dtype=torch.bool, device=device)

        # Initialize sequence guess (continuous values)
        current_seq_guess = torch.zeros(eff_bsz, N, C, device=device, dtype=batch_emb.dtype)

        # --- 3. Iterative Decoding Loop ---
        iter_range = tqdm(range(num_iter), desc="FLARE Sampling") if progress else range(num_iter)
        for step in iter_range:
            # Duplicate state for CFG *if* needed (only input guess needed duplication inside loop)
            if cfg_scale > 1.0 and current_seq_guess.shape[0] == bsz:
                 # If first CFG step or state got reduced somehow, duplicate
                 current_seq_guess = torch.cat([current_seq_guess, current_seq_guess.clone()], dim=0)
                 current_mask = torch.cat([current_mask[:bsz], current_mask[:bsz].clone()], dim=0) # Ensure mask matches eff_bsz

            # --- a. Run MAE Encoder-Decoder ---
            # Input is always continuous guess
            x_encoded = self.forward_mae_encoder(current_seq_guess, current_mask, batch_emb)
            decoder_output = self.forward_mae_decoder(x_encoded, current_mask) # [eff_bsz, N, D_dec]

            # --- b. Apply CFG ---
            if cfg_scale > 1.0:
                cond_out, uncond_out = decoder_output.chunk(2, dim=0)
                # Calculate CFG scale for this step
                if cfg_schedule == "linear":
                     # Linear decay from cfg to 1.0 (borrowed from MAR/ARINAR logic)
                     # This depends on number of *masked* tokens, let's use step directly for simplicity
                     # Alternative: decay based on mask ratio
                     # ratio_known = 1.0 - (current_mask[:bsz].sum() / N).item() # Approx fraction known
                     # cfg_iter = 1 + (cfg_scale - 1) * (1.0 - ratio_known)
                     cfg_iter = 1 + (cfg_scale - 1) * (1.0 - (step + 1) / num_iter) # Linear decay based on step
                else: # constant
                     cfg_iter = cfg_scale
                eff_decoder_output = uncond_out + cfg_iter * (cond_out - uncond_out) # [B, N, D_dec]
                current_mask_cond = current_mask[bsz:].clone() # Use mask from conditional part [B, N]
            else:
                eff_decoder_output = decoder_output # [B, N, D_dec]
                current_mask_cond = current_mask.clone() # [B, N]

            # --- c. Determine Mask for Next Step & Tokens to Predict ---
            # Cosine schedule for masking ratio (fraction of tokens to keep masked)
            mask_ratio_next = np.cos(math.pi / 2. * (step + 1) / num_iter)
            num_masked_target = math.ceil(N * mask_ratio_next)

            # Get mask for the *next* iteration
            mask_next = self.mask_by_order(num_masked_target, orders[:bsz], bsz, N, device) # [B, N]

            # Determine which tokens to predict *now*
            if step == num_iter - 1:
                 # Predict all remaining masked tokens in the last step
                 mask_to_pred = current_mask_cond # [B, N]
            else:
                 # Predict tokens that are masked now but known in the next step
                 mask_to_pred = torch.logical_xor(current_mask_cond, mask_next) # [B, N]

            # Find indices of tokens to predict
            masked_indices = mask_to_pred.nonzero(as_tuple=False) # [NumPredTotal, 2] -> [batch_idx, seq_idx_k]

            if masked_indices.shape[0] == 0:
                 if progress: iter_range.set_postfix({"status": "All tokens predicted"})
                 # print(f"Step {step}: No tokens left to predict.")
                 break # Nothing left to predict

            # --- d. Sample New Tokens using Inner Head ---
            # Select the conditioning vectors (z_k) for the tokens to predict
            masked_conditions = eff_decoder_output[masked_indices[:, 0], masked_indices[:, 1]] # [NumPredTotal, D_dec]

            # Call the inner head's sample method
            # It handles the channel-wise AR loop internally
            # TODO: Add top_k to DiscreteARHead.sample if needed
            new_token_values = self.inner_ar_head.sample(
                conditions=masked_conditions,
                temperature=temperature
            ) # Shape [NumPredTotal, C] (Indices for discrete, Floats for GMM)
            # --- e. Update Sequence Guess and Mask ---
            # Prepare continuous values to update the guess
            if self.head_type == "discrete_uniform":
                if not hasattr(self, 'quantizer') or self.quantizer is None:
                     raise RuntimeError("FLARE discrete model must have a quantizer initialized for sampling.")
                # Dequantize the sampled indices
                self.quantizer = self.quantizer.to(device) # Ensure device
                new_token_values_continuous = self.quantizer.dequantise(new_token_values) # [NumPredTotal, C]
            else: # GMM head already outputs continuous values
                new_token_values_continuous = new_token_values # [NumPredTotal, C]

            # Update the *conditional* part of the sequence guess
            # Make a copy to avoid in-place modification issues if needed later
            updated_seq_guess_cond = current_seq_guess[:bsz].clone()
            updated_seq_guess_cond[masked_indices[:, 0], masked_indices[:, 1]] = new_token_values_continuous.to(updated_seq_guess_cond.dtype)

            # Update main sequence guess (only need conditional part for next input)
            current_seq_guess = updated_seq_guess_cond # Shape [B, N, C]

            # Update the mask for the next iteration
            current_mask = mask_next # Shape [B, N]
            # No need to duplicate mask here, will be handled at start of next loop if cfg > 1.0

            if progress: iter_range.set_postfix({"masked_next": num_masked_target})


        # --- 4. Final Output ---
        final_sequence = current_seq_guess[:bsz].clone() # Ensure we only return the conditional batch part [B, N, C]
        self.train() # Set model back to training mode
        return final_sequence


# --- Factory functions (optional, similar to MAR) ---
def flare_base(**kwargs):
    # Example configuration for a "base" size model
    model = FLARE(
        encoder_embed_dim=768, encoder_depth=14, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=14, decoder_num_heads=12,
        inner_ar_embed_dim=256, inner_ar_depth=4, inner_ar_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def flare_large(**kwargs):
     # Example configuration for a "large" size model
    model = FLARE(
        encoder_embed_dim=1024, encoder_depth=18, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=18, decoder_num_heads=16,
        inner_ar_embed_dim=512, inner_ar_depth=6, inner_ar_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
