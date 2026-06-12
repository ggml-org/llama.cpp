"""
Minimal CLIP ViT + projector model definitions for CoreML export.

Self-contained: no dependency on transformers or external modeling code.
Only standard PyTorch modules used for reliable torch.jit.trace → coremltools.convert.

Currently used by:
  - Llava 1.5   (CLIP ViT-L/14-336, 24 layers, hidden=1024, patch=14, image=336)
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# CLIP Vision Transformer
# ---------------------------------------------------------------------------

class CLIPVisionConfig:
    """Bare-bones config for CLIP ViT."""

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 4096,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 336,
        patch_size: int = 14,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_patches = (image_size // patch_size) ** 2


class CLIPAttention(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        B, N, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(out)


class CLIPMLP(nn.Module):
    """QuickGELU activation, matching CLIP's original implementation."""

    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)

    @staticmethod
    def quick_gelu(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.quick_gelu(self.fc1(x)))


class CLIPEncoderLayer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.self_attn = CLIPAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = CLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = residual + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class CLIPVisionEmbeddings(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.image_size = config.image_size
        self.num_patches = config.num_patches  # e.g. 576

        self.patch_embedding = nn.Conv2d(
            config.num_channels, config.hidden_size,
            kernel_size=self.patch_size, stride=self.patch_size, padding=0,
        )
        self.class_embedding = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.position_embedding = nn.Embedding(self.num_patches + 1, config.hidden_size)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, 3, 336, 336) → (B, 577, 1024)"""
        B = pixel_values.shape[0]
        x = self.patch_embedding(pixel_values)          # (B, D, 24, 24)
        x = x.flatten(2).transpose(1, 2)               # (B, 576, D)

        cls = self.class_embedding.expand(B, -1, -1)    # (B, 1, D)
        x = torch.cat((cls, x), dim=1)                  # (B, 577, D)

        # Hardcode position_ids (shape never changes for this model)
        pos_ids = torch.arange(self.num_patches + 1, device=x.device, dtype=torch.int32).unsqueeze(0)
        return x + self.position_embedding(pos_ids)


class CLIPVisionTransformer(nn.Module):
    def __init__(self, config: CLIPVisionConfig):
        super().__init__()
        self.embeddings = CLIPVisionEmbeddings(config)
        self.pre_layrnorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.encoder = nn.ModuleList([CLIPEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.post_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values → CLIP ViT last hidden state (patch tokens only, no CLS)"""
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)
        for layer in self.encoder:
            x = layer(x)
        x = self.post_layernorm(x)
        # Drop CLS token — Llava uses patch tokens only
        return x[:, 1:, :]


# ---------------------------------------------------------------------------
# Llava multi-modal projector
# ---------------------------------------------------------------------------

class MLPProjector(nn.Module):
    """Simple 2-layer MLP: vision_hidden → intermediate → llm_hidden."""

    def __init__(self, vision_hidden: int, intermediate: int, llm_hidden: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(vision_hidden, intermediate, bias=True),
            nn.GELU(),
            nn.Linear(intermediate, llm_hidden, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# ---------------------------------------------------------------------------
# Full Llava vision pipeline
# ---------------------------------------------------------------------------

class LlavaVisionPipeline(nn.Module):
    """
    Complete Llava vision pipeline for CoreML export.

    Input:  pixel_values  F32  [1, 3, 336, 336]
    Output: embedding     F32  [1, 576, llm_embed_dim]

    576 = (336/14)^2, the number of non-overlapping patches.
    """

    def __init__(
        self,
        vit: CLIPVisionTransformer,
        projector: MLPProjector,
    ):
        super().__init__()
        self.vit = vit
        self.projector = projector

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.vit(pixel_values)
        return self.projector(x)
