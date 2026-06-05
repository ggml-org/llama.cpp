"""
Minimal SigLIP ViT + merger model definitions for CoreML export.

This file is self-contained: no dependency on transformers or any external
modeling code.  Only standard PyTorch modules are used so that
torch.jit.trace → coremltools.convert works reliably.

Currently supported:
  - MiniCPM-V 4.6   (SigLIP 980px, 27 layers, hidden=1152, insert_layer=6)
Extending to other families: add new detection logic in export_coreml.py and,
if needed, new merger / pipeline classes here.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _gelu_pytorch_tanh(x: torch.Tensor) -> torch.Tensor:
    """Exact copy of transformers.activations.gelu_pytorch_tanh."""
    return (
        0.5
        * x
        * (
            1.0
            + torch.tanh(
                math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
            )
        )
    )


# ---------------------------------------------------------------------------
# SigLIP Vision Transformer (eager attention, no flash-attn dependency)
# ---------------------------------------------------------------------------

class SiglipVisionConfig:
    """Bare-bones config — no PretrainedConfig inheritance needed for export."""

    def __init__(
        self,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_hidden_layers: int = 27,
        num_attention_heads: int = 16,
        num_channels: int = 3,
        image_size: int = 980,
        patch_size: int = 14,
        layer_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        hidden_act: str = "gelu_pytorch_tanh",
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
        self.hidden_act = hidden_act


class SiglipAttention(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(2, 3)) * self.scale
        if attention_mask is not None:
            attn = attn + attention_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(hidden_states.dtype)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(out), attn


class SiglipMLP(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act = _gelu_pytorch_tanh if config.hidden_act == "gelu_pytorch_tanh" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        x = self.layer_norm1(hidden_states)
        x, _ = self.self_attn(x, attention_mask=None)
        hidden_states = residual + x

        residual = hidden_states
        hidden_states = residual + self.mlp(self.layer_norm2(hidden_states))
        return hidden_states


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches_per_side = config.image_size // config.patch_size
        self.num_positions = self.num_patches_per_side**2

        self.patch_embedding = nn.Conv2d(
            config.num_channels, config.hidden_size,
            kernel_size=self.patch_size, stride=self.patch_size, padding=0,
        )
        self.position_embedding = nn.Embedding(self.num_positions, config.hidden_size)

    def forward(self, pixel_values: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """pixel_values: (B, 3, P, P*N), position_ids: (B, N) int32"""
        x = self.patch_embedding(pixel_values)          # (B, D, H, W)
        x = x.flatten(2).transpose(1, 2)               # (B, N, D)
        return x + self.position_embedding(position_ids)


class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        self.layers = nn.ModuleList([SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states: torch.Tensor, insert_layer_id: int = -1,
                merger: Optional[nn.Module] = None,
                merger_patch_w: Optional[torch.Tensor] = None) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states)
            if i == insert_layer_id and merger is not None:
                hidden_states = merger(hidden_states, merger_patch_w)
        return hidden_states


# ---------------------------------------------------------------------------
# MiniCPM-V 4.6 insert merger
# ---------------------------------------------------------------------------

class ViTInsertMerger(nn.Module):
    """Fixed-shape 2x2 window self-attention + ViTmlp downsampling.

    Input:  (B, N, D)     where N = patch_h * patch_w
    Output: (B, N//4, D)   after 2×2 grouping

    The 2×2 window permutation is self-inverse, so window_indices == reverse_indices.
    """

    def __init__(self, config: SiglipVisionConfig, num_patches: int):
        super().__init__()
        D = config.hidden_size
        I = config.intermediate_size
        self.D = D
        self.merged_D = D * 4
        self.merged_I = I * 4
        self.num_patches = num_patches
        self.num_windows = num_patches // 4

        self.layer_norm1 = nn.LayerNorm(D, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)

        self.pre_norm = nn.LayerNorm(self.merged_D, eps=1e-6)
        self.linear_1 = nn.Linear(self.merged_D, self.merged_I, bias=True)
        self.act = _gelu_pytorch_tanh
        self.linear_2 = nn.Linear(self.merged_I, D, bias=True)

        self.register_buffer("base_idx", torch.arange(num_patches, dtype=torch.int32))

    @staticmethod
    def compute_window_indices(base_idx: torch.Tensor, patch_w: torch.Tensor) -> torch.Tensor:
        half_w = patch_w // 2
        win_id = base_idx // 4
        local_pos = base_idx % 4
        local_row = local_pos // 2
        local_col = local_pos % 2
        win_row = win_id // half_w
        win_col = win_id % half_w
        return ((2 * win_row + local_row) * patch_w + (2 * win_col + local_col)).to(torch.int32)

    def forward(self, hidden_states: torch.Tensor, patch_w: torch.Tensor) -> torch.Tensor:
        B = hidden_states.shape[0]
        D = self.D
        NW = self.num_windows

        window_indices = self.compute_window_indices(self.base_idx, patch_w)

        # stage 1 — window self-attention
        residual = hidden_states
        x = self.layer_norm1(hidden_states)
        x = torch.index_select(x, 1, window_indices).view(B * NW, 4, D)
        x, _ = self.self_attn(x, attention_mask=None)
        x = x.view(B, self.num_patches, D)
        x = torch.index_select(x, 1, window_indices)  # self-inverse → reverse
        hidden_states = residual + x

        # stage 2 — ViTmlp 2×2 downsampling
        x = torch.index_select(hidden_states, 1, window_indices).view(B, NW, 4, D)
        residual = x.mean(dim=2)
        x = x.reshape(B, NW, self.merged_D)
        x = self.pre_norm(x)
        x = self.linear_2(self.act(self.linear_1(x)))
        return x + residual


# ---------------------------------------------------------------------------
# Full vision pipeline (pixel_values + patch_w → llm_embed_dim output)
# ---------------------------------------------------------------------------

class DownsampleMLP(nn.Module):
    """Final MLP merger: 4*D → D → llm_embed_dim."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.pre_norm = nn.LayerNorm(input_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim, bias=True),
            nn.GELU(),
            nn.Linear(input_dim, output_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.pre_norm(x))


class FullVisionPipeline(nn.Module):
    """
    Complete ViT pipeline for CoreML export.

    Inputs:
      pixel_values  F32  [B, 3, patch_size, patch_size * max_patches]
      patch_w       I32  [1]   — patch grid width

    Output:
      embedding     F32  [B, num_patches//16, llm_embed_dim]

    position_ids and all window indices are computed inside the graph
    from `patch_w` — the host only sends raw pixels and a single integer.
    """

    def __init__(
        self,
        vpm_embeddings: SiglipVisionEmbeddings,
        vpm_encoder: SiglipEncoder,
        vpm_post_ln: nn.LayerNorm,
        vit_merger: ViTInsertMerger,
        mlp_merger: DownsampleMLP,
        insert_layer_id: int,
        num_patches: int,
        num_patches_per_side: int,
    ):
        super().__init__()
        self.patch_embeddings = vpm_embeddings
        self.encoder = vpm_encoder
        self.post_layernorm = vpm_post_ln
        self.vit_merger = vit_merger
        self.mlp_merger = mlp_merger
        self.insert_layer_id = insert_layer_id
        self.num_patches = num_patches
        self.num_final = num_patches // 16
        self.nps = num_patches_per_side  # e.g. 70 for 980/14

        self.register_buffer("base_idx", torch.arange(num_patches, dtype=torch.int32))
        self.register_buffer("base_idx_m2", torch.arange(num_patches // 4, dtype=torch.int32))

    def forward(self, pixel_values: torch.Tensor, patch_w: torch.Tensor) -> torch.Tensor:
        # 0. position_ids from patch_w
        patch_h = self.num_patches // patch_w
        row = self.base_idx // patch_w
        col = self.base_idx % patch_w
        bucket_h = (row * self.nps // patch_h).to(torch.int32)
        bucket_w = (col * self.nps // patch_w).to(torch.int32)
        position_ids = bucket_h * self.nps + bucket_w

        # 1. patch embed + position embed
        x = self.patch_embeddings(pixel_values, position_ids)

        # 2. encoder (with insert merger in the middle)
        x = self.encoder(x, self.insert_layer_id, self.vit_merger, patch_w)

        # 3. post layernorm
        x = self.post_layernorm(x)

        # 4. MLP merger (2×2 downsample → llm_embed_dim)
        B = x.shape[0]
        D = x.shape[2]
        half_w = patch_w // 2
        m2_indices = ViTInsertMerger.compute_window_indices(self.base_idx_m2, half_w)
        x = torch.index_select(x, 1, m2_indices).view(B, self.num_final, 4 * D)
        return self.mlp_merger(x)
