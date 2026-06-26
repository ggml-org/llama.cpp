"""
Phi-4-Mini-Flash-Reasoning conversion for llama.cpp GGUF format.

Place this file in: llama.cpp/conversion/phi4flash.py

Then register it in: llama.cpp/conversion/__init__.py
by adding the following import at the bottom of the file, alongside
the other model imports:

    from .phi4flash import Phi4FlashModel

That's all that is needed — the @ModelBase.register decorator on the
class does the rest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

import numpy as np
import torch

import gguf

from .model_base import ModelBase, TextModel

if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("Phi4FlashForCausalLM")
class Phi4FlashModel(TextModel):
    model_arch = gguf.MODEL_ARCH.PHI4FLASH

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mb_per_layer = self.hparams.get("mb_per_layer", 2)
        self.sliding_window = self.hparams.get("sliding_window", 512)

    def set_gguf_parameters(self):
        n_embd      = self.find_hparam(["hidden_size", "n_embd"])
        n_head      = self.find_hparam(["num_attention_heads", "n_head"])
        n_head_kv   = self.find_hparam(["num_key_value_heads", "n_head_kv", "num_attention_heads"])
        rms_eps     = self.find_hparam(["layer_norm_eps", "rms_norm_eps"])
        max_pos     = self.find_hparam(["max_position_embeddings", "n_positions"])
        n_ff        = self.find_hparam(["intermediate_size"])

        self.gguf_writer.add_embedding_length(n_embd)
        self.gguf_writer.add_feed_forward_length(n_ff)
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_head_count(n_head)
        self.gguf_writer.add_head_count_kv(n_head_kv)
        self.gguf_writer.add_layer_norm_rms_eps(rms_eps)
        # Also write the standard LayerNorm epsilon used by the output norm
        self.gguf_writer.add_layer_norm_eps(rms_eps)
        self.gguf_writer.add_context_length(max_pos)
        self.gguf_writer.add_file_type(self.ftype)

        # ── Phi-4-Flash–specific metadata ────────────────────────────────────
        #
        # Layer-type array (one entry per block):
        #   0 = Mamba-1           (even blocks 0–16)
        #   1 = SWA diff-attn     (odd  blocks 1–15)
        #   2 = Full diff-attn    (block 17, YOCO pivot)
        #   3 = GMU               (even blocks 18–30)
        #   4 = Cross diff-attn   (odd  blocks 19–31)
        #
        layer_types = []
        for i in range(self.block_count):
            is_mamba = (i % self.mb_per_layer == 0)
            if i < self.block_count // 2:
                layer_types.append(0 if is_mamba else 1)
            elif i == self.block_count // 2:
                layer_types.append(0)          # layer 16: last Mamba
            elif i == self.block_count // 2 + 1:
                layer_types.append(2)          # layer 17: YOCO pivot (full diff-attn)
            else:
                layer_types.append(3 if is_mamba else 4)

        self.gguf_writer.add_array("phi4flash.layer_types",   layer_types)
        self.gguf_writer.add_uint32("phi4flash.mb_per_layer", self.mb_per_layer)
        self.gguf_writer.add_uint32("phi4flash.sliding_window", self.sliding_window)
        self.gguf_writer.add_uint32("phi4flash.pivot_layer",    17)
        self.gguf_writer.add_uint32("phi4flash.ssm_cache_layer", 16)

        # Mamba-1 intrinsic dimensions (fixed for this model)
        self.gguf_writer.add_uint32("phi4flash.ssm_d_conv",   4)
        self.gguf_writer.add_uint32("phi4flash.ssm_d_state",  16)
        self.gguf_writer.add_uint32("phi4flash.ssm_d_inner",  5120)
        self.gguf_writer.add_uint32("phi4flash.ssm_dt_rank",  160)
        self.gguf_writer.add_uint32("phi4flash.ssm_expand",   2)

    # ─────────────────────────────────────────────────────────────────────────
    # Tensor name mapping
    # ─────────────────────────────────────────────────────────────────────────
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # ── Global (non-layer) tensors ────────────────────────────────────────
        if name == "model.embed_tokens.weight":
            return [("token_embd.weight", data_torch)]
        if name == "model.final_layernorm.weight":
            return [("output_norm.weight", data_torch)]
        if name == "model.final_layernorm.bias":
            return [("output_norm.bias", data_torch)]
        if name == "lm_head.weight":
            # Tied to embed_tokens — skip; llama.cpp handles this automatically.
            return []

        if not name.startswith("model.layers.") or bid is None:
            return super().modify_tensors(data_torch, name, bid)

        rest = name.split(f"model.layers.{bid}.")[-1]

        # ── Determine layer type ──────────────────────────────────────────────
        is_mamba = (bid % self.mb_per_layer == 0)
        if bid < self.block_count // 2:
            layer_type = 0 if is_mamba else 1
        elif bid == self.block_count // 2:
            layer_type = 0          # layer 16
        elif bid == self.block_count // 2 + 1:
            layer_type = 2          # layer 17
        else:
            layer_type = 3 if is_mamba else 4

        # ── Per-block norms (shared by all layer types) ───────────────────────
        if rest == "input_layernorm.weight":
            return [(f"blk.{bid}.attn_norm.weight", data_torch)]
        if rest == "input_layernorm.bias":
            return [(f"blk.{bid}.attn_norm.bias",   data_torch)]
        if rest == "post_attention_layernorm.weight":
            return [(f"blk.{bid}.ffn_norm.weight", data_torch)]
        if rest == "post_attention_layernorm.bias":
            return [(f"blk.{bid}.ffn_norm.bias",   data_torch)]

        # ── FFN / MLP (shared by all layer types) ────────────────────────────
        # HF uses gate_up_proj (fused SwiGLU) or separate gate_proj/up_proj.
        # llama.cpp stores the fused [2*n_ff, n_embd] tensor as ffn_up and
        # handles the SwiGLU split internally.
        if rest == "mlp.gate_up_proj.weight":
            # Already fused [2*n_ff, n_embd] — ggml reversal → [n_embd, 2*n_ff] ✓
            return [(f"blk.{bid}.ffn_up.weight",   data_torch)]
        if rest == "mlp.up_proj.weight":
            return [(f"blk.{bid}.ffn_up.weight",   data_torch)]
        if rest == "mlp.down_proj.weight":
            return [(f"blk.{bid}.ffn_down.weight", data_torch)]
        # Alternative HF naming (fc1/fc2):
        if rest == "mlp.fc1.weight":
            return [(f"blk.{bid}.ffn_up.weight",   data_torch)]
        if rest == "mlp.fc2.weight":
            return [(f"blk.{bid}.ffn_down.weight", data_torch)]

        # ── MAMBA LAYERS (type 0, blocks 0/2/4/…/16) ─────────────────────────
        if layer_type == 0:
            if rest == "attn.in_proj.weight":
                # HF: [d_inner+2*d_state+d_inner, n_embd] — ggml reversal → [n_embd, …] ✓
                return [(f"blk.{bid}.ssm_in.weight", data_torch)]

            if rest == "attn.conv1d.weight":
                # HF shape: [d_inner, 1, d_conv].  Squeeze the channel dim → [d_inner, d_conv].
                # ggml reversal → [d_conv, d_inner] which is what llama.cpp expects.
                tensor = data_torch.squeeze(1)
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                return [(f"blk.{bid}.ssm_conv1d.weight", tensor)]

            if rest == "attn.conv1d.bias":
                return [(f"blk.{bid}.ssm_conv1d.bias", data_torch)]

            if rest == "attn.x_proj.weight":
                # HF: [dt_rank+2*d_state, d_inner] — ggml reversal → [d_inner, …] ✓
                return [(f"blk.{bid}.ssm_x.weight", data_torch)]

            if rest == "attn.dt_proj.weight":
                # HF: [d_inner, dt_rank] — ggml reversal → [dt_rank, d_inner] ✓
                return [(f"blk.{bid}.ssm_dt.weight", data_torch)]

            if rest == "attn.dt_proj.bias":
                return [(f"blk.{bid}.ssm_dt.bias", data_torch)]

            if rest == "attn.A_log":
                # Convert A_log → A = -exp(A_log).
                # HF: [d_inner, d_state] — ggml reversal → [d_state, d_inner] ✓
                A = -torch.exp(data_torch.float())
                return [(f"blk.{bid}.ssm_a", A)]

            if rest == "attn.D":
                # 1-D vector [d_inner] — no reshape needed.
                return [(f"blk.{bid}.ssm_d", data_torch)]

            if rest == "attn.out_proj.weight":
                # HF: [n_embd, d_inner] — ggml reversal → [d_inner, n_embd] ✓
                return [(f"blk.{bid}.ssm_out.weight", data_torch)]

        # ── GMU LAYERS (type 3, blocks 18/20/…/30) ───────────────────────────
        elif layer_type == 3:
            if rest == "attn.in_proj.weight":
                # HF: [2*n_embd, n_embd] — ggml reversal → [n_embd, 2*n_embd] ✓
                return [(f"blk.{bid}.gmu_in.weight",  data_torch)]

            if rest == "attn.out_proj.weight":
                # HF: [n_embd, 2*n_embd] — ggml reversal → [2*n_embd, n_embd] ✓
                return [(f"blk.{bid}.gmu_out.weight", data_torch)]

        # ── ATTENTION LAYERS (types 1, 2, 4) ─────────────────────────────────
        elif layer_type in (1, 2, 4):
            if rest == "attn.Wqkv.weight":
                w          = data_torch
                head_dim   = 64
                n_q_heads  = 40
                n_kv_heads = 20
                q_size     = n_q_heads  * head_dim   # 2560
                k_size     = n_kv_heads * head_dim   # 1280

                def reorder_heads(block: Tensor, n_heads: int) -> Tensor:
                    """Interleave even/odd differential-attention heads."""
                    h = block.reshape(n_heads, head_dim, block.shape[-1])
                    return torch.cat([h[0::2], h[1::2]], dim=0).reshape(n_heads * head_dim, block.shape[-1]).contiguous()

                if w.shape[0] == q_size:
                    # Cross-attention (type 4): Q projection only — no K/V block.
                    result = reorder_heads(w, n_q_heads)
                else:
                    # SWA (type 1) and full-attn (type 2): fused Q|K|V
                    q_block = w[:q_size]
                    k_block = w[q_size : q_size + k_size]
                    v_block = w[q_size + k_size :]
                    result = torch.cat([
                        reorder_heads(q_block, n_q_heads),
                        reorder_heads(k_block, n_kv_heads),
                        v_block,      # V is NOT reordered
                    ], dim=0).contiguous()

                return [(f"blk.{bid}.attn_qkv.weight", result)]

            if rest == "attn.Wqkv.bias":
                b          = data_torch
                head_dim   = 64
                n_q_heads  = 40
                n_kv_heads = 20
                q_size     = n_q_heads  * head_dim
                k_size     = n_kv_heads * head_dim

                def reorder_bias(block: Tensor, n_heads: int) -> Tensor:
                    h = block.reshape(n_heads, head_dim)
                    return torch.cat([h[0::2], h[1::2]], dim=0).reshape(-1).contiguous()

                if b.shape[0] == q_size:
                    result = reorder_bias(b, n_q_heads)
                else:
                    result = torch.cat([
                        reorder_bias(b[:q_size],              n_q_heads),
                        reorder_bias(b[q_size : q_size + k_size], n_kv_heads),
                        b[q_size + k_size :],   # V bias NOT reordered
                    ], dim=0).contiguous()

                return [(f"blk.{bid}.attn_qkv.bias", result)]

            # Lambda vectors (1-D, no reshape)
            if rest == "attn.inner_cross_attn.lambda_q1":
                return [(f"blk.{bid}.attn_lambda_q1.weight", data_torch)]
            if rest == "attn.inner_cross_attn.lambda_q2":
                return [(f"blk.{bid}.attn_lambda_q2.weight", data_torch)]
            if rest == "attn.inner_cross_attn.lambda_k1":
                return [(f"blk.{bid}.attn_lambda_k1.weight", data_torch)]
            if rest == "attn.inner_cross_attn.lambda_k2":
                return [(f"blk.{bid}.attn_lambda_k2.weight", data_torch)]

            # SubLN scale (1-D, no reshape)
            if rest == "attn.inner_cross_attn.subln.weight":
                return [(f"blk.{bid}.attn_subln.weight", data_torch)]

            # Output projection
            if rest == "attn.out_proj.weight":
                return [(f"blk.{bid}.attn_output.weight", data_torch)]
            if rest == "attn.out_proj.bias":
                return [(f"blk.{bid}.attn_output.bias",   data_torch)]

        # Anything not explicitly handled above falls through to the base class,
        # which applies the generic tensor-name map.
        return super().modify_tensors(data_torch, name, bid)
