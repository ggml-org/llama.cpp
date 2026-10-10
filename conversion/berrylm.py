from __future__ import annotations

import math

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("BerryLMForCausalLM")
@ModelBase.example("rwb-ai/BerryLM-OS")
class BerryLMModel(TextModel):
    model_arch = gguf.MODEL_ARCH.BERRYLM

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hp = self.hparams

        if hp.get("kda_safe_gate", False):
            raise NotImplementedError("BerryLM: kda_safe_gate=True (clamped forget gate) is not supported")
        if not hp.get("attn_res_gated", True):
            raise NotImplementedError("BerryLM: only gated AttnRes is supported")
        if not math.isclose(hp.get("attn_res_eps", 1e-6), hp["rms_norm_eps"]):
            raise NotImplementedError("BerryLM: attn_res_eps must equal rms_norm_eps")
        if not math.isclose(hp["rms_norm_eps"], 1e-6):
            raise NotImplementedError("BerryLM: the KDA q/k l2norm uses eps 1e-6, so rms_norm_eps must be 1e-6")
        if hp["linear_key_head_dim"] != hp["linear_value_head_dim"]:
            raise NotImplementedError("BerryLM: linear_key_head_dim must equal linear_value_head_dim")
        if not hp.get("norm_topk_prob", True):
            raise NotImplementedError("BerryLM: norm_topk_prob=False is not supported")
        if not hp.get("attn_output_gate", True) or hp.get("attention_bias", False):
            raise NotImplementedError("BerryLM: only gated, bias-free full attention is supported")
        if hp.get("hidden_act", "silu") != "silu":
            raise NotImplementedError("BerryLM: only the SiLU activation is supported")

        self.gguf_writer.add_vocab_size(hp["vocab_size"])

        # MoE
        self.gguf_writer.add_expert_feed_forward_length(hp["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hp["shared_expert_intermediate_size"])

        # KDA linear attention
        self.gguf_writer.add_ssm_conv_kernel(hp["linear_conv_kernel_dim"])
        self.gguf_writer.add_ssm_state_size(hp["linear_key_head_dim"])
        self.gguf_writer.add_ssm_group_count(hp["linear_num_key_heads"])
        self.gguf_writer.add_ssm_time_step_rank(hp["linear_num_value_heads"])
        self.gguf_writer.add_ssm_inner_size(hp["linear_value_head_dim"] * hp["linear_num_value_heads"])

        layer_types = hp["layer_types"]
        if any(t not in ("linear_attention", "full_attention") for t in layer_types):
            raise ValueError("BerryLM: layer_types must contain only linear_attention or full_attention")
        self.gguf_writer.add_recurrent_layers([t == "linear_attention" for t in layer_types])

        # partial RoPE on the full-attention layers
        self.gguf_writer.add_rope_dimension_count(int(hp["head_dim"] * self.rope_parameters.get("partial_rotary_factor", 0.25)))

        # low-rank KDA forget-gate projection
        self.gguf_writer.add_kda_gate_rank(hp["kda_gate_bottleneck"])

        # Gated Block AttnRes
        self.gguf_writer.add_attn_res_block_size(int(hp["attn_res_block_size"]))

    # V heads: grouped by K head in HF (repeat_interleave), tiled in ggml broadcast (v_head % n_k_heads).
    # Entries [HV * head_dim] along dim are permuted so that ggml head h reads the HF head it is broadcast against.
    def _v_heads_to_tiled(self, t: Tensor, dim: int, head_dim: int) -> Tensor:
        n_k = self.hparams["linear_num_key_heads"]
        n_v = self.hparams["linear_num_value_heads"]
        if n_k == n_v:
            return t
        idx = torch.arange(t.shape[dim]).reshape(n_k, n_v // n_k, head_dim).transpose(0, 1).reshape(-1)
        # index rather than reshape, so that LoRA tensors are supported too
        return t[(slice(None),) * dim + (idx,)]

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        hp = self.hparams

        n = name.removeprefix(f"model.layers.{bid}.") if bid is not None else name

        head_k = hp["linear_key_head_dim"]
        head_v = hp["linear_value_head_dim"]
        key_dim = head_k * hp["linear_num_key_heads"]

        # RMSNorm with a (1 + w) scale; the gated norm of the linear attention keeps its plain weight
        if n in ("model.norm.weight", "input_layernorm.weight", "post_attention_layernorm.weight",
                 "self_attn.q_norm.weight", "self_attn.k_norm.weight"):
            data_torch = data_torch + 1

        # Gated Block AttnRes: tanh(gate) is folded at conversion time
        elif n == "attn_res.pseudo_query":
            name += ".weight"
        elif n == "attn_res.gate":
            name += ".weight"
            data_torch = torch.tanh(data_torch.float()).reshape(1)

        # KDA linear attention
        elif n == "kda.in_proj_qkv.weight":
            qk, v = data_torch[:2 * key_dim], data_torch[2 * key_dim:]
            data_torch = torch.cat([qk, self._v_heads_to_tiled(v, 0, head_v)], dim=0)
        elif n == "kda.in_proj_z.weight":
            data_torch = self._v_heads_to_tiled(data_torch, 0, head_v)
        elif n == "kda.in_proj_b.weight":
            data_torch = self._v_heads_to_tiled(data_torch, 0, 1)
        elif n == "kda.conv1d.weight":
            w = data_torch.reshape(data_torch.shape[0], -1)  # [channels, 1, k] -> [channels, k]
            qk, v = w[:2 * key_dim], w[2 * key_dim:]
            data_torch = torch.cat([qk, self._v_heads_to_tiled(v, 0, head_v)], dim=0)
        elif n == "kda.f_up_proj.weight":
            # per-channel gate: rows are [HV * K], permuted in blocks of K
            data_torch = self._v_heads_to_tiled(data_torch, 0, head_k)
        elif n == "kda.dt_bias":
            name += ".bias"
            data_torch = self._v_heads_to_tiled(data_torch, 0, head_k)
        elif n == "kda.A_log":
            data_torch = self._v_heads_to_tiled(-torch.exp(data_torch.float()), 0, 1)
        elif n == "kda.out_proj.weight":
            data_torch = self._v_heads_to_tiled(data_torch, 1, head_v)

        # sparse MoE + gated shared expert
        elif n == "mlp.experts.gate_up_proj":
            # [n_expert, 2 * n_ff, n_embd], gate first
            n_ff = hp["moe_intermediate_size"]
            base = name.removesuffix(".gate_up_proj")
            yield from super().modify_tensors(data_torch[:, :n_ff, :].contiguous(), f"{base}.gate_proj.weight", bid)
            yield from super().modify_tensors(data_torch[:, n_ff:, :].contiguous(), f"{base}.up_proj.weight", bid)
            return
        elif n == "mlp.experts.down_proj":
            name += ".weight"
        elif n == "mlp.shared_expert_gate.weight":
            data_torch = data_torch.reshape(-1)

        yield from super().modify_tensors(data_torch, name, bid)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # the KDA decay gate projections are small and precision-sensitive: keep them in F32
        if new_name.endswith((".ssm_f_a.weight", ".ssm_f_b.weight")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)
