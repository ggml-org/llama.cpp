from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("BailingMoeV3ForCausalLM")
class BailingMoeV3Model(TextModel):
    model_arch = gguf.MODEL_ARCH.BAILINGMOE3

    _experts: list[dict[str, Tensor]] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if nextn_layers := self.hparams.get("num_nextn_predict_layers", 0):
            self.block_count = self.hparams["num_hidden_layers"] + nextn_layers
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def is_full_attention(self, bid: int) -> bool:
        n_layer = self.hparams["num_hidden_layers"]
        layer_group_size = self.hparams["layer_group_size"]
        return bid >= n_layer or (bid + 1) % layer_group_size == 0 or bid >= n_layer // layer_group_size * layer_group_size

    def set_gguf_parameters(self):
        if not self.hparams.get("no_kda_lora", False):
            raise ValueError("BailingMoeV3 KDA LoRA projections are not supported")
        if not self.hparams.get("kda_safe_gate", False):
            raise ValueError("BailingMoeV3 non-safe KDA gates are not supported")
        if self.hparams.get("gated_attention_proj_granularity_type") != "head_wise":
            raise ValueError("BailingMoeV3 requires head-wise attention gates")

        self.hparams["num_key_value_heads"] = 1
        super().set_gguf_parameters()

        n_head_kv = [1 if self.is_full_attention(il) else 0 for il in range(self.block_count)]
        self.gguf_writer.add_head_count_kv(n_head_kv)

        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        self.gguf_writer.add_ssm_conv_kernel(self.hparams["short_conv_kernel_size"])
        self.gguf_writer.add_kda_head_dim(self.hparams["head_dim"])
        self.gguf_writer.add_kda_safe_gate(self.hparams["kda_safe_gate"])
        self.gguf_writer.add_kda_gate_lower_bound(self.hparams["kda_lower_bound"])

        kv_lora_rank = self.hparams["kv_lora_rank"]
        qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
        qk_rope_head_dim = self.hparams["qk_rope_head_dim"]
        self.gguf_writer.add_kv_lora_rank(kv_lora_rank)
        self.gguf_writer.add_rope_dimension_count(qk_rope_head_dim)
        self.gguf_writer.add_key_length(kv_lora_rank + qk_rope_head_dim)
        self.gguf_writer.add_key_length_mla(qk_nope_head_dim + qk_rope_head_dim)
        self.gguf_writer.add_value_length_mla(self.hparams["v_head_dim"])

        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(self.hparams["moe_shared_expert_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(self.hparams["num_shared_experts"])
        self.gguf_writer.add_leading_dense_block_count(self.hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_weights_scale(self.hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(self.hparams["norm_topk_prob"])

        def clamp_limits(key: str) -> list[float] | None:
            values = self.hparams.get(key)
            if values is None:
                return None
            values = [0.0 if value is None else float(value) for value in values[:self.block_count]]
            return values + [0.0] * (self.block_count - len(values))

        if (values := clamp_limits("expert_swiglu_limit_list")) is not None:
            self.gguf_writer.add_swiglu_clamp_exp(values)
        if (values := clamp_limits("share_expert_swiglu_limit_list")) is not None:
            self.gguf_writer.add_swiglu_clamp_shexp(values)

        if nextn_layers := self.hparams.get("num_nextn_predict_layers", 0):
            self.gguf_writer.add_nextn_predict_layers(nextn_layers)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.endswith(".expert_bias"):
            name += ".bias"
        return super().filter_tensors((name, gen))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")) and data_torch.ndim in (2, 3):
            d_inner = data_torch.shape[0]
            d_conv = data_torch.shape[-1]
            data_torch = data_torch.reshape(1, d_inner, 1, d_conv)

        if name.endswith(".A_log"):
            data_torch = torch.exp(data_torch).reshape(-1, 1)

        if name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"

        if name.endswith(".attention.f_proj.weight"):
            assert bid is not None
            name = self.format_tensor_name(gguf.MODEL_TENSOR.SSM_F_A, bid)

        if name.endswith(".attention.g_proj.weight"):
            assert bid is not None
            tensor = gguf.MODEL_TENSOR.ATTN_GATE if self.is_full_attention(bid) else gguf.MODEL_TENSOR.SSM_G_A
            name = self.format_tensor_name(tensor, bid)

        if ".mlp.experts." in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                for weight_name in ("down_proj", "gate_proj", "up_proj"):
                    tensors = []
                    for expert_id in range(n_experts):
                        expert_name = f"model.layers.{bid}.mlp.experts.{expert_id}.{weight_name}.weight"
                        tensors.append(self._experts[bid].pop(expert_name))
                    merged_name = f"model.layers.{bid}.mlp.experts.{weight_name}.weight"
                    yield from super().modify_tensors(torch.stack(tensors, dim=0), merged_name, bid)
            return

        if name.endswith(".attention.kv_b_proj.weight"):
            assert bid is not None
            n_head = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
            assert data_torch.shape[0] == n_head * (v_head_dim + qk_nope_head_dim)
            kv_b = data_torch.view(n_head, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            name_k = self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K_B, bid)
            name_v = self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V_B, bid)
            yield from super().modify_tensors(k_b.transpose(1, 2), name_k, bid)
            yield from super().modify_tensors(v_b, name_v, bid)
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            experts = [name for layer in self._experts for name in layer]
            if experts:
                raise ValueError(f"Unprocessed experts: {experts}")
