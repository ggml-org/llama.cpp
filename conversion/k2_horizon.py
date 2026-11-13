from __future__ import annotations

import re
from collections.abc import Iterable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("K2HorizonForCausalLM")
@ModelBase.example("IFM/K2-Horizon-0.9B", "IFM/K2-Horizon-36B")
class K2HorizonModel(TextModel):
    model_arch = gguf.MODEL_ARCH.K2HORIZON

    _experts: list[dict[str, Tensor]] | None = None

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_group_norm_groups(int(hparams.get("layernorm_num_groups", 1)))
        if (rope_head_dim := hparams.get("rope_head_dim")) is not None:
            self.gguf_writer.add_rope_dimension_count(int(rope_head_dim))

        if int(hparams.get("num_experts", 0)) > 0:
            n_ff_exp = int(hparams["moe_intermediate_size"])
            n_shared = int(hparams.get("num_shared_experts", 0))

            # the leading dense layers are the prefix of mlp_only_layers, unless given explicitly
            n_dense = hparams.get("num_dense_layers")
            if n_dense is None:
                mlp_only_layers = {int(il) for il in hparams.get("mlp_only_layers", [])}
                n_dense = 0
                while n_dense in mlp_only_layers:
                    n_dense += 1

            self.gguf_writer.add_expert_feed_forward_length(n_ff_exp)
            self.gguf_writer.add_leading_dense_block_count(n_dense)
            self.gguf_writer.add_moe_every_n_layers(int(hparams.get("decoder_sparse_step", 1)))
            self.gguf_writer.add_expert_shared_count(n_shared)
            self.gguf_writer.add_expert_weights_norm(bool(hparams.get("norm_topk_prob", False)))
            if n_shared > 0:
                self.gguf_writer.add_expert_shared_feed_forward_length(n_ff_exp * n_shared)
            if (router_scale := hparams.get("router_scaling_factor")) is not None:
                self.gguf_writer.add_expert_weights_scale(float(router_scale))

        # MoVA
        n_value_expert      = int(hparams.get("mova_num_experts", 0))
        n_value_expert_used = int(hparams.get("mova_num_experts_per_tok", 0))
        if n_value_expert > 0 and n_value_expert_used > 0:
            assert n_value_expert_used <= n_value_expert
            self.gguf_writer.add_attention_value_expert_count(n_value_expert)
            self.gguf_writer.add_attention_value_expert_used_count(n_value_expert_used)

        if (gate_func := hparams.get("attention_gate_func")) not in (None, "softplus"):
            raise ValueError(f"Unsupported attention_gate_func: {gate_func!r}")

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # the MoE router bias only selects experts
        if name.endswith(".mlp.gate.bias"):
            assert bid is not None
            yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, bid, ".bias"), data_torch
            return

        if re.fullmatch(r"model\.layers\.\d+\.mlp\.experts\.\d+\.(down|gate|up)_proj\.weight", name):
            yield from self._stack_experts(data_torch, name, bid, int(self.hparams["num_experts"]),
                                           "model.layers.{bid}.mlp.experts.{xid}.{w}.weight", ("down_proj", "gate_proj", "up_proj"))
            return

        if re.fullmatch(r"model\.layers\.\d+\.self_attn\.v_experts\.\d+\.weight", name):
            yield from self._stack_experts(data_torch, name, bid, int(self.hparams["mova_num_experts"]),
                                           "model.layers.{bid}.self_attn.v_experts.{xid}{w}.weight", ("",))
            return

        yield from super().modify_tensors(data_torch, name, bid)

    # collect the per-expert weights of a layer, then emit one stacked 3D tensor per projection
    def _stack_experts(self, data_torch: Tensor, name: str, bid: int | None, n_experts: int,
                       fmt: str, projs: tuple[str, ...]) -> Iterable[tuple[str, Tensor]]:
        assert bid is not None
        if self._experts is None:
            self._experts = [{} for _ in range(self.block_count)]
        self._experts[bid][name] = data_torch

        names = {w: [fmt.format(bid=bid, xid=xid, w=w) for xid in range(n_experts)] for w in projs}
        if not all(n in self._experts[bid] for ns in names.values() for n in ns):
            return

        for w, ns in names.items():
            merged = torch.stack([self._experts[bid].pop(n) for n in ns], dim=0)
            yield from super().modify_tensors(merged, fmt.replace(".{xid}", "").format(bid=bid, w=w), bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten the list of dicts
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
