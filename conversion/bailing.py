from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("BailingMoeForCausalLM")
class BailingMoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.BAILINGMOE

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(rope_dim)
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling["original_max_position_embeddings"])
        else:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_weights_scale(1.0)
        self.gguf_writer.add_expert_count(hparams["num_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["num_shared_experts"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])
    _experts: list[dict[str, Tensor]] | None = None

    @staticmethod
    def permute(weights: Tensor, n_head: int, n_head_kv: int | None):
        if n_head_kv is not None and n_head != n_head_kv:
            n_head = n_head_kv
        return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
                .swapaxes(1, 2)
                .reshape(weights.shape))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")
        n_embd = self.hparams["hidden_size"]
        if (head_dim := self.hparams.get("head_dim")) is None:
            head_dim = n_embd // n_head
        output_name = self.format_tensor_name(gguf.MODEL_TENSOR.OUTPUT)
        if name.endswith("attention.dense.weight"):
            return [(self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_OUT, bid), data_torch)]
        elif name.endswith("query_key_value.weight"):
            q, k, v = data_torch.split([n_head * head_dim, n_kv_head * head_dim, n_kv_head * head_dim], dim=-2)
            return [
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_Q, bid), BailingMoeModel.permute(q, n_head, n_head)),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K, bid), BailingMoeModel.permute(k, n_head, n_kv_head)),
                (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V, bid), v)
            ]
        elif name.find("mlp.experts") != -1:
            n_experts = self.hparams["num_experts"]
            assert bid is not None
            tensors: list[tuple[str, Tensor]] = []
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    new_name = self.map_tensor_name(merged_name)
                    tensors.append((new_name, data_torch))
            return tensors
        new_name = self.map_tensor_name(name)
        if new_name == output_name and self.hparams.get("norm_head"):
            data_torch = data_torch.float()
            data_torch /= torch.norm(data_torch, p=2, dim=0, keepdim=True) + 1e-7
        return [(new_name, data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")


@ModelBase.register("BailingMoeV2ForCausalLM")
class BailingMoeV2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.BAILINGMOE2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if nextn_layers := self.hparams.get("num_nextn_predict_layers", 0):
            self.block_count = self.hparams["num_hidden_layers"] + nextn_layers
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        if (rope_dim := hparams.get("head_dim")) is None:
            rope_dim = hparams["hidden_size"] // hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rope_dim * self.hparams.get("partial_rotary_factor", 0.5)))
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling["original_max_position_embeddings"])
        else:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hparams.get("moe_shared_expert_intermediate_size", hparams["moe_intermediate_size"] * hparams["num_shared_experts"]))
        self.gguf_writer.add_expert_weights_scale(hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_count(hparams["num_experts"])
        self.gguf_writer.add_expert_shared_count(hparams["num_shared_experts"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])
        if hparams["score_function"] == "sigmoid":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        elif hparams["score_function"] == "softmax":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        else:
            raise ValueError(f"Unsupported score_function value: {hparams['score_function']}")
        if (nextn_layers := self.hparams.get("num_nextn_predict_layers")) is not None:
            self.gguf_writer.add_nextn_predict_layers(nextn_layers)
    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if "mlp.experts" in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None
            tensors: list[tuple[str, Tensor]] = []
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    new_name = self.map_tensor_name(merged_name)
                    tensors.append((new_name, data_torch))
            return tensors
        if name.endswith(".expert_bias"):
            name = name.replace(".expert_bias", ".expert_bias.bias")
        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
