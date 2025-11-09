from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor
from .llama import LlamaModel


@ModelBase.register("OlmoForCausalLM")
@ModelBase.register("OLMoForCausalLM")
class OlmoModel(TextModel):
    model_arch = gguf.MODEL_ARCH.OLMO

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_layer_norm_eps(1e-5)
        clip_qkv = self.hparams.get("clip_qkv")
        if clip_qkv is not None:
            self.gguf_writer.add_clamp_kqv(clip_qkv)

    # Same as super class, but permuting q_proj, k_proj
    # Copied from: LlamaModel
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        n_head = self.hparams["num_attention_heads"]
        n_kv_head = self.hparams.get("num_key_value_heads")
        if name.endswith("q_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_head)
        if name.endswith("k_proj.weight"):
            data_torch = LlamaModel.permute(data_torch, n_head, n_kv_head)
        return [(self.map_tensor_name(name), data_torch)]


@ModelBase.register("SeedOssForCausalLM")
class SeedOssModel(TextModel):
    model_arch = gguf.MODEL_ARCH.SEED_OSS


@ModelBase.register("Olmo3ForCausalLM")
@ModelBase.register("Olmo2ForCausalLM", "Olmo3ForCausalLM")
class Olmo2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.OLMO2

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_attn_factors(rope_scaling["attention_factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling["original_max_position_embeddings"])
        if "sliding_window" in self.hparams:
            self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
            sliding_window_pattern = []
            if "layer_types" in self.hparams:
                sliding_window_pattern = [t == "sliding_attention" for t in self.hparams["layer_types"]]
            else:
                # Olmo2 does not use sliding window attention.
                # Olmo3 defaults to using sliding window for all layers except every 4th.
                for i in range(self.hparams["num_hidden_layers"]):
                    sliding_window_pattern.append((i + 1) % 4 != 0)
            self.gguf_writer.add_sliding_window_pattern(sliding_window_pattern)


@ModelBase.register("OlmoeForCausalLM")
class OlmoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.OLMOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_layer_norm_rms_eps(1e-5)
        if (n_experts := self.hparams.get("num_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
    _experts: list[dict[str, Tensor]] | None = None

    # Copied from: Qwen2MoeModel
    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find("experts") != -1:
            n_experts = self.hparams["num_experts"]
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []
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
            else:
                return []
        return [(self.map_tensor_name(name), data_torch)]

    # Copied from: Qwen2MoeModel
    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
