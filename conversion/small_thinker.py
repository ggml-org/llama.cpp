from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch, logger
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("SmallThinkerForCausalLM")
class SmallThinkerModel(TextModel):
    model_arch = gguf.MODEL_ARCH.SMALLTHINKER

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (n_experts := self.hparams.get("num_experts", self.hparams.get("moe_num_primary_experts"))) is not None:
            self.gguf_writer.add_expert_count(n_experts)
        if (n_experts_used := self.hparams.get("num_experts_per_tok", self.hparams.get("moe_num_active_primary_experts"))) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
        if (moe_intermediate_size := self.hparams.get("moe_ffn_hidden_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            self.gguf_writer.add_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")
        if (self.hparams.get('moe_primary_router_apply_softmax')):
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        else:
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        # YaRN is not enabled by default
        # To enable it, please refer to this guide: https://huggingface.co/Qwen/Qwen3-30B-A3B#processing-long-texts
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling["original_max_position_embeddings"])
        sliding_window_layout = self.hparams.get("sliding_window_layout")
        if sliding_window_layout:
            for i in sliding_window_layout:
                if i != 0:
                    sliding_window = self.hparams.get("sliding_window_size")
                    if sliding_window:
                        self.gguf_writer.add_sliding_window(sliding_window)
                    break
    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # process the experts separately
        if name.find("experts") != -1:
            n_experts = self.hparams.get("num_experts", self.hparams.get("moe_num_primary_experts"))
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []
                # merge the experts into a single 3d tensor
                for w_name in ["down", "gate", "up"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.block_sparse_moe.experts.{w_name}.weight"
                    new_name = self.map_tensor_name(merged_name)
                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []
        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
