from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("MiniMaxM2ForCausalLM")
class MiniMaxM2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MINIMAXM2
    _experts_cache: dict[int, dict[str, Tensor]] = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams["num_experts"] = self.hparams["num_local_experts"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if self.hparams["scoring_func"] == "sigmoid":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
        elif self.hparams["scoring_func"] == "softmax":
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        else:
            raise ValueError(f"Unsupported scoring_func value: {self.hparams['scoring_func']}")
        self.gguf_writer.add_expert_feed_forward_length(self.find_hparam(["intermediate_size"]))
        self.gguf_writer.add_rope_dimension_count(self.find_hparam(["rotary_dim"]))

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")
        # merge expert weights
        if 'experts' in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None
            expert_cache = self._experts_cache.setdefault(bid, {})
            expert_cache[name] = data_torch
            expert_weights = ["w1", "w2", "w3"]
            # not enough expert weights to merge
            if len(expert_cache) < n_experts * len(expert_weights):
                return []
            tensors: list[tuple[str, Tensor]] = []
            for w_name in expert_weights:
                datas: list[Tensor] = []
                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.block_sparse_moe.experts.{xid}.{w_name}.weight"
                    datas.append(expert_cache[ename])
                    del expert_cache[ename]
                data_torch = torch.stack(datas, dim=0)
                merged_name = f"model.layers.{bid}.block_sparse_moe.experts.{w_name}.weight"
                new_name = self.map_tensor_name(merged_name)
                tensors.append((new_name, data_torch))
            del self._experts_cache[bid]
            return tensors
        return super().modify_tensors(data_torch, name, bid)
