from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf

from .qwen import Qwen2MoeModel


@ModelBase.register("Dots1ForCausalLM")
class Dots1Model(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.DOTS1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hparams["num_experts"] = self.hparams["n_routed_experts"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_leading_dense_block_count(self.hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_shared_count(self.hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_scale(self.hparams["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(self.hparams["norm_topk_prob"])

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        if "shared_experts" in name:
            yield from ModelBase.modify_tensors(self, data_torch, name, bid)
        else:
            yield from super().modify_tensors(data_torch, name, bid)
