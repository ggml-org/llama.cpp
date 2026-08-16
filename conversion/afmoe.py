from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MOE_HF_MLP, ModelBase, gguf

from .llama import LlamaModel


@ModelBase.register("AfmoeForCausalLM")
class AfmoeModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.AFMOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        # MoE parameters
        if (n_shared_experts := self.hparams.get("num_shared_experts")) is not None:
            self.gguf_writer.add_expert_shared_count(n_shared_experts)
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
        if (n_dense_layers := self.hparams.get("num_dense_layers")) is not None:
            self.gguf_writer.add_leading_dense_block_count(n_dense_layers)

        # Route normalization and scaling
        if (route_norm := self.hparams.get("route_norm")) is not None:
            self.gguf_writer.add_expert_weights_norm(route_norm)
        if (route_scale := self.hparams.get("route_scale")) is not None:
            self.gguf_writer.add_expert_weights_scale(route_scale)

        # Sliding window attention
        if (sliding_window := self.hparams.get("sliding_window")) is not None:
            self.gguf_writer.add_sliding_window(sliding_window)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if name.endswith(".expert_bias"):
            name = name.replace(".expert_bias", ".expert_bias.bias")

        return super().filter_tensors((name, gen))

    moe_experts = [MOE_HF_MLP._replace(weights=("gate_proj", "up_proj", "down_proj"))]

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        yield from ModelBase.modify_tensors(self, data_torch, name, bid)
