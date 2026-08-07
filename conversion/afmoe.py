from __future__ import annotations

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf

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

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Handle expert weights - they're already merged in the HF format
        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["gate_proj", "up_proj", "down_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename_to_retrieve = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename_to_retrieve])
                        del self._experts[bid][ename_to_retrieve]

                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    yield from ModelBase.modify_tensors(self, data_torch, merged_name, bid)

                return
            else:
                return

        yield from ModelBase.modify_tensors(self, data_torch, name, bid)
