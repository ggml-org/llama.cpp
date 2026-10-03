from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .llama import LlamaModel


@ModelBase.register("ETETMOE_LLAMA")
@ModelBase.example("RheoEcho/ETET-1.0-24E-1.8B-A1B-Preview")
class ETETModel(LlamaModel):
    # llama backbone with dense FFN in the leading layers and top-1 MoE in the rest
    model_arch = gguf.MODEL_ARCH.ETETMOE_LLAMA

    _experts: list[dict[str, Tensor]] | None = None

    def set_vocab(self):
        # the full vocab incl. added tokens lives in tokenizer.json; skip the sentencepiece path
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_expert_count(hparams["etet_num_experts"])
        self.gguf_writer.add_expert_used_count(hparams["etet_top_k"])
        # ETET uses the same FFN width for dense and expert layers
        self.gguf_writer.add_expert_feed_forward_length(hparams["intermediate_size"])
        self.gguf_writer.add_leading_dense_block_count(hparams["etet_moe_start_layer"])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        # top-1 softmax routing normalizes the expert weight to 1.0
        # the llama.cpp default scale is 0.0, which would zero the MoE output
        self.gguf_writer.add_expert_weights_scale(1.0)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # merge per-expert tensors into packed 3d expert tensors
        if ".mlp.experts." in name:
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch

            n_experts = self.hparams["etet_num_experts"]
            if len(self._experts[bid]) < n_experts * 3:
                return

            for wid in ["gate_proj", "up_proj", "down_proj"]:
                datas: list[Tensor] = []
                for xid in range(n_experts):
                    ename = f"model.layers.{bid}.mlp.experts.{xid}.{wid}.weight"
                    datas.append(self._experts[bid][ename])
                    del self._experts[bid][ename]
                merged = f"model.layers.{bid}.mlp.experts.{wid}.weight"
                yield from super().modify_tensors(torch.stack(datas, dim=0), merged, bid)
            return

        # the router weight is stored under mlp.router.linear; drop the module suffix
        if ".mlp.router.linear." in name:
            name = name.replace(".mlp.router.linear.", ".mlp.router.")

        yield from super().modify_tensors(data_torch, name, bid)
