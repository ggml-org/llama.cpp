from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("MellumForCausalLM")
class MellumModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MELLUM

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")

        use_sliding_window = self.hparams.get("use_sliding_window")
        sliding_window = self.hparams.get("sliding_window")
        if (use_sliding_window is True or use_sliding_window is None) and sliding_window is not None:
            self.gguf_writer.add_sliding_window(sliding_window)
            logger.info(f"gguf: sliding window = {sliding_window}")
            self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in self.hparams["layer_types"]])
            logger.info(f"gguf: sliding window pattern length = {len(self.hparams['layer_types'])}")

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.find("experts") != -1:
            n_experts = self.find_hparam(["num_local_experts", "num_experts"])
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]

                    data_torch = torch.stack(datas, dim=0)

                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return

        yield from super().modify_tensors(data_torch, name, bid)
