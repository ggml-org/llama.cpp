from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("BitnetForCausalLM")
class BitnetModel(TextModel):
    model_arch = gguf.MODEL_ARCH.BITNET

    def set_vocab(self):
        self._set_vocab_sentencepiece()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        self.gguf_writer.add_rope_scaling_factor(1.0)

    def weight_quant(self, weight: Tensor) -> Tensor:
        dtype = weight.dtype
        weight = weight.float()
        scale = weight.abs().mean().clamp(min=1e-5)
        iscale = 1 / scale
        # TODO: multiply by the scale directly instead of inverting it twice
        # (this is also unnecessarily doubly inverted upstream)
        # ref: https://huggingface.co/1bitLLM/bitnet_b1_58-3B/blob/af89e318d78a70802061246bf037199d2fb97020/utils_quant.py#L10
        result = (weight * iscale).round().clamp(-1, 1) / iscale
        return result.type(dtype)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        new_name = self.map_tensor_name(name)
        if any(self.match_model_tensor_name(new_name, key, bid) for key in [
            gguf.MODEL_TENSOR.ATTN_Q,
            gguf.MODEL_TENSOR.ATTN_K,
            gguf.MODEL_TENSOR.ATTN_V,
            gguf.MODEL_TENSOR.ATTN_OUT,
            gguf.MODEL_TENSOR.FFN_UP,
            gguf.MODEL_TENSOR.FFN_DOWN,
            gguf.MODEL_TENSOR.FFN_GATE,
        ]):
            # transform weight into 1/0/-1 (in fp32)
            data_torch = self.weight_quant(data_torch)
        yield (new_name, data_torch)
