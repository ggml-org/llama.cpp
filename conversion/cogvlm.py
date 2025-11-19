from __future__ import annotations
from .base import (
    ModelBase, MmprojModel, gguf
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor
from .llama import LlamaModel


@ModelBase.register("CogVLMForCausalLM")
class CogVLMVisionModel(MmprojModel):
    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams.get("layer_norm_eps", 1e-6))
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.COGVLM)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        if not name.startswith("model.vision."):
            return []
        return [(self.map_tensor_name(name), data_torch)]


@ModelBase.register("CogVLMForCausalLM")
class CogVLMModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.COGVLM

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        del bid  # unused
        # block vision tensors
        if name.startswith("model.vision."):
            return []
        return [(self.map_tensor_name(name), data_torch)]
