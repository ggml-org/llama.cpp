from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MmprojModel, ModelBase, gguf

from .llama import LlamaModel


@ModelBase.register("CogVLMForCausalLM")
class CogVLMVisionModel(MmprojModel):

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_vision_attention_layernorm_eps(self.hparams.get("layer_norm_eps", 1e-6))
        self.gguf_writer.add_clip_projector_type(gguf.VisionProjectorType.COGVLM)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if not name.startswith("model.vision."):
            return

        yield from super().modify_tensors(data_torch, name, bid)


@ModelBase.register("CogVLMForCausalLM")
class CogVLMModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.COGVLM

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # block vision tensors
        if name.startswith("model.vision."):
            return

        yield from ModelBase.modify_tensors(self, data_torch, name, bid)
