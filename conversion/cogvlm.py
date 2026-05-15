from __future__ import annotations

from typing import Callable, TYPE_CHECKING

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

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if not name.startswith("model.vision."):
            return None

        return super().filter_tensors(item)


@ModelBase.register("CogVLMForCausalLM")
class CogVLMModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.COGVLM
