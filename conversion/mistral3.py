from __future__ import annotations
from .base import (
    ModelBase, gguf
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch import Tensor
from .llama import LlamaModel


@ModelBase.register("Mistral3ForConditionalGeneration")
class Mistral3Model(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None):
        name = name.replace("language_model.", "")
        if "multi_modal_projector" in name or "vision_tower" in name:
            return []
        return super().modify_tensors(data_torch, name, bid)
