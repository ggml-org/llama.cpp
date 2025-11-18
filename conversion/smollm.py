from __future__ import annotations
from .base import (
    ModelBase, gguf
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass
from .llama import LlamaModel


@ModelBase.register("SmolLM3ForCausalLM")
class SmolLM3Model(LlamaModel):
    model_arch = gguf.MODEL_ARCH.SMOLLM3
