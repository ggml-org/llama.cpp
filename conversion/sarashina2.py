from __future__ import annotations

from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf

from .llama import LlamaModel
from .qwenvl import Qwen2VLVisionModel


@ModelBase.register("Sarashina2VisionForCausalLM")
class Sarashina2VLTextModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if name.startswith("llm."):
            name = name.replace("llm.", "", 1)
        elif name.startswith("norm."):
            return None
        return super().filter_tensors((name, gen))


@ModelBase.register("Sarashina2VisionForCausalLM")
class Sarashina2VLVisionModel(Qwen2VLVisionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_config['model_type'] = "qwen2_vl"
