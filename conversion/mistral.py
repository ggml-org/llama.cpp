from __future__ import annotations
from .base import (
    gguf, _mistral_import_error_msg
)
from pathlib import Path
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass
from .llama import LlamaModel


class MistralModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA
    model_name = "Mistral"
    hf_arch = ""
    is_mistral_format = True
    undo_permute = False