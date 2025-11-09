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

    def set_vocab(self):
        super().set_vocab()
        # remove unsupported array slicing in chat template
        # ref: https://huggingface.co/ggml-org/SmolLM3-3B-GGUF/discussions/1
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.dir_model)
        if tokenizer.chat_template is not None:
            chat_template = tokenizer.chat_template.replace("[:]", "")
            self.gguf_writer.add_chat_template(chat_template)
