from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass


@ModelBase.register("GPTBigCodeForCausalLM")
class StarCoderModel(TextModel):
    model_arch = gguf.MODEL_ARCH.STARCODER

    def set_gguf_parameters(self):
        block_count = self.hparams["n_layer"]
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


@ModelBase.register("Starcoder2ForCausalLM")
class StarCoder2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.STARCODER2
