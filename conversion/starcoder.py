from __future__ import annotations

from .base import ModelBase, TextModel, gguf


@ModelBase.register("GPTBigCodeForCausalLM")
@ModelBase.example("bigcode/gpt_bigcode-santacoder")
class StarCoderModel(TextModel):
    model_arch = gguf.MODEL_ARCH.STARCODER

    def set_gguf_parameters(self):
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(1)
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)


@ModelBase.register("Starcoder2ForCausalLM")
@ModelBase.example("bigcode/starcoder2-3b")
class StarCoder2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.STARCODER2
