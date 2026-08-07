from __future__ import annotations

from .base import ModelBase, TextModel, gguf


@ModelBase.register("CodeShellForCausalLM")
class CodeShellModel(TextModel):
    model_arch = gguf.MODEL_ARCH.CODESHELL

    def set_gguf_parameters(self):
        self.gguf_writer.add_context_length(self.hparams["n_positions"])
        self.gguf_writer.add_embedding_length(self.hparams["n_embd"])
        self.gguf_writer.add_feed_forward_length(4 * self.hparams["n_embd"])
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_head_count(self.hparams["n_head"])
        self.gguf_writer.add_head_count_kv(self.hparams["num_query_groups"])
        self.gguf_writer.add_layer_norm_eps(self.hparams["layer_norm_epsilon"])
        self.gguf_writer.add_file_type(self.ftype)
        self.gguf_writer.add_rope_freq_base(10000.0)
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.LINEAR)
        self.gguf_writer.add_rope_scaling_factor(1.0)
