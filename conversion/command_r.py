from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass


@ModelBase.register("CohereForCausalLM")
class CommandR2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.COMMAND_R

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # max_position_embeddings = 8192 in config.json but model was actually
        # trained on 128k context length
        # aya-23 models don't have model_max_length specified
        self.hparams["max_position_embeddings"] = self.find_hparam(["model_max_length", "max_position_embeddings"])

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)


@ModelBase.register("Cohere2ForCausalLM")
class Cohere2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.COHERE2

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_logit_scale(self.hparams["logit_scale"])
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_vocab_size(self.hparams["vocab_size"])
        rotary_pct = self.hparams["rotary_pct"]
        hidden_size = self.hparams["hidden_size"]
        num_attention_heads = self.hparams["num_attention_heads"]
        self.gguf_writer.add_rope_dimension_count(int(rotary_pct * (hidden_size // num_attention_heads)))
        self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)
