from __future__ import annotations

from .base import ModelBase, TextModel, gguf


@ModelBase.register("MaincoderForCausalLM")
class MaincoderModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MAINCODER

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        if (head_dim := self.hparams.get("head_dim")) is not None:
            self.gguf_writer.add_rope_dimension_count(head_dim)
