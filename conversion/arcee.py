from __future__ import annotations
from .base import (
    ModelBase, gguf
)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    pass
from .llama import LlamaModel


@ModelBase.register("ArceeForCausalLM")
class ArceeModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.ARCEE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self._try_set_pooling_type()
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling["original_max_position_embeddings"])
