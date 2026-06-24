from __future__ import annotations

from .base import ModelBase, gguf

from .llama import LlamaModel


@ModelBase.register(
    "modeling_sparsetral.MistralForCausalLM")
class SparsetralModel(LlamaModel):
    model_arch = gguf.MODEL_ARCH.LLAMA
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if "topk" in self.hparams:
            self.hparams["num_experts_per_tok"] = self.hparams["topk"]

        self.hparams["scoring_func"] = "softmax"
    