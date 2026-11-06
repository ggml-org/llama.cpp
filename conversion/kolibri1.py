from __future__ import annotations

from .afmoe import AfmoeModel
from .base import ModelBase, gguf


@ModelBase.register("Kolibri1ForCausalLM")
@ModelBase.example("Aleph-Alpha/Kolibri-1")
class Kolibri1Model(AfmoeModel):
    model_arch = gguf.MODEL_ARCH.KOLIBRI1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # AfmoeModel reads these keys, this config stores the same values under other names
        self.hparams["num_shared_experts"] = 1
        self.hparams["route_norm"] = self.hparams["norm_topk_prob"]
        self.hparams["intermediate_size"] = self.hparams["moe_intermediate_size"]

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        self.gguf_writer.add_expert_shared_feed_forward_length(self.hparams["shared_expert_intermediate_size"])
        self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in self.hparams["layer_types"]])
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
