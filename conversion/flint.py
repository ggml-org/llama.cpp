from __future__ import annotations

from .base import ModelBase, gguf
from .qwen import Qwen2MoeModel


@ModelBase.register("FlintForCausalLM")
@ModelBase.example("RecursiveSI/flint")
class FlintModel(Qwen2MoeModel):
    model_arch = gguf.MODEL_ARCH.FLINT

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        dense = self.hparams["mlp_only_layers"]
        if dense != list(range(len(dense))):
            raise ValueError("Flint dense layers must precede the MoE layers")
        self.gguf_writer.add_leading_dense_block_count(len(dense))
        self.gguf_writer.add_expert_shared_count(
            self.hparams["shared_expert_intermediate_size"] // self.hparams["moe_intermediate_size"]
        )
        self.gguf_writer.add_expert_weights_norm(self.hparams["norm_topk_prob"])
        self.gguf_writer.add_expert_weights_scale(1.0)
        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern([
            kind == "sliding_attention" for kind in self.hparams["layer_types"]
        ])
        self.gguf_writer.add_rope_pattern([
            i not in self.hparams["no_rope_layer_indices"] for i in range(self.block_count)
        ])

    def modify_tensors(self, data_torch, name, bid):
        if name.endswith(".softmax_offset"):
            name += ".weight"
        if name.endswith(".gate.expert_bias"):
            if not self.hparams["router_expert_bias"]:
                return
            name += ".bias"
        yield from super().modify_tensors(data_torch, name, bid)
