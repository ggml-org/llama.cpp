from __future__ import annotations

from .base import MOE_BLOCK_SPARSE, ModelBase, TextModel, gguf, logger


@ModelBase.register("SmallThinkerForCausalLM")
class SmallThinkerModel(TextModel):
    model_arch = gguf.MODEL_ARCH.SMALLTHINKER

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (n_experts := self.hparams.get("moe_num_primary_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
        if (n_experts_used := self.hparams.get("moe_num_active_primary_experts")) is not None:
            self.gguf_writer.add_expert_used_count(n_experts_used)
        if (moe_intermediate_size := self.hparams.get("moe_ffn_hidden_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            self.gguf_writer.add_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")
        if (self.hparams.get('moe_primary_router_apply_softmax')):
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SOFTMAX)
        else:
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        sliding_window_layout = self.hparams.get("sliding_window_layout")
        if sliding_window_layout:
            for i in sliding_window_layout:
                if i != 0:
                    sliding_window = self.hparams.get("sliding_window_size")
                    if sliding_window:
                        self.gguf_writer.add_sliding_window(sliding_window)
                    break

    moe_experts = [MOE_BLOCK_SPARSE._replace(
        weights=("down", "gate", "up"),
        n_expert=("moe_num_primary_experts", "num_local_experts", "num_experts"),
    )]
