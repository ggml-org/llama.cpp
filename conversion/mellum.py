from __future__ import annotations

from .base import MOE_HF_MLP, ModelBase, TextModel, gguf, logger


@ModelBase.register("MellumForCausalLM")
class MellumModel(TextModel):
    model_arch = gguf.MODEL_ARCH.MELLUM

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")

        use_sliding_window = self.hparams.get("use_sliding_window")
        sliding_window = self.hparams.get("sliding_window")
        if (use_sliding_window is True or use_sliding_window is None) and sliding_window is not None:
            self.gguf_writer.add_sliding_window(sliding_window)
            logger.info(f"gguf: sliding window = {sliding_window}")
            self.gguf_writer.add_sliding_window_pattern([t == "sliding_attention" for t in self.hparams["layer_types"]])
            logger.info(f"gguf: sliding window pattern length = {len(self.hparams['layer_types'])}")

    moe_experts = [MOE_HF_MLP]
