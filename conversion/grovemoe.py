from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import MOE_HF_MLP, ModelBase, TextModel, gguf, logger


@ModelBase.register("GroveMoeForCausalLM", "modeling_grove_moe.GroveMoeForCausalLM")
class GroveMoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GROVEMOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")
        # FIXME?: Hardcoded https://huggingface.co/inclusionAI/GroveMoE-Inst/blob/c4c69e5970d18907b5e6ddccdfd55176fe292df1/modeling_grove_moe.py#L299
        self.gguf_writer.add_expert_chunk_feed_forward_length(self.hparams.get("head_dim") or 128)
        # FIXME?: Hardcoded https://huggingface.co/inclusionAI/GroveMoE-Inst/blob/c4c69e5970d18907b5e6ddccdfd55176fe292df1/modeling_grove_moe.py#L298
        self.gguf_writer.add_experts_per_group(2)
        # FIXME?: Hardcoded https://huggingface.co/inclusionAI/GroveMoE-Inst/blob/c4c69e5970d18907b5e6ddccdfd55176fe292df1/modeling_grove_moe.py#L376
        self.gguf_writer.add_expert_group_scale(0.05)

    moe_experts = [
        MOE_HF_MLP,
        # chunk experts are grouped in pairs, see add_experts_per_group
        MOE_HF_MLP._replace(
            src="model.layers.{bid}.mlp.chunk_experts.{xid}.{w}.weight",
            dst="model.layers.{bid}.mlp.chunk_experts.{w}.weight",
            n_expert=lambda model: model.find_hparam(["num_local_experts", "num_experts"]) // 2,
        ),
    ]

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".expert_bias"):
            # FIXME?: Unused https://huggingface.co/inclusionAI/GroveMoE-Inst/blob/c4c69e5970d18907b5e6ddccdfd55176fe292df1/modeling_grove_moe.py#L303
            return

        yield from super().modify_tensors(data_torch, name, bid)
