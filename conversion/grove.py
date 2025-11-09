from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch, logger
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("GroveMoeForCausalLM", "modeling_grove_moe.GroveMoeForCausalLM")
class GroveMoeModel(TextModel):
    model_arch = gguf.MODEL_ARCH.GROVEMOE

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        if (n_experts := self.hparams.get("num_experts")) is not None:
            self.gguf_writer.add_expert_count(n_experts)
        if (moe_intermediate_size := self.hparams.get("moe_intermediate_size")) is not None:
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate_size)
            logger.info(f"gguf: expert feed forward length = {moe_intermediate_size}")
        # FIXME?: Hardcoded https://huggingface.co/inclusionAI/GroveMoE-Inst/blob/c4c69e5970d18907b5e6ddccdfd55176fe292df1/modeling_grove_moe.py#L299
        self.gguf_writer.add_expert_chunk_feed_forward_length(self.hparams.get("head_dim") or 128)
        # FIXME?: Hardcoded https://huggingface.co/inclusionAI/GroveMoE-Inst/blob/c4c69e5970d18907b5e6ddccdfd55176fe292df1/modeling_grove_moe.py#L298
        self.gguf_writer.add_experts_per_group(2)
        # FIXME?: Hardcoded https://huggingface.co/inclusionAI/GroveMoE-Inst/blob/c4c69e5970d18907b5e6ddccdfd55176fe292df1/modeling_grove_moe.py#L376
        self.gguf_writer.add_expert_group_scale(0.05)
        # YaRN is not enabled by default
        # To enable it, please refer to this guide: https://huggingface.co/Qwen/Qwen3-30B-A3B#processing-long-texts
        rope_scaling = self.hparams.get("rope_scaling") or {}
        if rope_scaling.get("rope_type", rope_scaling.get("type")) == "yarn" and "factor" in rope_scaling:
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_scaling["factor"])
            self.gguf_writer.add_rope_scaling_orig_ctx_len(rope_scaling["original_max_position_embeddings"])
    _experts: list[dict[str, Tensor]] | None = None
    _chunk_experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name.endswith(".expert_bias"):
            # FIXME?: Unused https://huggingface.co/inclusionAI/GroveMoE-Inst/blob/c4c69e5970d18907b5e6ddccdfd55176fe292df1/modeling_grove_moe.py#L303
            return []
        # process the experts separately
        if name.find("chunk_experts") != -1:
            n_experts = self.hparams["num_experts"] // 2 # see add_experts_per_group
            assert bid is not None
            if self._chunk_experts is None:
                self._chunk_experts = [{} for _ in range(self.block_count)]
            self._chunk_experts[bid][name] = data_torch
            if len(self._chunk_experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.chunk_experts.{xid}.{w_name}.weight"
                        datas.append(self._chunk_experts[bid][ename])
                        del self._chunk_experts[bid][ename]
                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.chunk_experts.{w_name}.weight"
                    new_name = self.map_tensor_name(merged_name)
                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []
        elif name.find("experts") != -1:
            n_experts = self.hparams["num_experts"]
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                tensors: list[tuple[str, Tensor]] = []
                # merge the experts into a single 3d tensor
                for w_name in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"
                    new_name = self.map_tensor_name(merged_name)
                    tensors.append((new_name, data_torch))
                return tensors
            else:
                return []
        return [(self.map_tensor_name(name), data_torch)]

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._chunk_experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            chunk_experts = [k for d in self._chunk_experts for k in d.keys()]
            if len(chunk_experts) > 0:
                raise ValueError(f"Unprocessed adjugate experts: {chunk_experts}")
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
