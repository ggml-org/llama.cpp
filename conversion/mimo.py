from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf


@ModelBase.register("MiMoV2FlashForCausalLM")
class MimoV2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.MIMO2

    def set_gguf_parameters(self):
        super().set_gguf_parameters()

        assert self.hparams["swa_head_dim"] == self.hparams["head_dim"]
        assert self.hparams["swa_num_attention_heads"] == self.hparams["num_attention_heads"]
        assert self.hparams["swa_v_head_dim"] == self.hparams["v_head_dim"]
        assert self.hparams["topk_method"] == "noaux_tc"

        n_head_kv = self.hparams["num_key_value_heads"]
        n_head_kv_swa = self.hparams["swa_num_key_value_heads"]
        n_head_kv_arr = [n_head_kv_swa if use_swa == 1 else n_head_kv for use_swa in self.hparams["hybrid_layer_pattern"]]
        self.gguf_writer.add_head_count_kv(n_head_kv_arr)

        self.gguf_writer.add_sliding_window(self.hparams["sliding_window"])
        self.gguf_writer.add_sliding_window_pattern(self.hparams["hybrid_layer_pattern"])
        self.gguf_writer.add_value_length(self.hparams["v_head_dim"])
        self.gguf_writer.add_expert_count(self.hparams["n_routed_experts"])
        self.gguf_writer.add_expert_feed_forward_length(self.hparams["moe_intermediate_size"])

        rope_dim = int(self.hparams["head_dim"] * self.hparams["partial_rotary_factor"])
        self.gguf_writer.add_rope_dimension_count(rope_dim)

        self.gguf_writer.add_layer_norm_rms_eps(self.hparams.get("layernorm_epsilon", 1e-5))

    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch, name, bid):
        if name.endswith("e_score_correction_bias"):
            name = name.replace("e_score_correction_bias", "e_score_correction.bias")

        if "attention_sink" in name and not name.endswith(".weight"):
            name += ".weight"

        # TODO: mimo v2 does not indicate the number of next-token-prediction layers, therefore we cannot do the same way as GLM4_MOE
        if "model.mtp." in name:
            return

        # process the experts separately
        if name.find("mlp.experts") != -1:
            n_experts = self.hparams["n_routed_experts"]
            assert bid is not None

            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]

            self._experts[bid][name] = data_torch

            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for w_name in ["gate_proj", "up_proj", "down_proj"]:
                    datas: list[Tensor] = []

                    for xid in range(n_experts):
                        ename_to_retrieve = f"model.layers.{bid}.mlp.experts.{xid}.{w_name}.weight"
                        datas.append(self._experts[bid][ename_to_retrieve])
                        del self._experts[bid][ename_to_retrieve]

                    data_torch = torch.stack(datas, dim=0)
                    merged_name = f"model.layers.{bid}.mlp.experts.{w_name}.weight"

                    yield from super().modify_tensors(data_torch, merged_name, bid)
                return
            else:
                return
        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()

        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
