from __future__ import annotations
from .base import (
    ModelBase, TextModel, gguf, torch, logger
)
from typing import TYPE_CHECKING, Iterable
if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("JambaForCausalLM")
class JambaModel(TextModel):
    model_arch = gguf.MODEL_ARCH.JAMBA

    def set_vocab(self):
        if (self.dir_model / "tokenizer.model").is_file():
            self._set_vocab_sentencepiece()
        else:
            self._set_vocab_llama_hf()
            self.gguf_writer.add_add_space_prefix(False)

    def set_gguf_parameters(self):
        d_model = self.find_hparam(["hidden_size", "mamba_d_model"])
        d_conv  = self.find_hparam(["mamba_d_conv"],  optional=True) or 4
        d_inner = self.hparams["mamba_expand"] * d_model
        d_state = self.find_hparam(["mamba_d_state"], optional=True) or 16
        # ceiling division
        # ref: https://stackoverflow.com/a/17511341/22827863
        # ref: https://github.com/state-spaces/mamba/blob/ce59daea3a090d011d6476c6e5b97f6d58ddad8b/mamba_ssm/modules/mamba_simple.py#L58
        dt_rank      = self.find_hparam(["mamba_dt_rank"], optional=True) or -(d_model // -16)
        rms_norm_eps = self.find_hparam(["layer_norm_epsilon", "rms_norm_eps"], optional=True) or 1e-6
        n_kv_head = self.hparams["num_key_value_heads"]
        attn_offset = self.hparams["attn_layer_offset"]
        attn_period = self.hparams["attn_layer_period"]
        n_kv_vec = [0 for _ in range(attn_offset)] + [
            n_kv_head if (i - attn_offset) % attn_period == 0 else 0 for i in range(attn_offset, self.block_count)
        ]
        self.gguf_writer.add_block_count(self.block_count)
        self.gguf_writer.add_context_length(self.find_hparam(["max_position_embeddings", "n_ctx"]))
        self.gguf_writer.add_embedding_length(d_model)
        self.gguf_writer.add_feed_forward_length(self.hparams["intermediate_size"])
        self.gguf_writer.add_head_count(self.hparams["num_attention_heads"])
        self.gguf_writer.add_head_count_kv(n_kv_vec)
        self.gguf_writer.add_ssm_conv_kernel(d_conv)
        self.gguf_writer.add_ssm_inner_size(d_inner)
        self.gguf_writer.add_ssm_state_size(d_state)
        self.gguf_writer.add_ssm_time_step_rank(dt_rank)
        self.gguf_writer.add_layer_norm_rms_eps(rms_norm_eps)
        self.gguf_writer.add_expert_count(self.hparams["num_experts"])
        self.gguf_writer.add_expert_used_count(self.hparams["num_experts_per_tok"])
        self.gguf_writer.add_file_type(self.ftype)
    _experts: list[dict[str, Tensor]] | None = None

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # Mini-Jamba
        name = name.replace(".moe.", ".feed_forward.")
        if bid is not None:
            moe_offset = self.hparams["expert_layer_offset"]
            moe_period = self.hparams["expert_layer_period"]
            if not (bid >= moe_offset and (bid - moe_offset) % moe_period == 0):
                name = name.replace(".experts.0.", ".")
        # process the experts separately
        if ".feed_forward.experts." in name:
            n_experts = self.hparams["num_experts"]
            assert bid is not None
            if self._experts is None:
                self._experts = [{} for _ in range(self.block_count)]
            self._experts[bid][name] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                # merge the experts into a single 3d tensor
                for wid in ["down_proj", "gate_proj", "up_proj"]:
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        ename = f"model.layers.{bid}.feed_forward.experts.{xid}.{wid}.weight"
                        datas.append(self._experts[bid][ename])
                        del self._experts[bid][ename]
                    data_torch = torch.stack(datas, dim=0)
                    # using the same merged name as qwen2moe
                    merged_name = f"model.layers.{bid}.mlp.experts.{wid}.weight"
                    new_name = self.map_tensor_name(merged_name)
                    yield new_name, data_torch
            return
        new_name = self.map_tensor_name(name)
        if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.SSM_CONV1D, bid):
            data_torch = data_torch.squeeze()
        if name.endswith(".A_log"):
            logger.debug("A_log --> A ==> " + new_name)
            data_torch = -torch.exp(data_torch)
        yield (new_name, data_torch)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            # flatten `list[dict[str, Tensor]]` into `list[str]`
            experts = [k for d in self._experts for k in d.keys()]
            if len(experts) > 0:
                raise ValueError(f"Unprocessed experts: {experts}")
