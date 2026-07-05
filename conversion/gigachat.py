from __future__ import annotations

import re
from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, split_gdn_qkvz
from .deepseek import DeepseekV2Model


@ModelBase.register("GigaChat35ForCausalLM")
class Gigachat35Model(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.GIGACHAT35
    supports_mtp_export = True
    _n_main_layers: int | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the MTP block is appended as an extra layer, convert it unless --no-mtp
        self.skip_mtp = self.no_mtp
        self.n_nextn_layers = 0 if self.no_mtp else self.hparams.get("num_nextn_predict_layers", 0)
        if self.n_nextn_layers > 0:
            self.block_count += self.n_nextn_layers
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def index_tensors(self, remote_hf_model_id: str | None = None):
        type(self)._n_main_layers = self.hparams["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        if (titem := super().filter_tensors(item)) is None:
            return None
        name, gen = titem

        assert cls._n_main_layers is not None
        is_mtp = (m := re.match(r"model\.layers\.(\d+)\.", name)) is not None and int(m.group(1)) >= cls._n_main_layers
        if is_mtp and cls.no_mtp:
            return None
        if cls.mtp_only and not is_mtp and name not in (
            "model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
            "model.norm.gate_up_projection.weight", "model.norm.gate_down_projection.weight",
        ):
            return None

        return name, gen

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        # drop head_dim (== qk_rope_head_dim) so the key/value lengths are not derived from it
        self.hparams.pop("head_dim", None)
        if self.hparams.get("scoring_func") is None:
            self.hparams["scoring_func"] = "sigmoid"
        if self.hparams.get("q_lora_rank") is None:
            self.hparams["q_lora_rank"] = 0
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["moe_intermediate_size"] * hparams["n_shared_experts"])

        # gated delta net
        self.gguf_writer.add_ssm_conv_kernel(hparams["linear_conv_kernel_dim"])
        self.gguf_writer.add_ssm_state_size(hparams["linear_key_head_dim"])
        self.gguf_writer.add_ssm_group_count(hparams["linear_num_key_heads"])
        self.gguf_writer.add_ssm_time_step_rank(hparams["linear_num_value_heads"])
        self.gguf_writer.add_ssm_inner_size(hparams["linear_value_head_dim"] * hparams["linear_num_value_heads"])

        # main layers not listed in full_attention_layers are recurrent, the MTP layer is not
        full_attention_layers = set(hparams["full_attention_layers"])
        n_main_layers = hparams["num_hidden_layers"]
        self.gguf_writer.add_recurrent_layers(
            [il < n_main_layers and il not in full_attention_layers for il in range(self.block_count)])

        swiglu_limit = float(hparams.get("swiglu_limit", 0.0))
        self.gguf_writer.add_swiglu_clamp_exp([swiglu_limit] * self.block_count)
        self.gguf_writer.add_swiglu_clamp_shexp([swiglu_limit] * self.block_count)

        if self.n_nextn_layers > 0:
            self.gguf_writer.add_nextn_predict_layers(self.n_nextn_layers)

    def prepare_metadata(self, vocab_only: bool):
        from_dir = self.fname_out.is_dir()
        norm_gate_name = "model.norm.gate_up_projection.weight"
        if not vocab_only or norm_gate_name in self.model_tensors:
            norm_gate_shape = self.model_tensors[norm_gate_name]().shape
            if len(norm_gate_shape) != 2 or norm_gate_shape[0] <= 0 or norm_gate_shape[1] != self.hparams["hidden_size"]:
                raise ValueError(f"Invalid norm gate shape: {norm_gate_shape}")
            self.gguf_writer.add_norm_gate_rank(norm_gate_shape[0])
        super().prepare_metadata(vocab_only=vocab_only)

        if not self.mtp_only or not from_dir:
            return

        self.set_mtp_filename()

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # All GigaChat norms are zero-centered, including self_attn.norm (unlike Qwen3-Next).
        if name.endswith("norm.weight"):
            data_torch = data_torch.to(torch.float32) + 1

        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch)
        elif name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
        elif name.endswith(".conv1d.weight"):
            data_torch = data_torch.squeeze()

        if name.endswith(".in_proj_qkvz.weight"):
            qkv, z = split_gdn_qkvz(
                data_torch, self.hparams["hidden_size"], self.hparams["linear_key_head_dim"],
                self.hparams["linear_value_head_dim"], self.hparams["linear_num_key_heads"], self.hparams["linear_num_value_heads"])
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_QKV,  bid, ".weight"), qkv)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_GATE, bid, ".weight"), z)
            return

        if name.endswith(".in_proj_ba.weight"):
            # original order:  [b, a] * head_count
            # corrected order: [b * head_count, a * head_count]
            num_v_heads = self.hparams["linear_num_value_heads"]
            num_k_heads = self.hparams["linear_num_key_heads"]
            hidden_size = self.hparams["hidden_size"]
            num_v_per_k = num_v_heads // num_k_heads
            data_torch = data_torch.permute(1, 0).contiguous()
            data_torch = data_torch.view(hidden_size, num_k_heads, 2 * num_v_per_k)
            beta, alpha = torch.split(data_torch, [num_v_per_k, num_v_per_k], dim=-1)
            beta = beta.contiguous().view(hidden_size, num_v_heads).permute(1, 0).contiguous()
            alpha = alpha.contiguous().view(hidden_size, num_v_heads).permute(1, 0).contiguous()
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_BETA,  bid, ".weight"), beta)
            yield (self.format_tensor_name(gguf.MODEL_TENSOR.SSM_ALPHA, bid, ".weight"), alpha)
            return

        yield from super().modify_tensors(data_torch, name, bid)
