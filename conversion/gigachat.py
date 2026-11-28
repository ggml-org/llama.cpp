from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf, split_gdn_qkvz
from .deepseek import DeepseekV2MTPModel


@ModelBase.register("GigaChat35ForCausalLM")
@ModelBase.example("ai-sage/GigaChat3.5-432B-A28B")
class GigaChat35Model(DeepseekV2MTPModel):
    model_arch = gguf.MODEL_ARCH.GIGACHAT35
    use_mtp_filename = True
    mtp_shared_tensors = DeepseekV2MTPModel.mtp_shared_tensors + (
        "model.norm.gate_up_projection.weight", "model.norm.gate_down_projection.weight",
    )

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        if self.hparams.get("use_mla_scaling_factor") is not True:
            raise ValueError("GigaChat35 requires use_mla_scaling_factor=True")
        for key, value in {
            "rope_interleave": True,
            "hidden_act": "silu",
            "norm_type": "ZeroCenteredGatedNorm",
            "layernorm_type": "pre_post",
            "layernorm_gating_weight": 2.0,
            "gated_attention": True,
            "linear_attention_type": "GigaChat35GatedDeltaNet",
            "linear_gating_type": "gated_rmsnorm_sigmoid_zero_centered",
            "linear_sigmoid_gate_scale": 2.0,
        }.items():
            if self.hparams.get(key) != value:
                raise ValueError(f"GigaChat35 requires {key}={value!r}")
        for key, value in {
            "attention_bias": False,
            "use_shared_expert_sigmoid": False,
            "linear_use_short_conv": True,
            "linear_use_legacy_qkvz_layout": False,
            "final_logit_softcapping": None,
            "scoring_func": "sigmoid",
            "nextn_is_sparse": False,
        }.items():
            if self.hparams.get(key, value) != value:
                raise ValueError(f"GigaChat35 requires {key}={value!r}")
        norm_eps = self.hparams.get("linear_attn_o_norm_eps")
        # the base and Reasoning configs have different defaults for this key
        if "linear_attn_o_norm_eps" not in self.hparams and self.hparams["rms_norm_eps"] != 1e-6:
            raise ValueError("GigaChat35 requires explicit linear_attn_o_norm_eps when rms_norm_eps differs from 1e-6")
        if norm_eps is not None and norm_eps != self.hparams["rms_norm_eps"]:
            raise ValueError("GigaChat35 requires linear_attn_o_norm_eps to match rms_norm_eps")
        if self.rope_parameters.get("rope_type") != "yarn":
            raise ValueError("GigaChat35 only supports YaRN RoPE scaling")
        rope_factor = self.rope_parameters.get("factor")
        if rope_factor is None or rope_factor < 1:
            raise ValueError("GigaChat35 requires a RoPE scaling factor >= 1")
        if self.rope_parameters.get("mscale", 1.0) != 1.0:
            raise ValueError("GigaChat35 requires RoPE mscale=1")
        if self.rope_parameters.get("mscale_all_dim") != 1.0:
            raise ValueError("GigaChat35 requires RoPE mscale_all_dim=1")
        for key in ("n_group", "topk_group", "first_k_dense_replace", "routed_scaling_factor", "norm_topk_prob"):
            if self.hparams.get(key) is None:
                raise ValueError(f"GigaChat35 requires explicit {key}")
        if "q_lora_rank" not in self.hparams:
            raise ValueError("GigaChat35 requires explicit q_lora_rank")

        # drop head_dim (== qk_rope_head_dim) so the key/value lengths are not derived from it
        self.hparams.pop("head_dim", None)
        if self.hparams["routed_scaling_factor"] == 0:
            raise ValueError("GigaChat35 does not support routed_scaling_factor=0")
        self.hparams["scoring_func"] = "sigmoid"
        if self.hparams["q_lora_rank"] is None:
            self.hparams["q_lora_rank"] = 0
        super().set_gguf_parameters()
        hparams = self.hparams

        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["moe_intermediate_size"] * hparams["n_shared_experts"])

        # the norm gate rank is only known from the weights, which a vocab-only conversion may not have
        if (norm_gate := self.model_tensors.get("model.norm.gate_up_projection.weight")) is not None:
            self.gguf_writer.add_norm_gate_rank(norm_gate().shape[0])

        # gated delta net
        self.gguf_writer.add_ssm_conv_kernel(hparams["linear_conv_kernel_dim"])
        self.gguf_writer.add_ssm_state_size(hparams["linear_key_head_dim"])
        self.gguf_writer.add_ssm_group_count(hparams["linear_num_key_heads"])
        self.gguf_writer.add_ssm_time_step_rank(hparams["linear_num_value_heads"])
        self.gguf_writer.add_ssm_inner_size(hparams["linear_value_head_dim"] * hparams["linear_num_value_heads"])

        # main layers not listed in full_attention_layers are recurrent, the MTP layers are not
        full_attention_layers = set(hparams["full_attention_layers"])
        n_main_layers = hparams["num_hidden_layers"]
        self.gguf_writer.add_recurrent_layers(
            [il < n_main_layers and il not in full_attention_layers for il in range(self.block_count)])

        # the reference clamps dense layers too, not only experts
        swiglu_limit = float(hparams["swiglu_limit"])
        self.gguf_writer.add_swiglu_clamp_exp([swiglu_limit] * self.block_count)
        self.gguf_writer.add_swiglu_clamp_shexp([swiglu_limit] * self.block_count)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # all GigaChat norms are zero-centered, including self_attn.norm (unlike Qwen3-Next)
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
                data_torch, self.hparams["linear_key_head_dim"],
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
