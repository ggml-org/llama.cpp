from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, gguf
from .deepseek import DeepseekV2Model


@ModelBase.register("GigaChat35ForCausalLM")
class Gigachat35Model(DeepseekV2Model):
    model_arch = gguf.MODEL_ARCH.GIGACHAT35

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the MTP block is appended as an extra layer, convert it unless --no-mtp
        self.skip_mtp = self.no_mtp
        self.n_nextn_layers = 0 if self.no_mtp else self.hparams.get("num_nextn_predict_layers", 0)
        if self.n_nextn_layers > 0:
            self.block_count += self.n_nextn_layers
            self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        # drop head_dim (== qk_rope_head_dim) so the key/value lengths are not derived from it
        self.hparams.pop("head_dim", None)
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

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        # fold the +1 into the weights
        if name.endswith("norm.weight"):
            data_torch = data_torch.to(torch.float32) + 1

        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch)
        elif name.endswith(".dt_bias"):
            name = name.rpartition(".dt_bias")[0] + ".dt_proj.bias"
        elif name.endswith(".conv1d.weight"):
            data_torch = data_torch.squeeze()

        if name.endswith(".in_proj_qkvz.weight"):
            # order: [q * head_count, k * head_count, v * head_count, z * head_count]
            head_k_dim = self.hparams["linear_key_head_dim"]
            head_v_dim = self.hparams["linear_value_head_dim"]
            num_v_heads = self.hparams["linear_num_value_heads"]
            num_k_heads = self.hparams["linear_num_key_heads"]
            hidden_size = self.hparams["hidden_size"]
            split_arg_list_qkvz = [
                head_k_dim, # q partition
                head_k_dim, # k partition
                (num_v_heads // num_k_heads * head_v_dim), # v partition
                (num_v_heads // num_k_heads * head_v_dim), # z partition
            ]
            # view as (n_embd, head_count, [q+k+v+z])
            data_torch = data_torch.permute(1, 0).contiguous()
            data_torch = data_torch.view(hidden_size, num_k_heads, sum(split_arg_list_qkvz))
            # split into q, k, v, z
            q, k, v, z = torch.split(data_torch, split_arg_list_qkvz, dim=-1)
            # flatten dim + head_count
            q = q.contiguous().view(hidden_size, -1)
            k = k.contiguous().view(hidden_size, -1)
            v = v.contiguous().view(hidden_size, -1)
            z = z.contiguous().view(hidden_size, -1)
            # stack back
            qkv = torch.cat([q, k, v], dim=-1).permute(1, 0).contiguous()
            z = z.permute(1, 0).contiguous()
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
