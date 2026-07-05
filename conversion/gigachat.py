from __future__ import annotations

import re

from typing import Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


@ModelBase.register("GigaChat35ForCausalLM")
class Gigachat35Model(TextModel):
    model_arch = gguf.MODEL_ARCH.GIGACHAT35

    _experts: list[dict[str, Tensor]] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_main_layers = self.hparams["num_hidden_layers"]
        self.n_nextn_layers = int(self.hparams.get("num_nextn_predict_layers", 0) or 0)
        self.block_count = self.n_main_layers + (0 if self.no_mtp else self.n_nextn_layers)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)
        self.full_attention_layers = set(self.hparams.get("full_attention_layers", []))

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        hparams = self.hparams

        original_num_key_value_heads = hparams.get("num_key_value_heads")
        original_head_dim = hparams.pop("head_dim", None)
        hparams["num_key_value_heads"] = 1
        try:
            super().set_gguf_parameters()
        finally:
            if original_head_dim is not None:
                hparams["head_dim"] = original_head_dim
            if original_num_key_value_heads is not None:
                hparams["num_key_value_heads"] = original_num_key_value_heads
            else:
                hparams.pop("num_key_value_heads", None)

        self.gguf_writer.add_vocab_size(hparams["vocab_size"])
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])

        self.gguf_writer.add_key_length(hparams["kv_lora_rank"] + hparams["qk_rope_head_dim"])
        self.gguf_writer.add_value_length(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length_mla(hparams["qk_nope_head_dim"] + hparams["qk_rope_head_dim"])
        self.gguf_writer.add_value_length_mla(hparams["v_head_dim"])
        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])

        if (rope_mscale_all := self.rope_parameters.get("mscale_all_dim")) is not None:
            self.gguf_writer.add_rope_scaling_yarn_log_mul(0.1 * rope_mscale_all)

        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_feed_forward_length(hparams["moe_intermediate_size"] * hparams.get("n_shared_experts", 0))
        self.gguf_writer.add_expert_count(hparams["n_routed_experts"])
        self.gguf_writer.add_expert_shared_count(hparams.get("n_shared_experts", 0))
        self.gguf_writer.add_expert_weights_scale(float(hparams["routed_scaling_factor"]))
        self.gguf_writer.add_expert_weights_norm(bool(hparams.get("norm_topk_prob", False)))

        self.gguf_writer.add_ssm_conv_kernel(hparams["linear_conv_kernel_dim"])
        self.gguf_writer.add_ssm_state_size(hparams["linear_key_head_dim"])
        self.gguf_writer.add_ssm_group_count(hparams["linear_num_key_heads"])
        self.gguf_writer.add_ssm_time_step_rank(hparams["linear_num_value_heads"])
        self.gguf_writer.add_ssm_inner_size(hparams["linear_value_head_dim"] * hparams["linear_num_value_heads"])

        recurrent_layers = [
            i < self.n_main_layers and i not in self.full_attention_layers
            for i in range(self.block_count)
        ]
        self.gguf_writer.add_array(
            gguf.Keys.Attention.RECURRENT_LAYERS.format(arch=self.gguf_writer.arch),
            recurrent_layers,
        )

        swiglu_limit = float(hparams.get("swiglu_limit", 0.0))
        self.gguf_writer.add_swiglu_clamp_exp([swiglu_limit] * self.block_count)
        self.gguf_writer.add_swiglu_clamp_shexp([swiglu_limit] * self.block_count)

        if self.n_nextn_layers > 0 and not self.no_mtp:
            self.gguf_writer.add_nextn_predict_layers(self.n_nextn_layers)

    @staticmethod
    def _suffix_for_name(name: str) -> str:
        if name.endswith(".weight"):
            return ".weight"
        if name.endswith(".bias"):
            return ".bias"
        return ""

    def _emit(self, tensor: gguf.MODEL_TENSOR, data_torch: Tensor, bid: int | None = None, suffix: str = ".weight"):
        yield (self.format_tensor_name(tensor, bid, suffix), data_torch)

    def _map_gated_norm(self, data_torch: Tensor, name: str, bid: int | None):
        suffix = self._suffix_for_name(name)
        bare = name.removesuffix(suffix)

        root_map = {
            "model.norm": gguf.MODEL_TENSOR.OUTPUT_NORM,
            "model.norm.gate_up_projection": gguf.MODEL_TENSOR.OUTPUT_NORM_GATE_UP,
            "model.norm.gate_down_projection": gguf.MODEL_TENSOR.OUTPUT_NORM_GATE_DOWN,
        }
        if bare in root_map:
            yield from self._emit(root_map[bare], data_torch, suffix=suffix)
            return True

        if bid is None:
            return False

        layer_prefix = f"model.layers.{bid}."
        if not bare.startswith(layer_prefix):
            return False
        rel = bare[len(layer_prefix):]

        layer_norm_map = {
            "input_layernorm": gguf.MODEL_TENSOR.ATTN_NORM,
            "input_layernorm.gate_up_projection": gguf.MODEL_TENSOR.ATTN_NORM_GATE_UP,
            "input_layernorm.gate_down_projection": gguf.MODEL_TENSOR.ATTN_NORM_GATE_DOWN,
            "post_self_attn_layernorm": gguf.MODEL_TENSOR.ATTN_POST_NORM,
            "post_self_attn_layernorm.gate_up_projection": gguf.MODEL_TENSOR.ATTN_POST_NORM_GATE_UP,
            "post_self_attn_layernorm.gate_down_projection": gguf.MODEL_TENSOR.ATTN_POST_NORM_GATE_DOWN,
            "post_attention_layernorm": gguf.MODEL_TENSOR.FFN_NORM,
            "post_attention_layernorm.gate_up_projection": gguf.MODEL_TENSOR.FFN_NORM_GATE_UP,
            "post_attention_layernorm.gate_down_projection": gguf.MODEL_TENSOR.FFN_NORM_GATE_DOWN,
            "post_feedforward_layernorm": gguf.MODEL_TENSOR.FFN_POST_NORM,
            "post_feedforward_layernorm.gate_up_projection": gguf.MODEL_TENSOR.FFN_POST_NORM_GATE_UP,
            "post_feedforward_layernorm.gate_down_projection": gguf.MODEL_TENSOR.FFN_POST_NORM_GATE_DOWN,
            "self_attn.q_a_layernorm": gguf.MODEL_TENSOR.ATTN_Q_A_NORM,
            "self_attn.q_a_layernorm.gate_up_projection": gguf.MODEL_TENSOR.ATTN_Q_A_NORM_GATE_UP,
            "self_attn.q_a_layernorm.gate_down_projection": gguf.MODEL_TENSOR.ATTN_Q_A_NORM_GATE_DOWN,
            "self_attn.kv_a_layernorm": gguf.MODEL_TENSOR.ATTN_KV_A_NORM,
            "self_attn.kv_a_layernorm.gate_up_projection": gguf.MODEL_TENSOR.ATTN_KV_A_NORM_GATE_UP,
            "self_attn.kv_a_layernorm.gate_down_projection": gguf.MODEL_TENSOR.ATTN_KV_A_NORM_GATE_DOWN,
        }
        tensor = layer_norm_map.get(rel)
        if tensor is None:
            return False
        yield from self._emit(tensor, data_torch, bid, suffix)
        return True

    def _map_linear_attention(self, data_torch: Tensor, name: str, bid: int | None):
        if bid is None or f"model.layers.{bid}.self_attn." not in name:
            return False

        hparams = self.hparams
        if name.endswith(".A_log"):
            yield from self._emit(gguf.MODEL_TENSOR.SSM_A, -torch.exp(data_torch), bid, suffix="")
            return True

        if name.endswith(".dt_bias"):
            yield from self._emit(gguf.MODEL_TENSOR.SSM_DT, data_torch, bid, suffix=".bias")
            return True

        if name.endswith(".conv1d.weight"):
            yield from self._emit(gguf.MODEL_TENSOR.SSM_CONV1D, data_torch.squeeze(), bid)
            return True

        if name.endswith(".norm.weight"):
            yield from self._emit(gguf.MODEL_TENSOR.SSM_NORM, data_torch, bid)
            return True

        if name.endswith(".out_proj.weight"):
            yield from self._emit(gguf.MODEL_TENSOR.SSM_OUT, data_torch, bid)
            return True

        if name.endswith(".in_proj_qkvz.weight"):
            head_k_dim = hparams["linear_key_head_dim"]
            head_v_dim = hparams["linear_value_head_dim"]
            num_v_heads = hparams["linear_num_value_heads"]
            num_k_heads = hparams["linear_num_key_heads"]
            hidden_size = hparams["hidden_size"]
            num_v_per_k = num_v_heads // num_k_heads
            split_arg_list_qkvz = [
                head_k_dim,
                head_k_dim,
                num_v_per_k * head_v_dim,
                num_v_per_k * head_v_dim,
            ]
            data_torch = data_torch.permute(1, 0).contiguous()
            data_torch = data_torch.view(hidden_size, num_k_heads, sum(split_arg_list_qkvz))
            q, k, v, z = torch.split(data_torch, split_arg_list_qkvz, dim=-1)
            q = q.contiguous().view(hidden_size, -1)
            k = k.contiguous().view(hidden_size, -1)
            v = v.contiguous().view(hidden_size, -1)
            z = z.contiguous().view(hidden_size, -1)
            qkv = torch.cat([q, k, v], dim=-1).permute(1, 0).contiguous()
            z = z.permute(1, 0).contiguous()
            yield from self._emit(gguf.MODEL_TENSOR.ATTN_QKV, qkv, bid)
            yield from self._emit(gguf.MODEL_TENSOR.ATTN_GATE, z, bid)
            return True

        if name.endswith(".in_proj_ba.weight"):
            hidden_size = hparams["hidden_size"]
            num_v_heads = hparams["linear_num_value_heads"]
            num_k_heads = hparams["linear_num_key_heads"]
            num_v_per_k = num_v_heads // num_k_heads
            data_torch = data_torch.permute(1, 0).contiguous()
            data_torch = data_torch.view(hidden_size, num_k_heads, 2 * num_v_per_k)
            beta, alpha = torch.split(data_torch, [num_v_per_k, num_v_per_k], dim=-1)
            beta = beta.contiguous().view(hidden_size, num_v_heads).permute(1, 0).contiguous()
            alpha = alpha.contiguous().view(hidden_size, num_v_heads).permute(1, 0).contiguous()
            yield from self._emit(gguf.MODEL_TENSOR.SSM_BETA, beta, bid)
            yield from self._emit(gguf.MODEL_TENSOR.SSM_ALPHA, alpha, bid)
            return True

        return False

    def _map_full_attention(self, data_torch: Tensor, name: str, bid: int | None):
        if bid is None or f"model.layers.{bid}.self_attn." not in name:
            return False

        full_attn_map = {
            "q_a_proj.weight": gguf.MODEL_TENSOR.ATTN_Q_A,
            "q_b_proj.weight": gguf.MODEL_TENSOR.ATTN_Q_B,
            "kv_a_proj_with_mqa.weight": gguf.MODEL_TENSOR.ATTN_KV_A_MQA,
            "o_proj.weight": gguf.MODEL_TENSOR.ATTN_OUT,
            "gate_proj.weight": gguf.MODEL_TENSOR.ATTN_GATE,
        }
        for suffix, tensor in full_attn_map.items():
            if name.endswith(f"self_attn.{suffix}"):
                yield from self._emit(tensor, data_torch, bid)
                return True

        if name.endswith("kv_b_proj.weight"):
            n_head = self.hparams["num_attention_heads"]
            v_head_dim = self.hparams["v_head_dim"]
            qk_nope_head_dim = self.hparams["qk_nope_head_dim"]
            expected_rows = n_head * (v_head_dim + qk_nope_head_dim)
            if data_torch.shape[0] != expected_rows:
                raise ValueError(f"Unexpected kv_b_proj shape for {name}: {tuple(data_torch.shape)}, expected first dim {expected_rows}")
            kv_b = data_torch.view(n_head, v_head_dim + qk_nope_head_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope_head_dim, v_head_dim], dim=1)
            k_b = k_b.transpose(1, 2).contiguous()
            v_b = v_b.contiguous()
            yield from self._emit(gguf.MODEL_TENSOR.ATTN_K_B, k_b, bid)
            yield from self._emit(gguf.MODEL_TENSOR.ATTN_V_B, v_b, bid)
            return True

        return False

    def _map_moe_expert(self, data_torch: Tensor, name: str, bid: int | None):
        if bid is None or ".mlp.experts." not in name:
            return False

        match = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight", name)
        if match is None:
            return False
        layer_id = int(match.group(1))
        expert_id = int(match.group(2))
        proj_type = match.group(3)
        if layer_id != bid:
            raise ValueError(f"Layer id mismatch for {name}: bid={bid}")

        n_experts = self.hparams["n_routed_experts"]
        if self._experts is None:
            self._experts = [{} for _ in range(self.block_count)]

        key = f"{proj_type}:{expert_id}"
        self._experts[bid][key] = data_torch
        keys_for_proj = [f"{proj_type}:{i}" for i in range(n_experts)]
        if not all(k in self._experts[bid] for k in keys_for_proj):
            return True

        datas = [self._experts[bid].pop(k) for k in keys_for_proj]
        merged = torch.stack(datas, dim=0)
        tensor_by_proj = {
            "gate_proj": gguf.MODEL_TENSOR.FFN_GATE_EXP,
            "up_proj": gguf.MODEL_TENSOR.FFN_UP_EXP,
            "down_proj": gguf.MODEL_TENSOR.FFN_DOWN_EXP,
        }
        yield from self._emit(tensor_by_proj[proj_type], merged, bid)
        return True

    def _map_nextn(self, data_torch: Tensor, name: str, bid: int | None):
        if bid is None:
            return False

        suffix = self._suffix_for_name(name)
        bare = name.removesuffix(suffix)

        layer_prefix = f"model.layers.{bid}."
        if not bare.startswith(layer_prefix):
            return False
        rel = bare[len(layer_prefix):]

        nextn_map = {
            "eh_proj": gguf.MODEL_TENSOR.NEXTN_EH_PROJ,
            "embed_tokens": gguf.MODEL_TENSOR.NEXTN_EMBED_TOKENS,
            "enorm": gguf.MODEL_TENSOR.NEXTN_ENORM,
            "enorm.gate_up_projection": gguf.MODEL_TENSOR.NEXTN_ENORM_GATE_UP,
            "enorm.gate_down_projection": gguf.MODEL_TENSOR.NEXTN_ENORM_GATE_DOWN,
            "hnorm": gguf.MODEL_TENSOR.NEXTN_HNORM,
            "hnorm.gate_up_projection": gguf.MODEL_TENSOR.NEXTN_HNORM_GATE_UP,
            "hnorm.gate_down_projection": gguf.MODEL_TENSOR.NEXTN_HNORM_GATE_DOWN,
            "shared_head.head": gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_HEAD,
            "shared_head.norm": gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM,
            "shared_head.norm.gate_up_projection": gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM_GATE_UP,
            "shared_head.norm.gate_down_projection": gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM_GATE_DOWN,
        }
        tensor = nextn_map.get(rel)
        if tensor is None:
            return False
        yield from self._emit(tensor, data_torch, bid, suffix)
        return True

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if self.hparams.get("tie_word_embeddings", False) and name == "lm_head.weight":
            logger.info("Skipping tied output layer 'lm_head.weight' (will use token_embd.weight)")
            return

        if bid is not None and bid >= self.n_main_layers:
            if self.no_mtp:
                return

            handled = yield from self._map_nextn(data_torch, name, bid)
            if handled:
                return

            handled = yield from self._map_gated_norm(data_torch, name, bid)
            if handled:
                return

            handled = yield from self._map_full_attention(data_torch, name, bid)
            if handled:
                return

            yield from super().modify_tensors(data_torch, name, bid)
            return

        handled = yield from self._map_gated_norm(data_torch, name, bid)
        if handled:
            return

        handled = yield from self._map_moe_expert(data_torch, name, bid)
        if handled:
            return

        handled = yield from self._map_linear_attention(data_torch, name, bid)
        if handled:
            return

        handled = yield from self._map_full_attention(data_torch, name, bid)
        if handled:
            return

        yield from super().modify_tensors(data_torch, name, bid)

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts is not None:
            leftovers = []
            for bid, tensors in enumerate(self._experts):
                leftovers.extend(f"blk.{bid}:{name}" for name in tensors)
            if leftovers:
                raise ValueError(f"Unprocessed experts: {leftovers[:20]}{'...' if len(leftovers) > 20 else ''}")
