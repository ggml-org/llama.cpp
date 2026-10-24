from __future__ import annotations

from typing import Any, Callable, Iterable

import torch
from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


def _with_default(config: dict[str, Any], key: str, default: Any) -> Any:
    # as in the HF config class, an absent key and an explicit null both take the default
    value = config.get(key)
    return default if value is None else value


def _dequant_mxfp8(weight: Tensor, scale: Tensor) -> Tensor:
    # E4M3 [N, K] with one E8M0 byte per 1x32 block [N, K/32]: value * 2**(byte - 127),
    # the fp32 product of dequantize_mxfp8 in the HF modeling code, so the result is exact
    # a float8_e8m0fnu scale already decodes to the same power of two
    factor = torch.exp2(scale.float() - 127) if scale.dtype == torch.uint8 else scale.float()
    rows, cols = weight.shape
    return (weight.float().reshape(rows, cols // 32, 32) * factor.unsqueeze(-1)).reshape(rows, cols)


@ModelBase.register("Raon1MiniForCausalLM")
@ModelBase.example("KRAFTON/Raon-1-mini-preview")
class Raon1Model(TextModel):
    model_arch = gguf.MODEL_ARCH.RAON1

    # mHC sites: HF module name -> tensor prefix; each site has 10 parameters, packed into fn / base / scale
    _MHC_SITES = (("self_attn_raon_mhc", "HC_ATTN"), ("mlp_raon_mhc", "HC_FFN"))
    _MHC_PARTS = ("phi_pre", "phi_post", "res.phi_z", "b_pre", "b_post", "res.b_z",
                  "alpha_pre", "alpha_post", "res.alpha_z", "res.eta")
    # kept unquantized by llama-quantize (src/llama-quant.cpp); a quantized --outtype keeps them at F16 the same way
    _KEEP_UNQUANTIZED = ("hc_attn_fn", "hc_ffn_fn", "ssm_f_a", "ssm_f_b", "ssm_beta", "ssm_rda")
    _KEEP_UNQUANTIZED_Q8_0 = ("attn_kv_a_mqa", "attn_k_b", "attn_v_b")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._experts: dict[tuple[int, str], dict[int, Tensor]] = {}
        self._mhc: dict[tuple[int, str], dict[str, Tensor]] = {}
        self._rda_layers = set(self.hparams["rda_config"]["rda_layers"])

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        tensors = super().index_tensors(remote_hf_model_id=remote_hf_model_id)
        # MTP is not supported: drop the nextn layers (model.layers.{num_hidden_layers}..) before any data is read
        n_layer = self.hparams["num_hidden_layers"]
        mtp = [name for name in tensors if name.startswith("model.layers.") and int(name.split(".")[2]) >= n_layer]
        if mtp:
            n_mtp = len({name.split(".")[2] for name in mtp})
            logger.info(f"Skipping {len(mtp)} MTP tensors in {n_mtp} layer(s) from model.layers.{n_layer}; MTP is not supported")
            for name in mtp:
                del tensors[name]
        return tensors

    def dequant_model(self):
        # the release stores the routed experts as MXFP8 (quant_method "mxfp8", weight + weight_scale_inv);
        # every other tensor is in quantization_config.ignored_layers and stays BF16
        quant_config = self.hparams.get("quantization_config") or {}
        if quant_config.get("quant_method") != "mxfp8":
            return super().dequant_model()
        if list(quant_config.get("weight_block_size") or [1, 32]) != [1, 32]:
            raise ValueError(f"Raon-1 MXFP8 needs weight_block_size [1, 32], got {quant_config['weight_block_size']}")
        for name in [name for name in self.model_tensors if name.endswith(".weight_scale_inv")]:
            weight_name = name.removesuffix("_scale_inv")
            if ".moe.experts." not in weight_name:
                raise ValueError(f"Unexpected MXFP8 tensor {weight_name}: only the routed experts are quantized")
            weight, scale = self.model_tensors[weight_name], self.model_tensors.pop(name)
            self.model_tensors[weight_name] = lambda w=weight, s=scale: _dequant_mxfp8(w(), s())

    def set_gguf_parameters(self):
        hp = self.hparams
        rda_config = hp["rda_config"]
        n_layer = hp["num_hidden_layers"]
        # 1-based layer ids, as in the HF config (is_rda_layer: layer_idx + 1 in rda_layers)
        rda = self._rda_layers
        mla = set(rda_config["mla_layers"])
        if rda & mla or rda | mla != set(range(1, n_layer + 1)):
            raise ValueError("Raon-1 attention layer lists must partition all layers")
        # the hybrid memory needs both layer kinds
        if not rda or not mla:
            raise ValueError("Raon-1 needs at least one RDA (rda_layers) and one MLA (mla_layers) layer")
        if hp.get("tie_word_embeddings"):
            raise ValueError("Raon-1 with tie_word_embeddings is not supported (the released model has a separate lm_head)")
        if (not hp["mla_use_nope"] or not hp["mla_use_output_gate"]
                or not rda_config["full_rank_gate"] or hp["hidden_act"] != "situ"
                or hp["moe_router_activation_func"] != "sigmoid"
                or hp.get("use_grouped_topk") or hp.get("routed_expert_hidden_size")
                or hp.get("attn_res_block_size") or hp["moe_layer_freq"] != 1):
            raise ValueError("Unsupported Raon-1 architecture variant")
        if rda_config["num_heads"] != hp["num_attention_heads"]:
            raise ValueError("Raon-1 RDA and MLA head counts must match")

        super().set_gguf_parameters()
        self.gguf_writer.add_vocab_size(hp["vocab_size"])
        self.gguf_writer.add_head_count_kv([0 if i + 1 in rda else 1 for i in range(n_layer)])
        self.gguf_writer.add_ssm_conv_kernel(rda_config["conv_kernel_size"])
        # the .get() defaults below are the HF config defaults
        self.gguf_writer.add_ssm_time_step_rank(hp.get("rda_decay_rank", 128))
        self.gguf_writer.add_kda_head_dim(rda_config["head_dim"])
        # HF uses -5.0 when the key is absent; an explicit null selects the unbounded gate
        if (bound := rda_config.get("gate_lower_bound", -5.0)) is not None:
            self.gguf_writer.add_kda_gate_lower_bound(bound)
        self.gguf_writer.add_q_lora_rank(hp["q_lora_rank"])
        self.gguf_writer.add_kv_lora_rank(hp["kv_lora_rank"])
        self.gguf_writer.add_rope_dimension_count(hp["qk_rope_head_dim"])
        # the generic mapping sizes these from config.head_dim, but the MLA cache keeps the latent width
        self.gguf_writer.remove_key(gguf.Keys.Attention.KEY_LENGTH.format(arch=self.gguf_writer.arch))
        self.gguf_writer.remove_key(gguf.Keys.Attention.VALUE_LENGTH.format(arch=self.gguf_writer.arch))
        self.gguf_writer.add_key_length(hp["kv_lora_rank"] + hp["qk_rope_head_dim"])
        self.gguf_writer.add_value_length(hp["kv_lora_rank"])
        self.gguf_writer.add_key_length_mla(hp["qk_nope_head_dim"] + hp["qk_rope_head_dim"])
        self.gguf_writer.add_value_length_mla(hp["v_head_dim"])
        self.gguf_writer.add_latent_norm_rms_eps(hp.get("mla_latent_norm_eps", 1e-6))
        self.gguf_writer.add_expert_feed_forward_length(hp["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hp["num_shared_experts"])
        self.gguf_writer.add_leading_dense_block_count(hp["first_k_dense_replace"])
        self.gguf_writer.add_expert_weights_scale(hp["routed_scaling_factor"])
        self.gguf_writer.add_expert_weights_norm(hp["moe_renormalize"])
        self.gguf_writer.add_activation_situ_beta(hp["activation_situ_beta"])
        self.gguf_writer.add_activation_situ_linear_beta(hp["activation_situ_linear_beta"])
        self.gguf_writer.add_hyper_connection_count(_with_default(hp, "raon_mhc_streams", 4))
        self.gguf_writer.add_hyper_connection_epsilon(_with_default(hp, "raon_mhc_selector_eps", 1e-5))
        self.gguf_writer.add_hyper_connection_magnitude(_with_default(hp, "raon_mhc_post_factor", 2.0))

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._experts or self._mhc:
            raise ValueError(f"Incomplete Raon-1 tensor groups: experts={list(self._experts)}, mHC={list(self._mhc)}")

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if ".hc_" in new_name and not new_name.endswith("_fn.weight"):
            return gguf.GGMLQuantizationType.F32
        quantized = (gguf.LlamaFileType.MOSTLY_Q8_0, gguf.LlamaFileType.MOSTLY_TQ1_0, gguf.LlamaFileType.MOSTLY_TQ2_0)
        if self.ftype in quantized and new_name.startswith("blk."):
            keep = self._KEEP_UNQUANTIZED
            if self.ftype == gguf.LlamaFileType.MOSTLY_Q8_0:
                keep += self._KEEP_UNQUANTIZED_Q8_0
            if new_name.split(".")[2] in keep:
                return gguf.GGMLQuantizationType.F16
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        for site, prefix in self._MHC_SITES:
            marker = f".{site}."
            if marker not in name:
                continue
            assert bid is not None
            parts = self._mhc.setdefault((bid, site), {})
            suffix = name.split(marker, 1)[1]
            if suffix in parts:
                raise ValueError(f"Duplicate tensor {name}")
            parts[suffix] = data_torch
            if len(parts) < len(self._MHC_PARTS):
                return
            fn = torch.cat([parts[k] for k in ("phi_pre", "phi_post", "res.phi_z")])
            base = torch.cat([parts[k].float() for k in ("b_pre", "b_post", "res.b_z")])
            scale = torch.stack([parts[k].float().reshape(()) for k in ("alpha_pre", "alpha_post", "res.alpha_z", "res.eta")])
            for kind, tensor in (("FN", fn), ("BASE", base), ("SCALE", scale)):
                yield self.format_tensor_name(getattr(gguf.MODEL_TENSOR, f"{prefix}_{kind}"), bid), tensor
            del self._mhc[bid, site]
            return

        if ".moe.experts." in name:
            assert bid is not None
            tail = name.split(".experts.", 1)[1].split(".")
            eid, wid = int(tail[0]), tail[1]
            group = self._experts.setdefault((bid, wid), {})
            if eid in group or not 0 <= eid < self.hparams["num_experts"]:
                raise ValueError(f"Invalid expert tensor {name}")
            group[eid] = data_torch
            if len(group) == self.hparams["num_experts"]:
                tid = {"gate_proj": gguf.MODEL_TENSOR.FFN_GATE_EXP, "down_proj": gguf.MODEL_TENSOR.FFN_DOWN_EXP,
                       "up_proj": gguf.MODEL_TENSOR.FFN_UP_EXP}[wid]
                yield self.format_tensor_name(tid, bid), torch.stack([group[i] for i in range(len(group))])
                del self._experts[bid, wid]
            return

        # moe.gate / moe.shared_experts -> the block_sparse_moe names the tensor map already has
        if ".moe." in name:
            name = name.replace(".moe.", ".block_sparse_moe.")
        if name.endswith("kv_b_proj.weight"):
            heads = self.hparams["num_attention_heads"]
            nk, nv = self.hparams["qk_nope_head_dim"], self.hparams["v_head_dim"]
            kv = data_torch.reshape(heads, nk + nv, self.hparams["kv_lora_rank"])
            k, v = torch.split(kv, [nk, nv], dim=1)
            yield self.map_tensor_name(name.replace("kv_b_proj", "k_b_proj")), k.transpose(1, 2).contiguous()
            yield self.map_tensor_name(name.replace("kv_b_proj", "v_b_proj")), v.contiguous()
            return
        if name.endswith(".self_attn.g_proj.weight"):
            assert bid is not None
            rda = bid + 1 in self._rda_layers
            tid = gguf.MODEL_TENSOR.SSM_G if rda else gguf.MODEL_TENSOR.ATTN_GATE
            yield self.format_tensor_name(tid, bid), data_torch
            return
        if name.endswith((".q_conv1d.weight", ".k_conv1d.weight", ".v_conv1d.weight")):
            data_torch = data_torch.reshape(1, -1, 1, data_torch.shape[-1])
        if name.endswith(".A_log"):
            data_torch = -torch.exp(data_torch.float())
        # dt_bias -> the name SSM_DT's mapping expects, as for qwen3next
        if name.endswith(".dt_bias"):
            name = name.removesuffix(".dt_bias") + ".dt_proj.bias"
        yield from super().modify_tensors(data_torch, name, bid)
