from __future__ import annotations

import re

from typing import Callable, Iterable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

from .base import LazyTorchTensor, ModelBase, TextModel, gguf, logger


@ModelBase.register("Ernie5_5_MoeForCausalLM")
@ModelBase.example("baidu/ERNIE-5.5-Flash-Base-BF16")
class Ernie5_5Model(TextModel):
    """
    ERNIE 5.5 (`ernie5_5_moe`): a DeepSeek-V4 hybrid. Most layers are V4 HCA layers (compress ratio 128
    with a 128-token window) with a sigmoid output gate and a VHA head mix; the layers with a negative
    compress ratio are MLA layers with a DSA indexer instead. mHC with 4 streams, one dense layer, then
    latent MoE layers (split-feature sigmoid routing, a learned per-expert scale, situ experts), and a
    MultiMax transform on the logits.

    The HCA layers are written as the SWA kind (head 512, K=V) and the MLA layers as the full kind
    (K = kv latent + rope = 576, V = latent 512), so both live in the DSV4 raw cache.

    The MTP layer (an MLA layer past num_hidden_layers) is kept as the NextN block unless --no-mtp:
    e_proj and h_proj become one eh_proj on [enorm(e) | hnorm(h)], and it has its own mHC head.
    """
    model_arch = gguf.MODEL_ARCH.ERNIE5_5
    supports_mtp_export = True

    _experts: list[dict[str, Tensor]] | None = None
    _hc_scales: dict[str, dict[int, Tensor]]
    _routers: dict[int, dict[int, Tensor]]
    _vha: dict[int, dict[str, Tensor]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # rope_scaling uses "type" (transformers 4.x)
        if (rope_scaling := self.hparams.get("rope_scaling")) is not None:
            if "rope_type" not in rope_scaling and (rope_type := rope_scaling.get("type")) is not None:
                rope_scaling["rope_type"] = rope_type
            self.rope_parameters.update(**rope_scaling)
            self.rope_parameters["rope_theta"] = self.hparams["rope_theta"]

        if self.mtp_only:
            raise ValueError("ERNIE 5.5: an MTP-only file is not supported, the NextN block goes in the main file")
        self.block_count = self.hparams["num_hidden_layers"]
        if not self.no_mtp:
            self.block_count += self.hparams.get("num_nextn_predict_layers", 0)
        self.tensor_map = gguf.get_tensor_name_map(self.model_arch, self.block_count)

        self._experts = [{} for _ in range(self.block_count)]
        self._hc_scales = {}
        self._routers = {}
        self._vha = {}
        self._eh_proj: dict[int, dict[str, Tensor]] = {}
        self._f32_sources: set[str] = set()
        for name, gen in self.model_tensors.items():
            if gen().dtype == torch.float32:
                self._f32_sources.add(name)

    def _is_dsa_layer(self, bid: int) -> bool:
        return self.hparams["compress_ratios"][bid] <= 0

    def set_vocab(self):
        tokens, toktypes, tokpre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(tokpre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(toktypes)

        # the post-processor prepends <|ERNIE_MASK|> (also the eos), not the config's bos <|ERNIE_BEGIN|>,
        # and the pair template's separator does not apply to single sequences
        special_vocab = gguf.SpecialVocab(self.dir_model, load_merges=True)
        special_vocab.special_token_ids["bos"] = tokens.index("<|ERNIE_MASK|>")
        special_vocab.add_special_token["bos"] = True
        special_vocab.add_special_token["sep"] = False
        special_vocab.add_to_gguf(self.gguf_writer)

    def get_vocab_base_pre(self, tokenizer) -> str:
        del tokenizer  # unused
        # digits in right-aligned groups of 3, CJK runs isolated, then the DeepSeek-V3 split
        return "ernie5"

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams
        n_layer = self.block_count

        # full (MLA) layers cache [kv latent | rope], SWA (HCA) layers a single 512 KV head
        for key in (gguf.Keys.Attention.KEY_LENGTH, gguf.Keys.Attention.VALUE_LENGTH):
            self.gguf_writer.remove_key(key.format(arch=self.gguf_writer.arch))
        self.gguf_writer.add_key_length(hparams["kv_lora_rank"] + hparams["hybrid_mla_qk_rope_head_dim"])
        self.gguf_writer.add_value_length(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length_swa(hparams["head_dim"])
        self.gguf_writer.add_value_length_swa(hparams["head_dim"])
        self.gguf_writer.add_rope_dimension_count(hparams["qk_rope_head_dim"])
        assert hparams["qk_rope_head_dim"] == hparams["hybrid_mla_qk_rope_head_dim"]

        # MLA layer dims
        assert hparams["hybrid_mla_num_attention_heads"] == hparams["num_attention_heads"]
        assert hparams["hybrid_mla_q_lora_rank"] == hparams["q_lora_rank"]
        assert hparams["hybrid_mla_kv_lora_rank"] == hparams["kv_lora_rank"]
        self.gguf_writer.add_kv_lora_rank(hparams["kv_lora_rank"])
        self.gguf_writer.add_key_length_mla(hparams["hybrid_mla_qk_nope_head_dim"] + hparams["hybrid_mla_qk_rope_head_dim"])
        self.gguf_writer.add_value_length_mla(hparams["hybrid_mla_v_head_dim"])

        self.gguf_writer.add_q_lora_rank(hparams["q_lora_rank"])
        self.gguf_writer.add_sliding_window(hparams["sliding_window"])
        self.gguf_writer.add_attention_output_group_count(hparams["o_groups"])
        self.gguf_writer.add_attention_output_lora_rank(hparams["o_lora_rank"])
        self.gguf_writer.add_attention_compress_ratios([max(r, 0) for r in hparams["compress_ratios"][:n_layer]])
        if n_layer > hparams["num_hidden_layers"]:
            self.gguf_writer.add_nextn_predict_layers(n_layer - hparams["num_hidden_layers"])
        self.gguf_writer.add_attention_compress_rope_freq_base(hparams["compress_rope_theta"])

        self.gguf_writer.add_indexer_head_count(hparams["index_n_heads"])
        self.gguf_writer.add_indexer_key_length(hparams["index_head_dim"])
        self.gguf_writer.add_indexer_top_k(hparams["index_topk"])

        self.gguf_writer.add_hyper_connection_count(hparams["hc_mult"])
        self.gguf_writer.add_hyper_connection_sinkhorn_iterations(hparams["hc_sinkhorn_iters"])
        self.gguf_writer.add_hyper_connection_epsilon(hparams["hc_eps"])

        assert hparams["moe_split_feature_routing"] and hparams["routed_scaling_factor_learnable"]
        self.gguf_writer.add_leading_dense_block_count(hparams["first_k_dense_replace"])
        self.gguf_writer.add_expert_feed_forward_length(hparams["moe_intermediate_size"])
        self.gguf_writer.add_expert_shared_count(hparams["n_shared_experts"])
        self.gguf_writer.add_expert_weights_norm(hparams["norm_topk_prob"])
        self.gguf_writer.add_expert_latent_length(hparams["moe_latent_size"])

        assert hparams["hidden_act"] == "situ"
        self.gguf_writer.add_activation_situ_beta(hparams["activation_situ_beta"])
        self.gguf_writer.add_activation_situ_linear_beta(hparams["activation_situ_linear_beta"])

    def _multimax_kv(self) -> None:
        ranges = LazyTorchTensor.to_eager(self.model_tensors["model.lm_head.multimax_ranges"]()).float().tolist()
        ts = LazyTorchTensor.to_eager(self.model_tensors["model.lm_head.multimax_ts"]()).float().tolist()
        self.gguf_writer.add_array(gguf.Keys.LLM.LOGIT_MULTIMAX.format(arch=self.gguf_writer.arch), ranges + ts)
        logger.info(f"gguf: logit multimax ranges = {ranges}, ts = {ts}")

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item

        if (m := re.match(r"model\.layers\.(\d+)\.", name)) is not None:
            # the MTP layer
            if cls.no_mtp and int(m.group(1)) >= cls.hparams_n_layer:
                return None

        return super().filter_tensors((name, gen))

    hparams_n_layer: int = 0

    def index_tensors(self, remote_hf_model_id: str | None = None) -> dict[str, Callable[[], Tensor]]:
        type(self).hparams_n_layer = self.hparams["num_hidden_layers"]
        return super().index_tensors(remote_hf_model_id=remote_hf_model_id)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if name in ("model.lm_head.multimax_ranges", "model.lm_head.multimax_ts"):
            if name.endswith("_ranges"):
                self._multimax_kv()
            return

        root: dict[str, gguf.MODEL_TENSOR] = {
            "model.embed_tokens.weight": gguf.MODEL_TENSOR.TOKEN_EMBD,
            "model.norm.weight":         gguf.MODEL_TENSOR.OUTPUT_NORM,
            "lm_head.weight":            gguf.MODEL_TENSOR.OUTPUT,
            "model.mhc_contract.hc_head_base":  gguf.MODEL_TENSOR.HC_HEAD_BASE,
            "model.mhc_contract.hc_head_scale": gguf.MODEL_TENSOR.HC_HEAD_SCALE,
        }
        if name in root:
            yield self.format_tensor_name(root[name]), data_torch
            return
        if name == "model.mhc_contract.hc_head_fn":
            # stored [in, out]
            yield self.format_tensor_name(gguf.MODEL_TENSOR.HC_HEAD_FN), data_torch.t().contiguous()
            return

        assert bid is not None, name
        prefix = f"model.layers.{bid}."
        assert name.startswith(prefix), name
        # the MTP layer nests its mHC under transformer_layer.
        key = name[len(prefix):].removeprefix("transformer_layer.")

        # the MTP layer's own pieces
        nextn: dict[str, gguf.MODEL_TENSOR] = {
            "enorm.weight":             gguf.MODEL_TENSOR.NEXTN_ENORM,
            "hnorm.weight":             gguf.MODEL_TENSOR.NEXTN_HNORM,
            "shared_head.norm.weight":  gguf.MODEL_TENSOR.NEXTN_SHARED_HEAD_NORM,
            "hc_head_base":             gguf.MODEL_TENSOR.NEXTN_HC_HEAD_BASE,
            "hc_head_scale":            gguf.MODEL_TENSOR.NEXTN_HC_HEAD_SCALE,
        }
        if key in nextn:
            yield self.format_tensor_name(nextn[key], bid, ".weight"), data_torch
            return
        if key == "hc_head_fn":
            # stored [in, out], like the trunk's
            yield self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_HC_HEAD_FN, bid, ".weight"), data_torch.t().contiguous()
            return
        if key in ("e_proj.weight", "h_proj.weight"):
            # h_proj(hnorm(h)) + e_proj(enorm(e)) = eh_proj [enorm(e) | hnorm(h)]
            eh = self._eh_proj.setdefault(bid, {})
            eh[key[0]] = data_torch
            if len(eh) == 2:
                del self._eh_proj[bid]
                yield self.format_tensor_name(gguf.MODEL_TENSOR.NEXTN_EH_PROJ, bid, ".weight"), torch.cat([eh["e"], eh["h"]], dim=1)
            return

        # mHC
        if (m := re.match(r"(self_attention|mlp)_hyper_connection\.(.+)$", key)) is not None:
            kind = "ATTN" if m.group(1) == "self_attention" else "FFN"
            leaf = m.group(2)
            if leaf == "mapping_proj.weight":
                yield self.format_tensor_name(getattr(gguf.MODEL_TENSOR, f"HC_{kind}_FN"), bid), data_torch.t().contiguous()
            elif leaf == "bias":
                yield self.format_tensor_name(getattr(gguf.MODEL_TENSOR, f"HC_{kind}_BASE"), bid), data_torch
            else:
                # alpha_pre, alpha_post, alpha_res -> scale[0..2]
                idx = ("alpha_pre", "alpha_post", "alpha_res").index(leaf)
                scales = self._hc_scales.setdefault(f"{bid}.{kind}", {})
                scales[idx] = data_torch
                if len(scales) == 3:
                    del self._hc_scales[f"{bid}.{kind}"]
                    data = torch.cat([scales[i].reshape(1) for i in range(3)])
                    yield self.format_tensor_name(getattr(gguf.MODEL_TENSOR, f"HC_{kind}_SCALE"), bid), data
            return

        # VHA: o[h] <- sum_g m[h, g] * o[g] with m = I + V U^T
        if key in ("self_attn.vha_postmix_U", "self_attn.vha_postmix_V"):
            vha = self._vha.setdefault(bid, {})
            vha[key[-1]] = data_torch
            if len(vha) == 2:
                del self._vha[bid]
                u = LazyTorchTensor.to_eager(vha["U"]).float()
                v = LazyTorchTensor.to_eager(vha["V"]).float()
                m = torch.eye(u.shape[0], dtype=torch.float32) + v @ u.t()
                yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_VHA, bid), m
            return

        # MLA (DSA) layer: split kv_b for the absorbed form
        if key == "self_attn.kv_b_proj.weight":
            n_head = self.hparams["hybrid_mla_num_attention_heads"]
            qk_nope = self.hparams["hybrid_mla_qk_nope_head_dim"]
            v_dim = self.hparams["hybrid_mla_v_head_dim"]
            kv_b = data_torch.view(n_head, qk_nope + v_dim, data_torch.shape[-1])
            k_b, v_b = torch.split(kv_b, [qk_nope, v_dim], dim=1)
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_K_B, bid), k_b.transpose(1, 2).contiguous()
            yield self.format_tensor_name(gguf.MODEL_TENSOR.ATTN_V_B, bid), v_b.contiguous()
            return

        # MoE: two routers, stacked into one [2E, H] matrix
        if key in ("block_sparse_moe.gate.weight", "block_sparse_moe.gate.weight_1"):
            routers = self._routers.setdefault(bid, {})
            routers[0 if key.endswith(".weight") else 1] = data_torch
            if len(routers) == 2:
                del self._routers[bid]
                yield self.format_tensor_name(gguf.MODEL_TENSOR.FFN_GATE_INP, bid), torch.cat([routers[0], routers[1]], dim=0)
            return

        if (m := re.match(r"block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$", key)) is not None:
            n_experts = self.hparams["n_routed_experts"]
            assert self._experts is not None
            self._experts[bid][key] = data_torch
            if len(self._experts[bid]) >= n_experts * 3:
                for w_name, tensor in (("w1", gguf.MODEL_TENSOR.FFN_GATE_EXP),
                                       ("w2", gguf.MODEL_TENSOR.FFN_DOWN_EXP),
                                       ("w3", gguf.MODEL_TENSOR.FFN_UP_EXP)):
                    datas: list[Tensor] = []
                    for xid in range(n_experts):
                        datas.append(self._experts[bid].pop(f"block_sparse_moe.experts.{xid}.{w_name}.weight"))
                    yield self.format_tensor_name(tensor, bid), torch.stack(datas, dim=0)
            return

        is_dsa = self._is_dsa_layer(bid)
        layer_map: dict[str, tuple[gguf.MODEL_TENSOR, str]] = {
            "input_layernorm.weight":          (gguf.MODEL_TENSOR.ATTN_NORM, ".weight"),
            "post_attention_layernorm.weight": (gguf.MODEL_TENSOR.FFN_NORM,  ".weight"),
            "self_attn.gate_proj.weight":      (gguf.MODEL_TENSOR.ATTN_GATE, ".weight"),
            # HCA
            "self_attn.linear_q_down_proj.weight": (gguf.MODEL_TENSOR.ATTN_Q_A,      ".weight"),
            "self_attn.q_layernorm.weight":        (gguf.MODEL_TENSOR.ATTN_Q_A_NORM, ".weight"),
            "self_attn.linear_q_up_proj.weight":   (gguf.MODEL_TENSOR.ATTN_Q_B,      ".weight"),
            "self_attn.linear_kv_proj.weight":     (gguf.MODEL_TENSOR.ATTN_KV,       ".weight"),
            "self_attn.kv_layernorm.weight":       (gguf.MODEL_TENSOR.ATTN_KV_NORM,  ".weight"),
            "self_attn.linear_o_group_proj":       (gguf.MODEL_TENSOR.ATTN_OUT_A,    ".weight"),
            "self_attn.core_attention.attn_sink":  (gguf.MODEL_TENSOR.ATTN_SINKS,    ".weight"),
            "self_attn.core_attention.compressor.ape":                (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_APE,   ".weight"),
            "self_attn.core_attention.compressor.linear_wkv.weight":  (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_WKV,   ".weight"),
            "self_attn.core_attention.compressor.linear_wgate.weight": (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_WGATE, ".weight"),
            "self_attn.core_attention.compressor.norm.weight":        (gguf.MODEL_TENSOR.ATTN_COMPRESSOR_NORM,  ".weight"),
            # MLA (DSA)
            "self_attn.q_a_proj.weight":           (gguf.MODEL_TENSOR.ATTN_Q_A,       ".weight"),
            "self_attn.q_a_layernorm.weight":      (gguf.MODEL_TENSOR.ATTN_Q_A_NORM,  ".weight"),
            "self_attn.q_b_proj.weight":           (gguf.MODEL_TENSOR.ATTN_Q_B,       ".weight"),
            "self_attn.kv_a_proj_with_mqa.weight": (gguf.MODEL_TENSOR.ATTN_KV_A_MQA,  ".weight"),
            "self_attn.kv_a_layernorm.weight":     (gguf.MODEL_TENSOR.ATTN_KV_A_NORM, ".weight"),
            "self_attn.core_attention.softmax_offset":                   (gguf.MODEL_TENSOR.ATTN_SINKS,       ".weight"),
            "self_attn.core_attention.indexer.wq_b.weight":              (gguf.MODEL_TENSOR.INDEXER_ATTN_Q_B, ".weight"),
            "self_attn.core_attention.indexer.wk.weight":                (gguf.MODEL_TENSOR.INDEXER_ATTN_K,   ".weight"),
            "self_attn.core_attention.indexer.k_norm.weight":            (gguf.MODEL_TENSOR.INDEXER_K_NORM,   ".weight"),
            "self_attn.core_attention.indexer.k_norm.bias":              (gguf.MODEL_TENSOR.INDEXER_K_NORM,   ".bias"),
            "self_attn.core_attention.indexer.weights_proj.weight":      (gguf.MODEL_TENSOR.INDEXER_PROJ,     ".weight"),
            # dense FFN
            "mlp.gate_proj.weight": (gguf.MODEL_TENSOR.FFN_GATE, ".weight"),
            "mlp.up_proj.weight":   (gguf.MODEL_TENSOR.FFN_UP,   ".weight"),
            "mlp.down_proj.weight": (gguf.MODEL_TENSOR.FFN_DOWN, ".weight"),
            # latent MoE
            "block_sparse_moe.e_score_correction.bias":          (gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, ".bias"),  # renamed by the base filter
            "block_sparse_moe.gate.routed_scaling_factor_param": (gguf.MODEL_TENSOR.FFN_EXP_SCALE,   ".weight"),
            "block_sparse_moe.fc1_latent_proj.weight": (gguf.MODEL_TENSOR.FFN_ROUTED_DOWN, ".weight"),
            "block_sparse_moe.fc2_latent_proj.weight": (gguf.MODEL_TENSOR.FFN_ROUTED_UP,   ".weight"),
            "block_sparse_moe.latent_norm.weight":     (gguf.MODEL_TENSOR.FFN_ROUTED_NORM, ".weight"),
            "block_sparse_moe.shared_experts.w1.weight": (gguf.MODEL_TENSOR.FFN_GATE_SHEXP, ".weight"),
            "block_sparse_moe.shared_experts.w2.weight": (gguf.MODEL_TENSOR.FFN_DOWN_SHEXP, ".weight"),
            "block_sparse_moe.shared_experts.w3.weight": (gguf.MODEL_TENSOR.FFN_UP_SHEXP,   ".weight"),
        }
        if key == "self_attn.o_proj.weight":
            tensor, suffix = (gguf.MODEL_TENSOR.ATTN_OUT if is_dsa else gguf.MODEL_TENSOR.ATTN_OUT_B), ".weight"
        elif key in layer_map:
            tensor, suffix = layer_map[key]
        else:
            raise ValueError(f"Unsupported ERNIE 5.5 tensor {name!r}")

        if tensor == gguf.MODEL_TENSOR.ATTN_SINKS:
            data_torch = data_torch.float()

        yield self.format_tensor_name(tensor, bid, suffix), data_torch

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        # mHC mixers, the APE, the router, sinks and the VHA mix are fp32 in the checkpoint or computed in fp32
        if name in self._f32_sources or new_name.endswith((".attn_vha.weight", ".attn_sinks.weight")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def prepare_tensors(self):
        super().prepare_tensors()

        leftovers = [k for d in (self._experts or []) for k in d.keys()]
        leftovers += list(self._hc_scales.keys()) + [str(b) for b in self._routers] + [str(b) for b in self._vha]
        leftovers += [f"{b}.eh_proj" for b in self._eh_proj]
        if leftovers:
            raise ValueError(f"Unprocessed ERNIE 5.5 tensors: {leftovers[:8]}")
