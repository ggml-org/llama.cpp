from __future__ import annotations

import re
from typing import Iterable

import torch

from .base import ModelBase, gguf, logger
from .deepseek import DeepseekV2Model


def split_kv_b_proj(weight: torch.Tensor, n_head: int, qk_nope: int, v_head_dim: int):
    """Split kv_b_proj into k_b (transposed) and v_b, matching DeepSeek MLA absorption.

    weight: [n_head*(qk_nope+v_head_dim), kv_lora_rank].
    Returns (k_b, v_b): k_b [n_head, kv_lora_rank, qk_nope], v_b [n_head, v_head_dim, kv_lora_rank].
    """
    kv_lora = weight.shape[-1]
    assert weight.shape[0] == n_head * (qk_nope + v_head_dim)
    kv_b = weight.view(n_head, qk_nope + v_head_dim, kv_lora)
    k_b, v_b = torch.split(kv_b, [qk_nope, v_head_dim], dim=1)
    k_b = k_b.transpose(1, 2).contiguous()  # [n_head, kv_lora, qk_nope]
    return k_b, v_b.contiguous()


def split_gate_up(weight: torch.Tensor, moe_intermediate_size: int):
    """Split a fused stacked gate_up expert tensor into (gate, up).

    weight: [n_expert, 2*moe_intermediate_size, hidden] (gate first, up second).
    Returns (gate, up) each [n_expert, moe_intermediate_size, hidden].
    """
    assert weight.shape[1] == 2 * moe_intermediate_size, f"{weight.shape[1]} != 2*{moe_intermediate_size}"
    gate = weight[:, :moe_intermediate_size, :].contiguous()
    up = weight[:, moe_intermediate_size:, :].contiguous()
    return gate, up


@ModelBase.register("HYV4ForCausalLM")
class HYV4Model(DeepseekV2Model):
    """HY_V4: DeepSeek-V3 style MLA + MoE with iHC, a gated MLA output and a learnable sink.

    Reuses DeepseekV2Model for the vocab and the MLA metadata, but overrides the tensor mapping
    because HY_V4 ships pre-stacked / fused experts plus extra iHC, gate and sink tensors. The
    rope rows are mapped straight through (no permute) - the graph rotates consecutive pairs.

    DSA is supported: indexer weights are exported for the layers marked "full" in indexer_types.
    "shared" layers reuse the top-k of the last preceding full layer at inference time, so they
    carry no indexer weights.

    MTP (num_nextn_predict_layers) is dropped, so the GGUF cannot be used for speculative
    decoding. The reference only runs the MTP layers while training or while speculating, so they
    cannot change single-token logits.
    """

    model_arch = gguf.MODEL_ARCH.HY_V4

    # tensors a "full" indexer layer must carry
    INDEXER_SUFFIXES = frozenset({
        "self_attn.indexer.wq_b.weight",
        "self_attn.indexer.wk.weight",
        "self_attn.indexer.k_norm.weight",
        "self_attn.indexer.k_norm.bias",
        "self_attn.indexer.weights_proj.weight",
    })

    @classmethod
    def filter_tensors(cls, item):
        # drop MTP here, not in modify_tensors, so the weights are never read
        if item[0].startswith("model.mtp_layers."):
            return None
        return super().filter_tensors(item)

    def _check_indexer_hparams(self):
        for key in ("index_n_heads", "index_head_dim", "index_topk"):
            if key not in self.hparams:
                raise ValueError(f"HY_V4 has DSA layers but no {key}")

    def indexer_is_full(self) -> list[bool] | None:
        """Per-layer indexer ownership, or None when the checkpoint has no DSA.

        indexer_types entries are "full" (owns an indexer) or "shared" (reuses the preceding
        full layer's top-k). Missing indexer_types with sparse layers means every sparse layer
        owns one.
        """
        hparams = self.hparams
        n_layer = hparams["num_hidden_layers"]
        indexer_types = hparams.get("indexer_types")

        # the reference drives DSA off indexer_types alone; layer_types is only a fallback for
        # checkpoints predating it (it was renamed to deepseek_sparse_attention upstream)
        if indexer_types is None:
            layer_types = hparams.get("layer_types") or []
            sparse = {"sparse_attention", "deepseek_sparse_attention"}
            if not any(t in sparse for t in layer_types):
                return None
            if len(layer_types) < n_layer:
                raise ValueError(f"HY_V4 layer_types has {len(layer_types)} entries, need {n_layer}")
            self._check_indexer_hparams()
            return [t in sparse for t in layer_types[:n_layer]]

        self._check_indexer_hparams()

        if len(indexer_types) < n_layer:
            raise ValueError(f"HY_V4 indexer_types has {len(indexer_types)} entries, need {n_layer}")
        unknown = {t for t in indexer_types[:n_layer]} - {"full", "shared"}
        if unknown:
            raise ValueError(f"HY_V4 unknown indexer_types values: {sorted(unknown)}")
        is_full = [t == "full" for t in indexer_types[:n_layer]]
        if is_full and not is_full[0]:
            raise ValueError("HY_V4 layer 0 must be indexer_types 'full' (nothing precedes it to share)")
        return is_full

    def set_gguf_parameters(self):
        hparams = self.hparams

        # HY4 has n_group == topk_group == 1 (no group routing). Drop the keys so the base does
        # not emit expert_group_count/used; llama.cpp then takes the ungrouped MoE path.
        if hparams.get("n_group") == 1 and hparams.get("topk_group") == 1:
            hparams.pop("n_group", None)
            hparams.pop("topk_group", None)

        # HY_V4 config expresses dense/sparse layers via mlp_layer_types, but DeepseekV2Model
        # needs first_k_dense_replace. Derive it as the contiguous leading "dense" block
        # (the real config.json also carries first_k_dense_replace; prefer it when present,
        # but assert the two agree so a mismatch fails loudly).
        mlp_types = hparams.get("mlp_layer_types")
        explicit = hparams.get("first_k_dense_replace")
        derived = None
        if mlp_types is not None:
            lead = 0
            for t in mlp_types:
                if t == "dense":
                    lead += 1
                else:
                    break
            if any(t == "dense" for t in mlp_types[lead:]):
                raise NotImplementedError("HY_V4 converter expects a contiguous leading dense block")
            derived = lead
        if explicit is not None and derived is not None and explicit != derived:
            raise ValueError(
                f"HY_V4 first_k_dense_replace ({explicit}) disagrees with mlp_layer_types "
                f"leading-dense count ({derived})"
            )
        if explicit is None:
            if derived is None:
                raise ValueError("HY_V4 needs first_k_dense_replace or mlp_layer_types to place dense layers")
            hparams["first_k_dense_replace"] = derived

        # reuse DeepseekV2 MLA + MoE metadata (forces num_key_value_heads=1, writes q/kv lora,
        # key/value lengths, expert counts, weights scale/norm, rope dims, etc.)
        super().set_gguf_parameters()

        # HY4 uses DeepSeek-V3 sigmoid routing with e_score_correction_bias. The config has no
        # scoring_func key, so the base does not write a gating func; set it explicitly.
        self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)

        # routed-expert SwiGLU logits clamp (only routed experts; shared/dense are not clamped,
        # so swiglu_clamp_shexp is intentionally not written). 0.0 disables the clamp.
        swiglu_limit = float(hparams.get("swiglu_limit", 0.0) or 0.0)
        if swiglu_limit > 0.0:
            self.gguf_writer.add_swiglu_clamp_exp([swiglu_limit] * self.block_count)

        # iHC (independent Hyper-Connections)
        self.gguf_writer.add_hyper_connection_count(hparams["hc_mult"])
        self.gguf_writer.add_hyper_connection_epsilon(hparams["hc_eps"])
        self.gguf_writer.add_hyper_connection_magnitude(hparams["hc_magnitude"])

        # is_full is written explicitly; the graph must not infer it from tensor presence
        is_full = self.indexer_is_full()
        if is_full is not None:
            self.gguf_writer.add_indexer_head_count(hparams["index_n_heads"])
            self.gguf_writer.add_indexer_key_length(hparams["index_head_dim"])
            self.gguf_writer.add_indexer_top_k(hparams["index_topk"])
            self.gguf_writer.add_indexer_types(is_full)
            logger.info(
                "HY_V4 DSA: %d/%d layers own an indexer (top_k=%d, n_heads=%d, head_dim=%d)",
                sum(is_full), len(is_full), hparams["index_topk"],
                hparams["index_n_heads"], hparams["index_head_dim"],
            )

        if hparams.get("num_nextn_predict_layers", 0):
            logger.warning(
                "HY_V4: dropping %d MTP (nextn) layer(s) - the reference runs them only under "
                "training / speculative decoding. This GGUF cannot be used for speculative decoding.",
                hparams["num_nextn_predict_layers"],
            )

    def prepare_tensors(self):
        # validate before the base materializes tensors, so a mismatch fails early
        is_full = self.indexer_is_full()
        if is_full is not None:
            present: dict[int, set[str]] = {}
            for name in self.model_tensors:
                m = re.match(r"model\.layers\.(\d+)\.(self_attn\.indexer\..+)$", name)
                if m:
                    present.setdefault(int(m.group(1)), set()).add(m.group(2))
            for il, expect_full in enumerate(is_full):
                seen = present.get(il, set())
                if expect_full and seen != self.INDEXER_SUFFIXES:
                    raise ValueError(
                        f"HY_V4 layer {il} is indexer_types 'full' but is missing indexer tensors: "
                        f"{sorted(self.INDEXER_SUFFIXES - seen)}"
                    )
                if not expect_full and seen:
                    raise ValueError(
                        f"HY_V4 layer {il} is indexer_types 'shared' but carries indexer tensors: "
                        f"{sorted(seen)}"
                    )

        super().prepare_tensors()

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        # iHC mixing matrices are 2D .weight tensors that the reference keeps in fp32
        # (_keep_in_fp32_modules_strict). 1D tensors (hc_base/scale, attn_sinks,
        # e_score_correction_bias) and the router (FFN_GATE_INP) are already forced F32 by the
        # base rules. Force the HC *_fn matrices here.
        if new_name.endswith(("hc_attn_fn.weight", "hc_ffn_fn.weight", "output_hc_fn.weight")):
            return gguf.GGMLQuantizationType.F32
        # indexer k_norm is fp32 in the reference; the base rules already cover
        # *_norm.weight and INDEXER_PROJ, but not this bias
        if self.match_model_tensor_name(new_name, gguf.MODEL_TENSOR.INDEXER_K_NORM, bid, suffix=".bias"):
            return gguf.GGMLQuantizationType.F32
        # enable_lm_head_fp32: mirror the reference fp32 LM-head matmul by keeping output F32.
        if new_name == "output.weight" and self.hparams.get("enable_lm_head_fp32", False):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: torch.Tensor, name: str, bid: int | None) -> Iterable[tuple[str, torch.Tensor]]:
        hparams = self.hparams
        n_head = hparams["num_attention_heads"]
        qk_nope = hparams["qk_nope_head_dim"]
        v_head_dim = hparams["v_head_dim"]
        moe_inter = hparams["moe_intermediate_size"]

        tn = self.format_tensor_name

        # ---- global (non per-layer) ----
        if name == "model.embed_tokens.weight":
            return [(tn(gguf.MODEL_TENSOR.TOKEN_EMBD), data_torch)]
        if name == "model.norm.weight":
            return [(tn(gguf.MODEL_TENSOR.OUTPUT_NORM), data_torch)]
        if name == "lm_head.weight":
            return [(tn(gguf.MODEL_TENSOR.OUTPUT), data_torch)]
        if name == "model.hc_head.hc_head_fn":
            return [(tn(gguf.MODEL_TENSOR.HC_HEAD_FN), data_torch)]
        if name == "model.hc_head.hc_head_base":
            return [(tn(gguf.MODEL_TENSOR.HC_HEAD_BASE), data_torch)]
        if name == "model.hc_head.hc_head_scale":
            return [(tn(gguf.MODEL_TENSOR.HC_HEAD_SCALE), data_torch)]

        assert bid is not None, f"expected a per-layer tensor, got {name!r}"

        # ---- per-layer, keyed by suffix after 'model.layers.{bid}.' ----
        suffix = name.split(f"model.layers.{bid}.", 1)[-1]

        # note: q_b_proj and kv_a_proj_with_mqa are mapped straight through (no RoPE permute),
        # the graph rotates consecutive pairs so the rows need no reordering
        simple = {
            "input_layernorm.weight":          (gguf.MODEL_TENSOR.ATTN_NORM, ".weight"),
            "post_attention_layernorm.weight": (gguf.MODEL_TENSOR.FFN_NORM,  ".weight"),
            "self_attn.q_a_proj.weight":       (gguf.MODEL_TENSOR.ATTN_Q_A, ".weight"),
            "self_attn.q_a_layernorm.weight":  (gguf.MODEL_TENSOR.ATTN_Q_A_NORM, ".weight"),
            "self_attn.q_b_proj.weight":       (gguf.MODEL_TENSOR.ATTN_Q_B, ".weight"),
            "self_attn.kv_a_proj_with_mqa.weight": (gguf.MODEL_TENSOR.ATTN_KV_A_MQA, ".weight"),
            "self_attn.kv_a_layernorm.weight": (gguf.MODEL_TENSOR.ATTN_KV_A_NORM, ".weight"),
            "self_attn.o_proj.weight":         (gguf.MODEL_TENSOR.ATTN_OUT, ".weight"),
            "self_attn.linear_gate.weight":    (gguf.MODEL_TENSOR.ATTN_GATE, ".weight"),
            "self_attn.learnable_sink_param":  (gguf.MODEL_TENSOR.ATTN_SINKS, ".weight"),
            "self_attn.indexer.wq_b.weight":   (gguf.MODEL_TENSOR.INDEXER_ATTN_Q_B, ".weight"),
            "self_attn.indexer.wk.weight":     (gguf.MODEL_TENSOR.INDEXER_ATTN_K, ".weight"),
            "self_attn.indexer.k_norm.weight": (gguf.MODEL_TENSOR.INDEXER_K_NORM, ".weight"),
            "self_attn.indexer.k_norm.bias":   (gguf.MODEL_TENSOR.INDEXER_K_NORM, ".bias"),
            "self_attn.indexer.weights_proj.weight": (gguf.MODEL_TENSOR.INDEXER_PROJ, ".weight"),
            "hc_attn_layer.hc_pre.hc_fn":      (gguf.MODEL_TENSOR.HC_ATTN_FN, ".weight"),
            "hc_attn_layer.hc_pre.hc_base":    (gguf.MODEL_TENSOR.HC_ATTN_BASE, ".weight"),
            "hc_attn_layer.hc_pre.hc_scale":   (gguf.MODEL_TENSOR.HC_ATTN_SCALE, ".weight"),
            "hc_mlp_layer.hc_pre.hc_fn":       (gguf.MODEL_TENSOR.HC_FFN_FN, ".weight"),
            "hc_mlp_layer.hc_pre.hc_base":     (gguf.MODEL_TENSOR.HC_FFN_BASE, ".weight"),
            "hc_mlp_layer.hc_pre.hc_scale":    (gguf.MODEL_TENSOR.HC_FFN_SCALE, ".weight"),
            "mlp.gate.weight":                 (gguf.MODEL_TENSOR.FFN_GATE_INP, ".weight"),
            "mlp.gate.e_score_correction.bias":(gguf.MODEL_TENSOR.FFN_EXP_PROBS_B, ".bias"),
            "mlp.gate_proj.weight":            (gguf.MODEL_TENSOR.FFN_GATE, ".weight"),
            "mlp.up_proj.weight":              (gguf.MODEL_TENSOR.FFN_UP, ".weight"),
            "mlp.down_proj.weight":            (gguf.MODEL_TENSOR.FFN_DOWN, ".weight"),
            "mlp.shared_experts.gate_proj.weight": (gguf.MODEL_TENSOR.FFN_GATE_SHEXP, ".weight"),
            "mlp.shared_experts.up_proj.weight":   (gguf.MODEL_TENSOR.FFN_UP_SHEXP, ".weight"),
            "mlp.shared_experts.down_proj.weight": (gguf.MODEL_TENSOR.FFN_DOWN_SHEXP, ".weight"),
        }
        if suffix in simple:
            key, sfx = simple[suffix]
            return [(tn(key, bid, sfx), data_torch)]

        # kv_b_proj: split into k_b (transposed) and v_b
        if suffix == "self_attn.kv_b_proj.weight":
            k_b, v_b = split_kv_b_proj(data_torch, n_head, qk_nope, v_head_dim)
            return [
                (tn(gguf.MODEL_TENSOR.ATTN_K_B, bid), k_b),
                (tn(gguf.MODEL_TENSOR.ATTN_V_B, bid), v_b),
            ]

        # fused stacked experts: split gate_up into gate/up
        if suffix == "mlp.experts.gate_up_proj":
            gate, up = split_gate_up(data_torch, moe_inter)
            return [
                (tn(gguf.MODEL_TENSOR.FFN_GATE_EXP, bid), gate),
                (tn(gguf.MODEL_TENSOR.FFN_UP_EXP, bid), up),
            ]
        if suffix == "mlp.experts.down_proj":
            return [(tn(gguf.MODEL_TENSOR.FFN_DOWN_EXP, bid), data_torch)]

        raise ValueError(f"Unsupported HY_V4 tensor {name!r} (suffix {suffix!r})")
