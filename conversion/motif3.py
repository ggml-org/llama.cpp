from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

from .base import ModelBase, TextModel, gguf, logger

if TYPE_CHECKING:
    from torch import Tensor


@ModelBase.register("MotifForCausalLM")
class Motif3Model(TextModel):
    """Motif-3 GDLA attention + grouped-PolyNorm MoE + mHC.
    ref: https://huggingface.co/Motif-Technologies/Motif-3-Beta
    """

    model_arch = gguf.MODEL_ARCH.MOTIF3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # collect alpha_{pre,post,res} scalars into a single [3] tensor per mHC block
        self._mhc_alpha: dict[str, dict[str, Tensor]] = {}

    def get_vocab_base_pre(self, tokenizer) -> str:
        try:
            return super().get_vocab_base_pre(tokenizer)
        except NotImplementedError:
            logger.warning(
                "unrecognized pre-tokenizer (hash not in convert_hf_to_gguf_update.py); "
                "falling back to 'gpt-2'. Tokenization of unusual strings may differ slightly - "
                "consider registering the proper pre-tokenizer for production use.")
            return "gpt-2"

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        super().set_gguf_parameters()
        hparams = self.hparams

        # GDL Attention
        qk_rope = int(hparams["qk_rope_head_dim"])
        head_dim = int(hparams["head_dim"])       # qk_nope + qk_rope
        v_head_dim = int(hparams["v_head_dim"])

        self.gguf_writer.add_key_length(head_dim)
        self.gguf_writer.add_value_length(v_head_dim)
        self.gguf_writer.add_rope_dimension_count(qk_rope)
        self.gguf_writer.add_q_lora_rank(int(hparams["q_lora_rank"]))
        self.gguf_writer.add_kv_lora_rank(int(hparams["kv_lora_rank"]))

        n_noise = int(hparams["num_noise_heads"])
        self.gguf_writer.add_uint32(f"{self.gguf_writer.arch}.attention.noise_head_count", n_noise)

        # RoPE / YaRN
        rope_theta = float(hparams.get("rope_theta", 10000.0))
        self.gguf_writer.add_rope_freq_base(rope_theta)

        rope_scaling = hparams.get("rope_scaling") or {}
        rope_factor = float(rope_scaling.get("factor", hparams.get("rope_factor", 1.0)))
        if rope_factor > 1.0:
            orig_ctx = int(rope_scaling.get(
                "original_max_position_embeddings",
                hparams.get("original_seq_len", 32768)))
            self.gguf_writer.add_rope_scaling_type(gguf.RopeScalingType.YARN)
            self.gguf_writer.add_rope_scaling_factor(rope_factor)
            self.gguf_writer.add_rope_scaling_orig_ctx_len(orig_ctx)
            mscale = float(rope_scaling.get("mscale", hparams.get("mscale", 1.0)))
            self.gguf_writer.add_float32(f"{self.gguf_writer.arch}.attention.yarn_mscale", mscale)

        # interleaved SWA
        if hparams.get("use_sliding_window") and hparams.get("sliding_window") is not None:
            self.gguf_writer.add_sliding_window(int(hparams["sliding_window"]) + 1)
            pattern = hparams.get("sliding_window_pattern", "interleave")
            if pattern != "interleave":
                raise ValueError(f"unsupported sliding_window_pattern: {pattern}")
            self.gguf_writer.add_sliding_window_pattern(int(hparams.get("sliding_window_period", 2)))
            swa_rope_theta = hparams.get("swa_rope_theta")
            if swa_rope_theta is not None:
                self.gguf_writer.add_rope_freq_base_swa(float(swa_rope_theta))

        # MoE
        n_expert = int(hparams.get("num_experts", 0) or 0)
        if n_expert > 0:
            if hparams.get("score_func", "sigmoid") != "sigmoid":
                raise ValueError("only score_func == sigmoid is supported")
            if hparams.get("score_before_experts"):
                raise ValueError("score_before_experts == True is not supported")
            moe_intermediate = int(hparams.get("moe_intermediate_size", hparams["intermediate_size"]))
            self.gguf_writer.add_expert_count(n_expert)
            self.gguf_writer.add_expert_used_count(int(hparams["experts_top_k"]))
            self.gguf_writer.add_expert_feed_forward_length(moe_intermediate)
            self.gguf_writer.add_expert_shared_feed_forward_length(moe_intermediate)
            self.gguf_writer.add_expert_shared_count(int(hparams.get("num_shared_experts", 0)))
            self.gguf_writer.add_expert_weights_scale(float(hparams.get("route_scale", 1.0)))
            self.gguf_writer.add_expert_weights_norm(bool(hparams.get("route_norm", False)))
            self.gguf_writer.add_expert_gating_func(gguf.ExpertGatingFuncType.SIGMOID)
            self.gguf_writer.add_leading_dense_block_count(int(hparams.get("n_dense_first_layers", 0)))
            self.gguf_writer.add_interleave_moe_layer_step(int(hparams.get("interleave_moe_layer_step", 1)))

        # PolyNorm
        arch = self.gguf_writer.arch
        self.gguf_writer.add_float32(f"{arch}.polynorm.epsilon", 1e-6)
        self.gguf_writer.add_float32(f"{arch}.polynorm.output_scale", float(hparams.get("polynorm_output_scale", 1.0)))
        bias_clamp = hparams.get("polynorm_bias_clamp")
        if bias_clamp is not None:
            self.gguf_writer.add_float32(f"{arch}.polynorm.bias_clamp", float(bias_clamp))
        hidden_clamp = hparams.get("hidden_clamp")
        if hidden_clamp is not None:
            self.gguf_writer.add_float32(f"{arch}.polynorm.hidden_clamp", float(hidden_clamp))
        self.gguf_writer.add_bool(f"{arch}.polynorm.sigmoid_weight", bool(hparams.get("polynorm_sigmoid_weight", True)))

        # mHC
        if hparams.get("mhc_enabled"):
            self.gguf_writer.add_uint32(f"{arch}.hyper_connection.count", int(hparams.get("mhc_expansion_rate", 4)))
            self.gguf_writer.add_uint32(f"{arch}.hyper_connection.sinkhorn_iterations", int(hparams.get("mhc_sinkhorn_iters", 20)))
            post_coeff = 1.0 + float(hparams.get("mhc_h_post_alpha_end", 0.0))
            self.gguf_writer.add_float32(f"{arch}.hyper_connection.h_post_coeff", post_coeff)

    _SMALL_F32_KEYWORDS = (
        "ffn_gate_inp", "exp_probs_b",
        "ffn_poly", "mhc_",
        "attn_lambda",
    )

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        del name, bid, n_dims
        if any(k in new_name for k in self._SMALL_F32_KEYWORDS):
            return gguf.GGMLQuantizationType.F32
        return False

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        T = gguf.MODEL_TENSOR

        # skip the MTP (multi-token-prediction) head layer(s)
        if bid is not None and bid >= self.block_count:
            logger.debug(f"skipping MTP tensor {name}")
            return []
        if name.endswith(".rotary_emb.inv_freq"):
            return []

        def out(t: gguf.MODEL_TENSOR, tensor: Tensor, suffix: str = ".weight"):
            return [(self.format_tensor_name(t, bid, suffix=suffix), tensor)]

        # global tensors
        if name == "model.embed_tokens.weight":
            return out(T.TOKEN_EMBD, data_torch)
        if name == "model.norm.weight":
            return out(T.OUTPUT_NORM, data_torch)
        if name == "lm_head.weight":
            return out(T.OUTPUT, data_torch)

        assert bid is not None, f"unexpected non-layer tensor: {name}"

        prefix = f"model.layers.{bid}."
        assert name.startswith(prefix), f"unexpected tensor: {name}"
        sub = name[len(prefix):]

        # layer norms
        if sub == "input_layernorm.weight":
            return out(T.ATTN_NORM, data_torch)
        if sub == "post_attention_layernorm.weight":
            return out(T.FFN_NORM, data_torch)

        # GDL attention
        attn_map = {
            "self_attn.wq_a.weight":        T.ATTN_Q_A,
            "self_attn.q_norm.weight":      T.ATTN_Q_A_NORM,
            "self_attn.wq_b.weight":        T.ATTN_Q_B,
            "self_attn.wq_b_gate.weight":   T.ATTN_GATE,
            "self_attn.wkv_a.weight":       T.ATTN_KV_A_MQA,
            "self_attn.kv_norm.weight":     T.ATTN_KV_A_NORM,
            "self_attn.wkv_b.weight":       T.ATTN_KV_B,
            "self_attn.lambda_proj.weight": T.ATTN_LAMBDA,
            "self_attn.wo.weight":          T.ATTN_OUT,
        }
        if sub in attn_map:
            return out(attn_map[sub], data_torch)

        # dense MLP (PolyNorm)
        dense_map = {
            "mlp.gate_proj.weight": (T.FFN_GATE, ".weight"),
            "mlp.up_proj.weight":   (T.FFN_UP,   ".weight"),
            "mlp.down_proj.weight": (T.FFN_DOWN, ".weight"),
            "mlp.act_fn.weight":    (T.FFN_POLY, ".weight"),
            "mlp.act_fn.bias":      (T.FFN_POLY, ".bias"),
        }
        if sub in dense_map:
            t, suffix = dense_map[sub]
            return out(t, data_torch, suffix)

        # MoE
        if sub == "moe.router.gate.weight":
            return out(T.FFN_GATE_INP, data_torch)
        if sub == "moe.expert_bias":
            return out(T.FFN_EXP_PROBS_B, data_torch, ".bias")
        if sub == "moe.experts.gate_up_proj":
            # [n_expert, 2 * moe_intermediate, n_embd] -> split into gate / up
            n_ff = data_torch.shape[1] // 2
            return [
                (self.format_tensor_name(T.FFN_GATE_EXP, bid), data_torch[:, :n_ff, :].contiguous()),
                (self.format_tensor_name(T.FFN_UP_EXP,   bid), data_torch[:, n_ff:, :].contiguous()),
            ]
        if sub == "moe.experts.down_proj":
            return out(T.FFN_DOWN_EXP, data_torch)
        if sub == "moe.experts.act_fn.weight":
            return out(T.FFN_POLY_EXPS, data_torch)         # [n_expert, 3]
        if sub == "moe.experts.act_fn.bias":
            return out(T.FFN_POLY_EXPS, data_torch, ".bias")  # [n_expert, 1]
        shexp_map = {
            "moe.shared_experts.gate_proj.weight": (T.FFN_GATE_SHEXP, ".weight"),
            "moe.shared_experts.up_proj.weight":   (T.FFN_UP_SHEXP,   ".weight"),
            "moe.shared_experts.down_proj.weight": (T.FFN_DOWN_SHEXP, ".weight"),
            "moe.shared_experts.act_fn.weight":    (T.FFN_POLY_SHEXP, ".weight"),
            "moe.shared_experts.act_fn.bias":      (T.FFN_POLY_SHEXP, ".bias"),
        }
        if sub in shexp_map:
            t, suffix = shexp_map[sub]
            return out(t, data_torch, suffix)

        # mHC
        for which, t_norm, t_pre, t_post, t_res, t_alpha in (
            ("mhc_attn", T.MHC_ATTN_NORM, T.MHC_ATTN_PRE, T.MHC_ATTN_POST, T.MHC_ATTN_RES, T.MHC_ATTN_ALPHA),
            ("mhc_ffn",  T.MHC_FFN_NORM,  T.MHC_FFN_PRE,  T.MHC_FFN_POST,  T.MHC_FFN_RES,  T.MHC_FFN_ALPHA),
        ):
            if not sub.startswith(which + "."):
                continue
            field = sub[len(which) + 1:]
            simple = {
                "rms_norm.weight":  (t_norm, ".weight"),
                "proj_pre.weight":  (t_pre,  ".weight"),
                "bias_pre":         (t_pre,  ".bias"),
                "proj_post.weight": (t_post, ".weight"),
                "bias_post":        (t_post, ".bias"),
                "proj_res.weight":  (t_res,  ".weight"),
                "bias_res":         (t_res,  ".bias"),
            }
            if field in simple:
                t, suffix = simple[field]
                return out(t, data_torch, suffix)
            if field in ("alpha_pre", "alpha_post", "alpha_res"):
                key = f"{bid}.{which}"
                store = self._mhc_alpha.setdefault(key, {})
                store[field] = data_torch.reshape(1)
                if len(store) == 3:
                    import torch
                    alpha = torch.cat([store["alpha_pre"], store["alpha_post"], store["alpha_res"]], dim=0)
                    del self._mhc_alpha[key]
                    return out(t_alpha, alpha)
                return []

        raise ValueError(f"unmapped tensor: {name}")

    def prepare_tensors(self):
        super().prepare_tensors()
        if self._mhc_alpha:
            raise ValueError(f"incomplete mHC alpha groups: {list(self._mhc_alpha.keys())}")
