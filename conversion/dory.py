from __future__ import annotations

from .base import ModelBase, TextModel, gguf


@ModelBase.register("DoryForCausalLM")
@ModelBase.example("nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16")
class DoryModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DORY

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if n_dims == 1 and any(self.match_model_tensor_name(new_name, key, bid) for key in (
            gguf.MODEL_TENSOR.RESIDUAL_ALPHA,
            gguf.MODEL_TENSOR.ATTN_QK_SCALE,
            gguf.MODEL_TENSOR.ATTN_GATE_SCALE,
            gguf.MODEL_TENSOR.FFN_GATE_SCALE,
            gguf.MODEL_TENSOR.FFN_UP_SCALE,
            gguf.MODEL_TENSOR.OUTPUT_SCALE,
        )):
            return {
                gguf.LlamaFileType.MOSTLY_BF16: gguf.GGMLQuantizationType.BF16,
                gguf.LlamaFileType.MOSTLY_F16: gguf.GGMLQuantizationType.F16,
            }.get(self.ftype, gguf.GGMLQuantizationType.F32)
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        h = self.hparams.copy()
        public_config = "layer_types" in h
        if public_config:
            layer_types = h["layer_types"]
            patterns = {"sliding_attention": "S", "full_attention": "*", "mlp": "-"}
            if not isinstance(layer_types, list) or len(layer_types) != h["num_hidden_layers"] or any(not isinstance(t, str) or t not in patterns for t in layer_types):
                raise ValueError("Invalid Dory layer_types")
            rope = h["rope_parameters"]
            for kind in ("full_attention", "sliding_attention"):
                if rope[kind].get("rope_type") != "default" or rope[kind].get("partial_rotary_factor", 1.0) != 1.0:
                    raise ValueError("Dory port requires default, full-head RoPE")
            # Public configs use shared attention dimensions and per-type RoPE settings.
            converted = {
                "attention_head_dim": h["head_dim"],
                "swa_head_dim": h["head_dim"],
                "swa_num_attention_heads": h["num_attention_heads"],
                "swa_num_key_value_heads": h["num_key_value_heads"],
                "init_method_std": h["initializer_range"],
                "mlp_hidden_act": h["hidden_act"],
                "swa_window_size": h["sliding_window"],
                "hybrid_layer_pattern": "".join(patterns[t] for t in layer_types),
                "rope_profile_layers": [2 if t == "sliding_attention" else 1 for t in layer_types],
                "no_rope_layers": [0] * len(layer_types),
                "rope_theta": rope["full_attention"]["rope_theta"],
                "rope_theta_2": rope["sliding_attention"]["rope_theta"],
            }
            for key, value in converted.items():
                if key in h and h[key] != value:
                    raise ValueError(f"Conflicting Dory configuration for {key}")
            h.update(converted)
        required = {
            "dory_format_version": 1,
            "ngpt": True, "fngpt": True, "ngpt_enable_final_norm": False,
            "attention_output_gate": True, "mlp_hidden_act": "silu",
            "attention_bias": False, "mlp_bias": False, "initial_state": "input_copy",
            "input_transform": "residual", "output_transform": "replace",
            "partial_rotary_factor": 1.0, "partial_rotary_factor_2": 1.0,
            "mixer_type": None,
        }
        for key, value in required.items():
            if (key in h and h[key] != value) or (not public_config and key not in h):
                raise ValueError(f"Dory port requires {key}={value!r}")
        cache_mode = h.get("recurrent_kv_cache_mode", "per_loop")
        if cache_mode not in ("per_loop", "last_loop"):
            raise ValueError("Dory recurrent_kv_cache_mode must be per_loop or last_loop")
        if h.get("tie_word_embeddings", False):
            raise ValueError("Dory requires untied embeddings")
        n = h["num_hidden_layers"]
        pattern = h["hybrid_layer_pattern"]
        if len(pattern) != n or set(pattern) - set("S*-"):
            raise ValueError("Unsupported Dory layer pattern")
        if sum(h[k] for k in ("n_input_layers", "n_recurrent_layers", "n_output_layers")) != n:
            raise ValueError("Dory physical layer counts disagree")
        if h["swa_num_attention_heads"] != h["num_attention_heads"] or h["swa_num_key_value_heads"] != h["num_key_value_heads"] or h["swa_head_dim"] != h["attention_head_dim"]:
            raise ValueError("Dory port requires equal SWA/global head dimensions")
        profiles, no_rope = h["rope_profile_layers"], h["no_rope_layers"]
        if len(profiles) != n or len(no_rope) != n or any(p not in (1, 2) for p in profiles) or any(no_rope):
            raise ValueError("Invalid Dory RoPE profile")
        if self.fuse_qkv:
            raise ValueError("Dory Q4_K_M requires separate Q/K/V tensor roles; omit --fuse-qkv")
        super().set_gguf_parameters()
        w = self.gguf_writer
        w.add_key_length(h["attention_head_dim"])
        w.add_value_length(h["attention_head_dim"])
        w.add_rope_dimension_count(h["attention_head_dim"])
        w.add_rope_freq_base(h["rope_theta"])
        w.add_sliding_window(h["swa_window_size"])
        w.add_input_layer_count(h["n_input_layers"])
        w.add_recurrent_layer_count(h["n_recurrent_layers"])
        w.add_output_layer_count(h["n_output_layers"])
        w.add_num_loops(h["n_recurrent_loops"])
        w.add_layer_pattern(pattern)
        w.add_recurrent_kv_cache_mode(cache_mode)
        w.add_rope_profile(profiles)
        w.add_rope_freq_base_2(h["rope_theta_2"])
        w.add_init_std(h["init_method_std"])
        w.add_alpha_init(h["ngpt_alpha_init"])
        w.add_gate_init(h.get("attention_output_gate_logit_scale_init") or h["hidden_size"]**0.5)

    def modify_tensors(self, data_torch, name, bid):
        # Scale vectors have no suffix in the HF export.
        mapped = self.map_tensor_name(name)
        if not mapped.endswith(".weight"):
            mapped += ".weight"
        yield mapped, data_torch
