from __future__ import annotations

from .base import ModelBase, TextModel, gguf


@ModelBase.register("DoryForCausalLM")
class DoryModel(TextModel):
    model_arch = gguf.MODEL_ARCH.DORY
    allow_non_f32_vectors = True

    def tensor_force_quant(self, name, new_name, bid, n_dims):
        if n_dims <= 1:
            return {
                gguf.LlamaFileType.MOSTLY_BF16: gguf.GGMLQuantizationType.BF16,
                gguf.LlamaFileType.MOSTLY_F16: gguf.GGMLQuantizationType.F16,
            }.get(self.ftype, gguf.GGMLQuantizationType.F32)
        return False

    def set_vocab(self):
        self._set_vocab_gpt2()

    def set_gguf_parameters(self):
        h = self.hparams
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
            if key not in h or h[key] != value:
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
        for dest, source in (("input_layer_count", "n_input_layers"), ("recurrent_layer_count", "n_recurrent_layers"), ("output_layer_count", "n_output_layers"), ("recurrent_loop_count", "n_recurrent_loops")):
            w.add_uint32(f"dory.{dest}", h[source])
        w.add_string("dory.layer_pattern", pattern)
        w.add_string("dory.recurrent_kv_cache_mode", cache_mode)
        w.add_array("dory.rope.profile", [0 if no_rope[i] else p for i, p in enumerate(profiles)])
        w.add_float32("dory.rope.freq_base_2", h["rope_theta_2"])
        w.add_float32("dory.init_std", h["init_method_std"])
        w.add_float32("dory.alpha_init", h["ngpt_alpha_init"])
        w.add_float32("dory.gate_init", h["attention_output_gate_logit_scale_init"] or h["hidden_size"]**0.5)

    def modify_tensors(self, data_torch, name, bid):
        # Scale vectors have no suffix in the HF export.
        mapped = self.map_tensor_name(name)
        if not mapped.endswith(".weight"):
            mapped += ".weight"
        yield mapped, data_torch
