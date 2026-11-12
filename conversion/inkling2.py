from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor

from .base import ModelBase, TextModel, gguf, logger


def _is_inkling2(dir_model: Path) -> bool:
    if not (dir_model / "config.json").is_file():
        return False
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return json.load(f).get("model_type") in ("inkling2", "inkling2_text")


@ModelBase.register_hparams_loader(_is_inkling2)
def _load_inkling2_hparams(dir_model: Path):
    # AutoConfig injects MXFP8 defaults and can replace the original layer schedule.
    with open(dir_model / "config.json", encoding="utf-8") as f:
        return json.load(f)


@ModelBase.register("Inkling2ForConditionalGeneration", "Inkling2ForCausalLM")
@ModelBase.example("thinkingmachines/Inkling-2-Small")
class Inkling2Model(TextModel):
    model_arch = gguf.MODEL_ARCH.INKLING2

    def __init__(self, dir_model: Path, *args, **kwargs):
        quant_path = dir_model / "hf_quant_config.json"
        if quant_path.is_file():
            with open(quant_path, encoding="utf-8") as f:
                groups = json.load(f).get("model", {}).get("parameter_groups", [])
            schemes = sorted({g.get("weight_transform", "unknown") for g in groups} - {"cast"})
            if schemes:
                raise ValueError(f"Inkling2: unsupported packed expert quantization {schemes}; NVFP4 amax_denominator and MXFP8 block-scale layouts are not verified. Supply dequantized floating-point weights.")
        super().__init__(dir_model, *args, **kwargs)
        logger.warning("Inkling2: converting the text backbone only; vision and MTP tensors are skipped")
        hp = self.hparams
        for name, default in (("q_bias", False), ("o_bias", False), ("n_shared_experts", 0), ("use_value_norm_no_gamma", True), ("attn_res_normalize_values", True), ("scoring_func", "sigmoid")):
            if hp.get(name, default) != default:
                raise ValueError(f"Inkling2: unsupported {name}={hp[name]!r}")
        types = hp.get("layer_types")
        if types is None:
            local = hp.get("local_layer_ids", [i for i in range(self.block_count) if (i + 1) % 6])
            types = ["hybrid_sliding" if i in local else "hybrid" for i in range(self.block_count)]
        if len(types) != self.block_count or set(types) - {"hybrid", "hybrid_sliding"}:
            raise ValueError("Inkling2: invalid layer_types")
        self.is_swa = [t == "hybrid_sliding" for t in types]

    @classmethod
    def filter_tensors(cls, item: tuple[str, Callable[[], Tensor]]) -> tuple[str, Callable[[], Tensor]] | None:
        name, gen = item
        if any(p in name for p in ("visual.", "vision_tower.", "mtp.", "markov_head", "nextn_predict")):
            return None
        name = name.replace("model.language_model.", "model.").replace("model.llm.", "model.")
        name = name.replace("model.unembed.weight", "lm_head.weight")
        name = name.replace("model.embed.weight", "model.embed_tokens.weight")
        name = name.replace("model.embed_norm.", "model.embed_tokens.embed_norm.")
        name = re.sub(r"attn_res\.attn_queries\.(\d+)", lambda m: f"layers.{int(m[1]) + 1}.attention_input.query", name)
        name = re.sub(r"attn_res\.mlp_queries\.(\d+)", r"layers.\1.mlp_input.query", name)
        name = name.replace("attn_res.output_query", "final_input.query")
        for old, new in (("wq_du", "q_proj"), ("wk_dv", "k_proj"), ("wv_dv", "v_proj"), ("wr_du", "r_proj"), ("wo_ud", "o_proj"), ("gate_attn", "gate_proj"), ("attn_sink_proj", "sink_proj")):
            name = name.replace(f".attn.{old}", f".self_attn.{new}")
        name = name.replace(".attn.", ".self_attn.")
        name = name.replace("attn_sconv.", "post_attention_convolution.").replace("mlp_sconv.", "post_mlp_convolution.")
        name = re.sub(r"(convolution|sconv)\.weight$", r"\1.conv1d.weight", name)
        name = re.sub(r"(convolution|sconv)\.static_w$", r"\1.static_weight", name)
        name = re.sub(r"(convolution|sconv)\.weight_proj$", r"\1.weight_proj.weight", name)
        name = name.replace(".mlp_norm.", ".post_attention_layernorm.").replace(".attn_norm.", ".input_layernorm.")
        name = re.sub(r"\.(q|k|v)_input_norm\.", r".\1_input_layernorm.", name)
        name = name.replace("mlp.gate.bias", "mlp.gate.e_score_correction_bias")
        return name, gen

    def get_vocab_base(self):
        tokens, types, pre = super().get_vocab_base()
        n = self.hparams.get("unpadded_vocab_size", 200058)
        return tokens[:n], types[:n], pre

    def set_vocab(self):
        tokens, types, pre = self.get_vocab_base()
        self.gguf_writer.add_tokenizer_model("gpt2")
        self.gguf_writer.add_tokenizer_pre(pre)
        self.gguf_writer.add_token_list(tokens)
        self.gguf_writer.add_token_types(types)
        special = gguf.SpecialVocab(self.dir_model, load_merges=True)
        eos = self.hparams.get("eos_token_id", 200006)
        special.special_token_ids["eos"] = eos
        special.special_token_ids["eot"] = eos
        special.add_special_token["bos"] = False
        special.add_to_gguf(self.gguf_writer)

    def set_gguf_parameters(self):
        hp = self.hparams
        hp["intermediate_size"] = hp.get("moe_intermediate_size", hp.get("intermediate_size", 2304))
        hp["num_attention_heads"] = [hp.get("swa_num_attention_heads", hp["num_attention_heads"]) if s else hp["num_attention_heads"] for s in self.is_swa]
        hp["num_key_value_heads"] = [hp.get("swa_num_key_value_heads", hp["num_key_value_heads"]) if s else hp["num_key_value_heads"] for s in self.is_swa]
        hp.setdefault("n_routed_experts", 384)
        hp.setdefault("num_experts_per_tok", 8)
        super().set_gguf_parameters()
        w = self.gguf_writer
        w.add_vocab_size(hp["vocab_size"])
        w.add_key_length_swa(hp.get("swa_head_dim", hp["head_dim"]))
        w.add_value_length_swa(hp.get("swa_head_dim", hp["head_dim"]))
        w.add_rope_dimension_count(0)
        w.add_sliding_window(hp.get("sliding_window_size", 1024))
        w.add_sliding_window_pattern(self.is_swa)
        w.add_shortconv_l_cache(hp.get("conv_kernel_size", hp.get("sconv_kernel_size", 4)))
        w.add_expert_weights_scale(hp.get("route_scale", 8.0))
        w.add_activation_situ_beta(hp.get("mlp_gate_softcap", 4.0))
        w.add_activation_situ_linear_beta(hp.get("mlp_up_softcap", 25.0))
        w.add_attn_res_block_size(hp.get("n_layers_per_attn_res_block", 6))
        w.add_final_logit_softcapping(hp.get("final_logit_softcapping", 30.0))
        for key, field, default in (
            ("ATTENTION_RELATIVE_LENGTH", "d_rel", 16),
            ("ATTENTION_RELATIVE_EXTENT", "rel_extent", 1024),
            ("ATTENTION_QK_CLIP", "qk_l2_clip_threshold", 22.627416997969522),
            ("ATTENTION_RELATIVE_CLIP", "rel_l2_clip_threshold", 8.0),
            ("ATTENTION_LOG_SCALE_FLOOR", "log_scaling_n_floor", 128000),
            ("ATTENTION_LOG_SCALE_ALPHA", "log_scaling_alpha", 0.1),
            ("VOCAB_SIZE_UNPADDED", "unpadded_vocab_size", 200058),
        ):
            value = hp.get(field, default)
            add = w.add_float32 if isinstance(default, float) else w.add_uint32
            add(getattr(gguf.Keys.LLM, key).format(arch="inkling2"), value)

    def tensor_force_quant(self, name: str, new_name: str, bid: int | None, n_dims: int) -> gguf.GGMLQuantizationType | bool:
        if new_name.endswith(("_conv.weight", "_clip.weight", "_res_query.weight", "attn_rel_bias.weight")):
            return gguf.GGMLQuantizationType.F32
        return super().tensor_force_quant(name, new_name, bid, n_dims)

    def modify_tensors(self, data_torch: Tensor, name: str, bid: int | None) -> Iterable[tuple[str, Tensor]]:
        if not data_torch.dtype.is_floating_point or data_torch.element_size() == 1 or "_scale" in name or "_amax" in name:
            raise ValueError(f"Inkling2: unsupported packed expert tensor {name!r}; supply dequantized floating-point weights")
        if name.endswith(".w13_weight"):
            shape = data_torch.shape
            data_torch = data_torch.reshape(shape[0], shape[1] // 2, 2, shape[2]).transpose(1, 2).reshape(shape)
            name = name.replace(".w13_weight", ".gate_up_proj")
        name = name.replace(".w2_weight", ".down_proj")
        if name.endswith(".static_weight"):
            data_torch = data_torch.flip(0).T.contiguous()
            name = name.replace(".static_weight", ".conv1d.weight")
        elif name.endswith(".conv1d.weight"):
            data_torch = data_torch.squeeze(1) if data_torch.ndim == 3 else data_torch
        elif name.endswith(".rel_logits_proj.proj"):
            data_torch = data_torch.T.contiguous()
        if name.endswith((".query", ".scale", ".proj", ".gate_up_proj", ".down_proj", ".e_score_correction_bias")):
            new_name = self.map_tensor_name(name, try_suffixes=())
            suffix = ".bias" if name.endswith(".e_score_correction_bias") else ".weight"
            yield new_name + suffix, data_torch
        else:
            yield from super().modify_tensors(data_torch, name, bid)
